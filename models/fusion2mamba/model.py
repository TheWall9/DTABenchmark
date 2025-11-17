
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch_geometric import nn as gnn
from torch_geometric.utils import softmax as gnn_softmax
from torch_scatter import scatter_add
from lightning.pytorch.callbacks import EarlyStopping

import numpy as np

from toolbox import datamodule
from toolbox.featurizer.ligand import SimpleSmilesTokenFeaturizer, Mol3dGraphFeaturizer, MorganFeaturizer, \
    GraphMVPFeaturizer, GraphMVP3dPointFeaturizer, MolFeaturizerBase, UnimolFeaturizer
from toolbox.featurizer.protein import SimpleProtTokenFeaturizer, LLMStructFeaturizer, STRUCT_ROOT_DIR, LLMFeaturizer, PocketGVPFeaturizer
from toolbox.featurizer.tools import FeaturizerBase, GraphData, FeatData
from toolbox.utils import serialize3d
from toolbox import ModelBase
from models.fusion2mamba.layer import MambaEncoder


def reorder_graph(graph, node_order, reverse_order, attr_keys=None):
    ans = {}
    attr_keys = graph.attr_keys if attr_keys is None else attr_keys
    for key in attr_keys:
        data = getattr(graph, key, None)
        if data is None:
            continue
        if 'edge_index' in key:
            index_map = {o:i for i, o in enumerate(node_order)}
            new_edge_index = np.array([index_map[item] for item in data.flatten()]).reshape(data.shape)
            ans[key] = new_edge_index
        elif 'edge' in key:
            ans[key] = data
        elif len(node_order)==len(data):
            if isinstance(data, np.ndarray):
                data = data[node_order]
            else:
                data = [data[i] for i in node_order]
            ans[key] = data
        else:
            ans[key] = data
    ans['reverse_order_batch'] = reverse_order
    return GraphData(**ans)

def serailize_graph(graph, batch_key, keys):
    pos = getattr(graph, 'pos')
    batch_idx = getattr(graph, batch_key, None) if batch_key is not None else None
    forward_order, forward_inverse_order = serialize3d(pos, batch_idx, order='hilbert')
    backward_order, backward_inverse_order = serialize3d(pos, batch_idx, order='hilbert-trans')
    forward_graph = reorder_graph(graph, forward_order, forward_inverse_order, attr_keys=keys)
    backward_graph = reorder_graph(graph, backward_order, backward_inverse_order, attr_keys=keys)
    return forward_graph, backward_graph


class FusionLigandFeaturizer(FeaturizerBase):
    def __init__(self, smiles_max_lengths=100):
        super().__init__()
        self.smiles_max_lengths = smiles_max_lengths
        self.morgan_featurizer = MorganFeaturizer()
        self.smiles_featurizer = SimpleSmilesTokenFeaturizer(smiles_max_lengths=smiles_max_lengths)
        self.struct_featurizer = GraphMVP3dPointFeaturizer()

    def _featurize(self, datapoint, **kwargs):
        morgan = self.morgan_featurizer.featurize([datapoint])[0]
        tokens = self.smiles_featurizer.featurize([datapoint])[0]
        struct = self.struct_featurizer.featurize([datapoint])[0]
        forward_graph, backward_graph = serailize_graph(struct.graph, batch_key=None, keys=['raw_features', 'pos'])
        forward_graph.node_features = forward_graph.raw_features
        ans = FeatData(embedding=morgan.embedding, input_ids=tokens.input_ids,
                       graph=forward_graph, prefix='ligand')
        return ans

    def get_feat_info(self, data=None):
        ans = super().get_feat_info(data=data)
        ans['num_ligand_tokens'] = self.smiles_featurizer.tokenizer.get_vocab_size()
        return ans

class FusionProteinFeaturizer(FeaturizerBase):
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR, struct_type='esmfold', pocket_type='dogsite3', pocket_top=3, n_res_expand=10,
                 edge_num=10, rbf_num=16, pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D", feat_type='cls'):
        super().__init__()
        self.struct_root_dir = struct_root_dir
        self.struct_type = struct_type
        self.pocket_type = pocket_type
        self.pocket_top = pocket_top
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.feat_type = feat_type
        self.struct_featurizer = LLMStructFeaturizer(struct_root_dir=struct_root_dir, pocket_top=pocket_top,
                                                     struct_type=struct_type, pocket_type=pocket_type, n_res_expand=n_res_expand,
                                                     pretrained_model_name_or_path=pretrained_model_name_or_path)
        self.seq_featurizer = LLMFeaturizer(pretrained_model_name_or_path=pretrained_model_name_or_path, feat_type=feat_type)

    def _featurize(self, datapoint, **kwargs):
        struct = self.struct_featurizer.featurize([datapoint])[0]
        cls = self.seq_featurizer.featurize([datapoint])[0]
        graph = struct.graph
        graph.pos = graph.pos[:,1]
        forward_graph, backward_graph = serailize_graph(graph, batch_key='node_batch', keys=['node_features', 'pos',])
        return FeatData(embedding=cls.embedding, graph=forward_graph, prefix='protein')


class FusionDataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = FusionLigandFeaturizer
    protein_featurizer_cls = FusionProteinFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.to_pandas()
        self.to_tensor()


class PoolLayer(nn.Module):
    def __init__(self, dim_in, avg_head=False, max_head=True, attn_head=False, n_heads=8):
        super(PoolLayer, self).__init__()
        self.avg_head = avg_head
        self.max_head = max_head
        self.attn_head = attn_head
        if attn_head:
            self.query = nn.Parameter(torch.randn(dim_in, n_heads))
        self.identity = (not attn_head) and (not max_head) and (not avg_head)

    def mean_pool(self, x, mask):
        """x:BxLxD  mask:BxL"""
        if x.ndim==2:
            return gnn.global_mean_pool(x, mask.long())
        return (x*mask.unsqueeze(-1)).sum(dim=1)/mask.sum(dim=1, keepdim=True)

    def max_pool(self, x, mask):
        if x.ndim==2:
            return gnn.global_max_pool(x, mask.long())
        x = x.masked_fill(~mask.unsqueeze(-1).bool(), -torch.inf)
        h, _ = x.max(dim=1)
        return h

    def attention_pool(self, x, mask):
        scores = x@self.query
        if x.ndim==2:
            scores = gnn_softmax(scores, index=mask.long())
            # h = gnn.global_add_pool(scores*x, mask)
            value = scatter_add(x.unsqueeze(1) * scores.unsqueeze(-1), index=mask.long(), dim=0)
            h = value.mean(dim=1)
            return h
        # BxTxDx1 BxTx1xH
        scores = scores.masked_fill(~mask.unsqueeze(-1).bool(), -torch.inf)
        alpha = F.softmax(scores, dim=1)
        h = (x.unsqueeze(-1) * alpha.unsqueeze(2)).sum(dim=1).mean(dim=-1)
        return h

    def forward(self, x, mask):
        if self.identity:
            return x
        ans = []
        if self.attn_head:
            ans.append(self.attention_pool(x, mask))
        if self.avg_head:
            ans.append(self.mean_pool(x, mask))
        if self.max_head:
            ans.append(self.max_pool(x, mask))
        return torch.cat(ans, dim=-1)


class MambaEncoder3d(nn.Module):
    def __init__(self, model, input_dim, embedding_dim=128, avg_head=False, max_head=True, attn_head=False):
        super(MambaEncoder3d, self).__init__()
        self.model = model
        self.embedding_pos = nn.Sequential(nn.Linear(3, embedding_dim),
                                           nn.ReLU(),
                                           nn.BatchNorm1d(embedding_dim),
                                           nn.Linear(embedding_dim, embedding_dim),
                                           )
        self.embedding_feat = nn.Linear(input_dim, embedding_dim)
        self.pool = PoolLayer(embedding_dim, avg_head=avg_head, max_head=max_head, attn_head=attn_head)


    def forward(self, pos, pos_idx, pos_feat=None, pool_index=None):
        inputs_embeds = self.embedding_pos(pos)
        if pos_feat is not None:
            pos_feat = self.embedding_feat(pos_feat)
            inputs_embeds = inputs_embeds+pos_feat
        inputs = inputs_embeds.unsqueeze(0)
        seq_idx = pos_idx.unsqueeze(0).int()
        pool_index = pool_index if pool_index is not None else pos_idx
        outputs = self.model(inputs, seq_idx=seq_idx).squeeze(0)
        ans = self.pool(outputs, mask=pool_index)
        return ans


class MambaTokenEncoder(nn.Module):
    def __init__(self, model, hidden_dim, input_dim=None, num_embeddings=None, avg_head=False, max_head=True, attn_head=False):
        super(MambaTokenEncoder, self).__init__()
        if num_embeddings is not None:
            self.embedding = nn.Embedding(num_embeddings, hidden_dim)
        elif input_dim is not None:
            self.embedding = nn.Linear(input_dim, hidden_dim)
        self.model = model
        self.pool = PoolLayer(hidden_dim, avg_head=avg_head, max_head=max_head, attn_head=attn_head)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        seq_len, seq_idx = None, None
        if input_ids is not None:
            inputs_embeds = self.embedding(input_ids)
            if attention_mask is None:
                attention_mask = input_ids != 0
            seq_len = torch.sum(attention_mask, dim=-1)
        else:
            inputs_embeds = self.embedding(inputs_embeds).unsqueeze(0)
            seq_idx = attention_mask.unsqueeze(0).int()

        output = self.model(inputs_embeds, seq_len=seq_len, seq_idx=seq_idx)
        output = self.pool(output, attention_mask)
        return output


class CombinedDecoder(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dims=(1024, 1024, 512), dropout=0.1, norm=False):
        super().__init__()
        dims = [input_dim1+input_dim2]+list(hidden_dims)
        model = []
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            model.append(nn.Linear(dim_in, dim_out))
            if norm:
                model.append(nn.BatchNorm1d(dim_out))
            model.append(nn.ReLU(inplace=True))
            model.append(nn.Dropout(dropout))
        model.append(nn.Linear(dims[-1], 1))
        self.model = nn.Sequential(*model)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        output = self.model(x).squeeze(-1)
        return output


class LigandEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_embedding_dim = config['encoder_embedding_dim']
        encoder_output_dim = config['encoder_output_dim']
        n_layer = config['ligand_encoder_n_layers']
        mamba_version = config['mamba_version']
        avg_head = config['avg_head']
        max_head = config['max_head']
        attn_head = config['attn_head']
        bidirectional = config['bidirectional']
        dropout = config['encoder_dropout']
        d_ff = 4*encoder_embedding_dim
        mamba1 = MambaEncoder(d_model=encoder_embedding_dim, n_layer=n_layer, ssm_cfg={"layer": mamba_version},
                              d_ff=d_ff, bidirectional=bidirectional, dropout=dropout)
        mamba2 = MambaEncoder(d_model=encoder_embedding_dim, n_layer=n_layer, ssm_cfg={"layer": mamba_version},
                              d_ff=d_ff, bidirectional=bidirectional, dropout=dropout)
        mamba3 = MambaEncoder(d_model=encoder_embedding_dim, n_layer=n_layer, ssm_cfg={"layer": mamba_version},
                              d_ff=d_ff, bidirectional=bidirectional, dropout=dropout)
        mamba1.binding_ssm(mamba2)
        mamba1.binding_ssm(mamba3)
        self.encoder_0d = nn.Sequential(nn.Linear(config['ligand_embedding_dim'], encoder_embedding_dim*2),
                                        nn.GLU())
        self.encoder_1d = MambaTokenEncoder(mamba1, encoder_embedding_dim, num_embeddings=config['num_ligand_tokens'],
                                            avg_head=avg_head, max_head=max_head, attn_head=attn_head)
        self.encoder_3d = MambaEncoder3d(mamba3, config['ligand_graph_node_features_dim'], encoder_embedding_dim,
                                         avg_head=avg_head, max_head=max_head, attn_head=attn_head)
        dims = [encoder_embedding_dim]
        if attn_head:
            dims.append(encoder_embedding_dim*2)
        if avg_head:
            dims.append(encoder_embedding_dim*2)
        if max_head:
            dims.append(encoder_embedding_dim*2)
        self.output_project = nn.Linear(sum(dims), encoder_output_dim)

    def forward(self, embedding, input_ids, graph):
        embedding_0d = self.encoder_0d(embedding)
        embedding_1d = self.encoder_1d(input_ids=input_ids)
        embedding_3d = self.encoder_3d(pos=graph.pos, pos_idx=graph.batch, pos_feat=graph.node_features)
        embedding = torch.cat([embedding_0d, embedding_1d, embedding_3d], dim=-1)
        output = self.output_project(embedding)
        output = F.normalize(output, dim=-1)
        return output


class ProtEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        encoder_embedding_dim = config['encoder_embedding_dim']
        encoder_output_dim = config['encoder_output_dim']
        avg_head = config['avg_head']
        max_head = config['max_head']
        attn_head = config['attn_head']
        n_layer = config['protein_encoder_n_layers']
        mamba_version = config['mamba_version']
        bidirectional = config['bidirectional']
        dropout = config['encoder_dropout']
        d_ff = 4*encoder_embedding_dim
        mamba1 = MambaEncoder(d_model=encoder_embedding_dim, n_layer=n_layer, ssm_cfg={"layer": mamba_version},
                              d_ff=d_ff, bidirectional=bidirectional, dropout=dropout)
        # mamba2 = MambaEncoder(d_model=encoder_embedding_dim, n_layer=n_layer, ssm_cfg={"layer": mamba_version},
        #                       d_ff=d_ff, bidirectional=bidirectional, dropout=dropout)
        self.encoder_0d = nn.Sequential(nn.Linear(config['protein_embedding_dim'], encoder_embedding_dim*2),
                                        nn.GLU())
        self.encoder_3d = MambaEncoder3d(mamba1, config['protein_graph_node_features_dim'], encoder_embedding_dim,
                                         avg_head=avg_head, max_head=max_head, attn_head=attn_head)
        dims = [encoder_embedding_dim]
        if attn_head:
            dims.append(encoder_embedding_dim)
        if avg_head:
            dims.append(encoder_embedding_dim)
        if max_head:
            dims.append(encoder_embedding_dim)
        self.output_project = nn.Linear(sum(dims), encoder_output_dim)
        encoder_n_res_layers = config['encoder_n_res_layers']
        self.res_layers = nn.ModuleList()
        for i in range(encoder_n_res_layers):
            self.res_layers.append(nn.Sequential(nn.Linear(encoder_output_dim, encoder_output_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(dropout)))


    def forward(self, embedding, graph):
        embedding_0d = self.encoder_0d(embedding)
        embedding_3d = self.encoder_3d(pos=graph.pos, pos_idx=graph.batch, pos_feat=graph.node_features,
                                       pool_index=graph.batch)
        embedding = torch.cat([embedding_0d, embedding_3d], dim=-1)
        output = self.output_project(embedding)
        output = F.normalize(output, dim=-1)
        for layer in self.res_layers:
            output = layer(output)+output
        return output



class Fusion2Mamba5(ModelBase):
    dataset_cls = FusionDataset
    def __init__(self, config):
        super().__init__(config)
        encoder_output_dim = config['encoder_output_dim']
        decoder_hidden_dims = config['decoder_hidden_dims']
        decoder_dropout = config['decoder_dropout']
        decoder_use_norm = config['decoder_use_norm']
        self.ligand_encoder = LigandEncoder(config)
        self.protein_encoder = ProtEncoder(config)
        # self.ligand_encoder.encoder_1d.model.binding_ssm(self.protein_encoder.encoder_3d.model)
        self.decoder = CombinedDecoder(encoder_output_dim, encoder_output_dim, hidden_dims=decoder_hidden_dims,
                                       dropout=decoder_dropout, norm=decoder_use_norm)
        self.register_buffer('mean', torch.tensor(config['affinity_mean']))
        self.register_buffer('std', torch.tensor(config['affinity_std']))

    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        parser.add_argument("--avg_head", default=True, action="store_true")
        parser.add_argument("--max_head", default=False, action="store_true")
        parser.add_argument("--attn_head", default=False, action="store_true")
        parser.add_argument("--protein_encoder_n_layers", default=1, type=int)
        parser.add_argument("--ligand_encoder_n_layers", default=1, type=int)
        parser.add_argument("--mamba_version", default='Mamba2', type=str, choices=['Mamba1', 'Mamba2'])
        parser.add_argument("--bidirectional", default=False, action="store_true")
        parser.add_argument("--encoder_dropout", default=0.1, type=float)
        parser.add_argument('--encoder_output_dim', type=int, default=256)
        parser.add_argument('--encoder_embedding_dim', type=int, default=128)
        parser.add_argument('--decoder_hidden_dims', type=int, nargs='+', default=(2048, 1024, 512))
        parser.add_argument('--decoder_dropout', type=float, default=0.1)
        parser.add_argument('--decoder_use_norm', action='store_true')
        parser.add_argument("--early_stop_patient", type=int, default=100)
        parser.add_argument("--encoder_n_res_layers", type=int, default=4)
        parser.set_defaults(batch_size=256, lr=1e-3)


    def forward(self, ligand_embedding, ligand_input_ids, ligand_graph, protein_embedding, protein_graph):
        ligand = self.ligand_encoder(ligand_embedding, ligand_input_ids, ligand_graph)
        protein = self.protein_encoder(protein_embedding, protein_graph)
        predict = self.decoder(ligand, protein)
        return predict

    def step(self, ligand_embedding, ligand_input_ids, ligand_graph,
             protein_embedding, protein_graph, affinity):
        predict = self.forward(ligand_embedding, ligand_input_ids, ligand_graph,
                               protein_embedding, protein_graph)
        loss = F.mse_loss(predict, affinity)
        if torch.isnan(loss):
            print("WARNING: loss is NaN")
            self.trainer.should_stop = True
        return {"loss": loss, "predict": predict}

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        preds = torch.cat(self.val_metric.preds)
        self.logger.experiment.add_histogram(tag='val/preds', global_step=self.global_step, values=preds)
        if self.current_epoch==0:
            targets = torch.cat(self.val_metric.targets)
            self.logger.experiment.add_histogram(tag='val/targets', global_step=self.global_step, values=targets)


    def configure_optimizers(self):
        lr = self.config['lr']
        lr_scale = self.config['lr_scale']
        lr_gamma = self.config['lr_gamma']
        optimizer = optim.Adam(self.parameters(), lr=lr)
        lr_scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_scale*lr, max_lr=lr,
                                                   gamma=lr_gamma, mode="exp_range", step_size_up=20,
                                                   cycle_momentum=False)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "epoch",
                }}

    def configure_callbacks(self):
        early_stop_callback = EarlyStopping(
            monitor="val/loss_epoch",
            patience=self.config['early_stop_patient'],
            mode="min",
            verbose=True
        )
        return [early_stop_callback]


if __name__ == '__main__':
    from toolbox import Evaluator
    dataset_name = 'kiba_pocketdta'
    dataset_name = "kiba"
    max_epochs = 700
    ################################# 5 fold #############################################


    Evaluator(model_name='Fusion2Mamba5', deterministic=True, dataset_name=dataset_name, max_epochs=max_epochs, pocket_tops=3,
              max_head=False, attn_head=False, avg_head=True,
              monitor_metric = 'val/CI', monitor_mode='max',
              use_wandb_logger=True, wandb_project='DTA Benchmark model', wandb_online=False, comment='exp_v6/cold',
              encoder_n_res_layers=4, merge_train_val=True, n_res_expand=0,
              use_ligand_0d=True, use_ligand_1d=True, use_ligand_3d=True,
              use_protein_0d=True, use_protein_1d=False, use_protein_3d=True,
              num_workers=10).run() #

    ################################# cold  #############################################
    Evaluator(model_name='Fusion2Mamba5', deterministic=True, dataset_name=dataset_name, max_epochs=max_epochs, pocket_tops=3,
              max_head=False, attn_head=False, avg_head=True,
              monitor_metric = 'val/CI', monitor_mode='max', cv_split_type='cold_drug', cv_n_splits=4,
              use_wandb_logger=True, wandb_project='DTA Benchmark model', wandb_online=False, comment='exp_v6/cold',
              encoder_n_res_layers=4, merge_train_val=True, n_res_expand=0,
              num_workers=10).run()

    Evaluator(model_name='Fusion2Mamba5', deterministic=True, dataset_name=dataset_name, max_epochs=max_epochs, pocket_tops=3,
              max_head=False, attn_head=False, avg_head=True,
              monitor_metric = 'val/CI', monitor_mode='max', cv_split_type='cold_target', cv_n_splits=4,
              use_wandb_logger=True, wandb_project='DTA Benchmark model', wandb_online=False, comment='exp_v6/cold',
              encoder_n_res_layers=4, merge_train_val=True, n_res_expand=0,
              num_workers=10).run()

    Evaluator(model_name='Fusion2Mamba5', deterministic=True, dataset_name=dataset_name, max_epochs=max_epochs, pocket_tops=3,
              max_head=False, attn_head=False, avg_head=True,
              monitor_metric = 'val/CI', monitor_mode='max', cv_split_type='all_cold', cv_n_splits=4,
              use_wandb_logger=True, wandb_project='DTA Benchmark model', wandb_online=False, comment='exp_v6/cold',
              encoder_n_res_layers=4, merge_train_val=True, n_res_expand=0,
              num_workers=10).run()

