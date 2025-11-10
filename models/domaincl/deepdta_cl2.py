
import torch
from torch import nn, optim
from torch.nn import functional as F

from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import softmax as gnn_softmax
from torch_geometric import nn as gnn
from torch_scatter import scatter_add
from lightning.pytorch.callbacks import EarlyStopping
from toolbox import ModelBase
from models.domaincl.dataset import MorganDomainDataset, DomainDataset, PocketDomainDataset, MorganGVPDataset
from repo.gvp.models import MQAModel


class LinkAttention(nn.Module):
    def __init__(self, input_dim, n_heads):
        super(LinkAttention, self).__init__()
        self.query = nn.Linear(input_dim, n_heads)

    def forward(self, x, batch):
        score = self.query(x)
        score = gnn_softmax(score, index=batch)
        """Lx1xD  LxHx1"""
        value = scatter_add(x.unsqueeze(1)*score.unsqueeze(-1), index=batch, dim=0)
        value = value.sum(dim=1)
        return value, score

class ConvEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128, hidden_dim=32, num_layers=3, kernel_size=8):
        super().__init__()
        self.embedding = nn.Linear(input_dim, embedding_dim, bias=False)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size,]*num_layers

        dims = [embedding_dim]+[hidden_dim*i for i in range(1, num_layers+1)]
        model = []
        for dim_in, dim_out, window_size in zip(dims[:-1], dims[1:], kernel_size):
            model.append(nn.ConstantPad1d((0, window_size-1), 0.0))
            model.append(nn.Conv1d(dim_in, dim_out, window_size, stride=1, padding=0))
            model.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*model)
        # self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input_ids, pool=False):
        x = self.embedding(input_ids)
        x = x.permute(0,2,1)
        x = self.model(x)
        if pool:
            x, _ = x.max(dim=-1)
        # x = self.pool(x).squeeze(-1)
        return x

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, hidden_layers=1, act=nn.GELU, dropout=0.0):
        super(MLP, self).__init__()
        dims = [dim_in]+[dim_hidden]*hidden_layers+[dim_out]
        blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            block = []
            block.append(nn.Linear(in_dim, out_dim))
            if i!=len(dims)-2:
                block.append(act())
                block.append(nn.Dropout(dropout))
            blocks.append(nn.Sequential(*block))
        self.layers = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = x+layer(x)
        if len(self.layers)!=1:
            x = self.layers[-1](x)
        return x

class GVPEncoder(MQAModel):
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.4):
        super(GVPEncoder, self).__init__(node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, seq_in,
                                         num_layers, drop_rate)
        self.ln = nn.LayerNorm(node_h_dim[0])

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None, padding=False):
        '''
        :param h_V: tuple (s, V) of node embeddings
        :param edge_index: `torch.Tensor` of shape [2, num_edges]
        :param h_E: tuple (s, V) of edge embeddings
        :param seq: if not `None`, int `torch.Tensor` of shape [num_nodes]
                    to be embedded and appended to `h_V`
        '''
        if seq is not None:
            seq = self.W_s(seq)
            h_V = (torch.cat([h_V[0], seq], dim=-1), h_V[1])
        h_V = self.W_v(h_V)
        h_E = self.W_e(h_E)
        for layer in self.layers:
            h_V = layer(h_V, edge_index, h_E)
        out = self.W_out(h_V)

        out = self.ln(out)
        # if padding:
        #     lengths = torch.bincount(batch)
        #     out = torch.split(out, lengths.tolist())
        #     out = pad_sequence(out, batch_first=True)
        return out


class CombinedDecoder(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim=1024, hidden_layers=2, dropout=0.1, input_dim3=0):
        super().__init__()
        self.model = MLP(input_dim1+input_dim2+input_dim3, hidden_dim, 1, dropout=dropout, hidden_layers=hidden_layers)

    def forward(self, x1, x2, x3=None):
        if x3 is None:
            x = torch.cat([x1, x2], dim=-1)
        else:
            x = torch.cat([x1, x2, x3], dim=-1)
        output = self.model(x).squeeze(-1)
        return output


class GVPDTA(ModelBase):
    dataset_cls = MorganGVPDataset
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        """CombinedCategoricalModel"""
        # protein_inputs_embeds_dim = config['protein_graph_inputs_embeds_dim']
        # embedding_dim = config['encoder_embedding_dim']
        encoder_hidden_dim = config['encoder_hidden_dim']
        # num_layers = config['encoder_num_layers']

        decoder_hidden_dim = config['decoder_hidden_dim']
        decoder_num_layers = config['decoder_num_layers']
        decoder_dropout = config['decoder_dropout']

        protein_node_s_dim = config['protein_graph_node_s_dim']
        protein_node_v_dim = config['protein_graph_node_v_dim']
        protein_edge_s_dim = config['protein_graph_edge_s_dim']
        protein_edge_v_dim = config['protein_graph_edge_v_dim']

        protein_struct_encoder_dropout = config['protein_struct_encoder_dropout']
        protein_struct_encoder_num_layers = config['protein_struct_encoder_num_layers']
        protein_struct_encoder_node_h_dim = config['protein_struct_encoder_node_h_dim']
        protein_struct_encoder_edge_h_dim = config['protein_struct_encoder_edge_h_dim']

        self.protein_encoder = GVPEncoder(node_in_dim=(protein_node_s_dim, protein_node_v_dim),
                                          node_h_dim=protein_struct_encoder_node_h_dim,
                                          edge_in_dim=(protein_edge_s_dim, protein_edge_v_dim),
                                          edge_h_dim=protein_struct_encoder_edge_h_dim,
                                          num_layers=protein_struct_encoder_num_layers,
                                          drop_rate=protein_struct_encoder_dropout,
                                          seq_in=True)

        self.ligand_encoder = nn.Sequential(nn.Linear(config['ligand_embedding_dim'], encoder_hidden_dim),
                                            # nn.ReLU(inplace=True),
                                            # nn.Linear(config['encoder_embedding_dim'], config['encoder_hidden_dim']),
                                            # nn.ReLU(inplace=True),
                                            # nn.Linear(config['encoder_hidden_dim'], config['encoder_hidden_dim']*2),
                                            nn.ReLU(inplace=True),
                                            # nn.Linear(config['encoder_embedding_dim'], config['encoder_hidden_dim']*3),
                                            )
        self.affinity_decoder = CombinedDecoder(protein_struct_encoder_node_h_dim[0], encoder_hidden_dim,
                                                decoder_hidden_dim, decoder_num_layers, decoder_dropout)
        if config['use_att_pool']:
            self.att_pool = LinkAttention(protein_struct_encoder_node_h_dim[0], 8)
        else:
            self.att_pool = None
        self.use_att_pool = config['use_att_pool']


    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        # parser.add_argument("--protein_window_size", type=int, default=8)
        # parser.add_argument("--encoder_embedding_dim", type=int, default=128)
        parser.add_argument("--encoder_hidden_dim", type=int, default=128)
        # parser.add_argument("--encoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dim", type=int, default=1024)
        parser.add_argument("--decoder_num_layers", type=int, default=2)
        parser.add_argument("--decoder_dropout", type=float, default=0.1)
        parser.add_argument("--use_att_pool", action="store_true")

        parser.add_argument("--protein_struct_encoder_dropout", default=0.4, type=float)
        parser.add_argument("--protein_struct_encoder_num_layers", default=1, type=int)
        parser.add_argument('--protein_struct_encoder_node_h_dim', default=(128, 3), type=int, nargs='+')
        parser.add_argument('--protein_struct_encoder_edge_h_dim', default=(32, 1), type=int, nargs='+')

    @classmethod
    def generate_optuna_params(cls, trial):
        protein_window_size = trial.suggest_categorical("protein_window_size", [4, 8, 12])
        ligand_window_size = trial.suggest_categorical("ligand_window_size", [4, 6, 8])
        return {"protein_window_size": protein_window_size,
                "ligand_window_size": ligand_window_size}

    def forward(self, ligand_embedding, protein_graph):
        # protein_inputs_embeds = protein_graph.inputs_embeds
        # batch = protein_graph.inputs_embeds_batch
        # mask = protein_graph.domain_mask.bool()
        # protein_inputs_embeds = protein_inputs_embeds[mask]
        # batch = batch[mask]
        # inputs_embeds, _ = to_dense_batch(protein_inputs_embeds, batch)
        # protein = self.protein_encoder(inputs_embeds, pool=not self.use_attn_pool)

        protein_embeds = self.protein_encoder((protein_graph.node_s, protein_graph.node_v),
                                               protein_graph.edge_index,
                                               (protein_graph.edge_s, protein_graph.edge_v),
                                               seq=protein_graph.seq,
                                               batch=protein_graph.batch)
        if self.use_att_pool:
            protein = self.att_pool(protein_embeds, protein_graph.batch)
        else:
            protein = gnn.global_max_pool(protein_embeds, protein_graph.batch)

        ligand = self.ligand_encoder(ligand_embedding)
        predict = self.affinity_decoder(protein, ligand)
        return predict

    def step(self, ligand_embedding, protein_graph, affinity):
        predict = self.forward(ligand_embedding, protein_graph)
        loss = F.mse_loss(predict, affinity)
        return {"loss": loss, "predict": predict}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer


class SoftContrastiveLoss(nn.Module):
    def __init__(self, input_dim, alpha, tau, use_project_head=False):
        super(SoftContrastiveLoss, self).__init__()
        self.register_buffer("alpha", torch.tensor(alpha))
        self.register_buffer("tau", torch.tensor(tau))
        if use_project_head:
            self.mlp = nn.Linear(input_dim, input_dim)
        else:
            self.mlp = None

    def forward(self, z1, z2, ids, eps=1e-8):
        if self.mlp is not None:
            z1 = self.mlp(z1)
            z2 = self.mlp(z2)
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        sim = torch.matmul(z1, z2.t()) / self.tau

        # 基于ID的基本mask
        id_mask = (ids.unsqueeze(1) == ids.unsqueeze(0)).float()

        # soft权重：相似度越高，越接近正样本
        soft_weight = torch.softmax(sim, dim=-1)
        weight = torch.clamp(id_mask + (1 - id_mask) * soft_weight, 0, 1)

        # weight = id_mask/id_mask.sum(dim=-1, keepdim=True)
        exp_sim = torch.exp(sim-sim.max(dim=1, keepdim=True)[0])
        numerator = (exp_sim * weight).sum(dim=1)
        denominator = exp_sim.sum(dim=1)
        loss = -torch.log((numerator / (denominator + eps)) + eps)
        return loss.mean()

class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, hidden_layers=1, act=nn.GELU, dropout=0.0):
        super(MLP, self).__init__()
        dims = [dim_in]+[dim_hidden]*hidden_layers+[dim_out]
        blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            block = []
            block.append(nn.Linear(in_dim, out_dim))
            if i!=len(dims)-2:
                block.append(act())
                block.append(nn.Dropout(dropout))
            blocks.append(nn.Sequential(*block))
        self.layers = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.layers[0](x)
        for layer in self.layers[1:-1]:
            x = x+layer(x)
        if len(self.layers)!=1:
            x = self.layers[-1](x)
        return x

if __name__ == '__main__':
    from toolbox import Evaluator
    # Evaluator(model_name='DomainDTACL2', deterministic=True, dataset_name='davis', max_epochs=200, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max', split_type='cv', cv_n_splits=10,
    #           merge_train_val=True, batch_size=256, comment='debug2', use_project_head=True, use_llm=True,
    #           num_workers=12).run(debug=True)
    # exit()
    # Evaluator(model_name='GVPDTA', deterministic=True, dataset_name='davis_domain', max_epochs=300, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max', pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
    #           decoder_dropout=0.1,
    #           use_attn_pool=True, pocket_top=3, pocket_type=None,
    #           protein_struct_encoder_dropout=0.1, edge_num=5, protein_struct_encoder_num_layers=1,
    #           merge_train_val=True, batch_size=256, comment='debug4', use_project_head=True, use_llm=True,
    #           num_workers=12).run(debug=True)

    # Evaluator(model_name='GVPDTA', deterministic=True, dataset_name='davis', max_epochs=300, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max', pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
    #           decoder_dropout=0.1,
    #           use_attn_pool=False, pocket_top=3, pocket_type='dogsite3', n_res_expand=15,
    #           protein_struct_encoder_dropout=0.1, edge_num=5, protein_struct_encoder_num_layers=1,
    #           merge_train_val=True, batch_size=256, comment='debug5', use_project_head=True, use_llm=True,
    #           num_workers=12).run(debug=True)
    #
    # Evaluator(model_name='GVPDTA', deterministic=True, dataset_name='davis', max_epochs=300, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max', pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
    #           decoder_dropout=0.1,
    #           use_attn_pool=False, pocket_top=3, pocket_type='dogsite3', n_res_expand=20,
    #           protein_struct_encoder_dropout=0.1, edge_num=5, protein_struct_encoder_num_layers=1,
    #           merge_train_val=True, batch_size=256, comment='debug5', use_project_head=True, use_llm=True,
    #           num_workers=12).run(debug=True)

    Evaluator(model_name='GVPDTA', deterministic=True, dataset_name='davis', max_epochs=300, lr=1e-3,
              use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
              monitor_metric = 'val/CI', monitor_mode='max', pretrained_model_name_or_path=None,
              decoder_dropout=0.1,
              use_attn_pool=False, pocket_top=3, pocket_type='dogsite3', n_res_expand=10,
              protein_struct_encoder_dropout=0.1, edge_num=5, protein_struct_encoder_num_layers=1,
              merge_train_val=True, batch_size=256, comment='debug6',
              num_workers=12).run()

    Evaluator(model_name='GVPDTA', deterministic=True, dataset_name='kiba', max_epochs=300, lr=1e-3,
              use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
              monitor_metric = 'val/CI', monitor_mode='max', pretrained_model_name_or_path=None,
              decoder_dropout=0.1,
              use_attn_pool=False, pocket_top=3, pocket_type='dogsite3', n_res_expand=10,
              protein_struct_encoder_dropout=0.1, edge_num=5, protein_struct_encoder_num_layers=1,
              merge_train_val=True, batch_size=256, comment='debug6',
              num_workers=12).run(debug=True)

    Evaluator(model_name='GVPDTA', deterministic=True, dataset_name='kiba', max_epochs=300, lr=1e-3,
              use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
              monitor_metric = 'val/CI', monitor_mode='max', pretrained_model_name_or_path=None,
              decoder_dropout=0.1,
              use_attn_pool=False, pocket_top=3, pocket_type='dogsite3', n_res_expand=10,
              protein_struct_encoder_dropout=0.1, edge_num=5, protein_struct_encoder_num_layers=1,
              merge_train_val=False, batch_size=256, comment='debug6',
              num_workers=12).run(debug=True)

    Evaluator(model_name='GVPDTA', deterministic=True, dataset_name='davis', max_epochs=300, lr=1e-3,
              use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
              monitor_metric = 'val/CI', monitor_mode='max', pretrained_model_name_or_path=None,
              decoder_dropout=0.1,
              use_attn_pool=False, pocket_top=3, pocket_type='dogsite3', n_res_expand=10,
              protein_struct_encoder_dropout=0.1, edge_num=5, protein_struct_encoder_num_layers=1,
              merge_train_val=False, batch_size=256, comment='debug6',
              num_workers=12).run()
    exit()




    # Evaluator(model_name='GVPDTA', deterministic=True, dataset_name='davis', max_epochs=300, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max', pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
    #           decoder_dropout=0.1,
    #           use_attn_pool=False, pocket_top=3, pocket_type='fpocket', n_res_expand=10,
    #           protein_struct_encoder_dropout=0.1, edge_num=5, protein_struct_encoder_num_layers=1,
    #           merge_train_val=True, batch_size=256, comment='debug5', use_project_head=True, use_llm=True,
    #           num_workers=12).run(debug=True)
    exit()
    Evaluator(model_name='GVPDTA', deterministic=True, dataset_name='davis_domain', max_epochs=300, lr=1e-3,
              use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
              monitor_metric = 'val/CI', monitor_mode='max', pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
              decoder_dropout=0.1,
              use_attn_pool=True, pocket_top=3, pocket_type=None,
              protein_struct_encoder_dropout=0.1, edge_num=5, protein_struct_encoder_num_layers=1,
              merge_train_val=False, batch_size=256, comment='debug4', use_project_head=True, use_llm=True,
              num_workers=12).run(debug=True)

    Evaluator(model_name='GVPDTA', deterministic=True, dataset_name='davis_domain', max_epochs=300, lr=1e-3,
              use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
              monitor_metric = 'val/CI', monitor_mode='max', pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
              decoder_dropout=0.1,
              use_attn_pool=False, pocket_top=3, pocket_type=None,
              protein_struct_encoder_dropout=0.1, edge_num=5, protein_struct_encoder_num_layers=1,
              merge_train_val=False, batch_size=256, comment='debug4', use_project_head=True, use_llm=True,
              num_workers=12).run(debug=True)
    exit()
    # Evaluator(model_name='MorganDTA', deterministic=True, dataset_name='davis_domain', max_epochs=200, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max',
    #           merge_train_val=True, batch_size=256, comment='debug',
    #           num_workers=12).run(debug=True)
    # exit()
    # Evaluator(model_name='DomainDTACL', deterministic=True, dataset_name='davis', max_epochs=200, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max',
    #           merge_train_val=True, batch_size=256, comment='debug2',
    #           use_project_head=True, use_llm=True, pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
    #           num_workers=12).run(debug=True) #0.9047
    #
    # Evaluator(model_name='DomainDTACL', deterministic=True, dataset_name='davis', max_epochs=200, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max',
    #           merge_train_val=True, batch_size=256, comment='debug2',
    #           use_attn_pool=True, encoder_hidden_dim=32,
    #           use_project_head=True, use_llm=True, pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
    #           num_workers=12).run(debug=True) #0.9061
    #
    # Evaluator(model_name='DomainDTACL', deterministic=True, dataset_name='davis', max_epochs=200, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max',
    #           merge_train_val=True, batch_size=256, comment='debug2',
    #           use_attn_pool=True, encoder_hidden_dim=64,
    #           use_project_head=True, use_llm=True, pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
    #           num_workers=12).run(debug=True) # 0.9059
    #
    # Evaluator(model_name='DomainDTACL', deterministic=True, dataset_name='davis', max_epochs=200, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max',
    #           merge_train_val=True, batch_size=256, comment='debug2',
    #           use_attn_pool=True, encoder_hidden_dim=32, decoder_dropout=0.4,
    #           use_project_head=True, use_llm=True, pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
    #           num_workers=12).run(debug=True) # 0.9063
    #
    # Evaluator(model_name='DomainDTACL', deterministic=True, dataset_name='davis', max_epochs=200, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max',
    #           merge_train_val=True, batch_size=256, comment='debug2',
    #           use_attn_pool=True, encoder_hidden_dim=64, decoder_dropout=0.4,
    #           use_project_head=True, use_llm=True, pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
    #           num_workers=12).run(debug=True) # 0.9087
    #
    # Evaluator(model_name='DomainDTACL', deterministic=True, dataset_name='davis', max_epochs=200, lr=1e-3,
    #           use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
    #           monitor_metric = 'val/CI', monitor_mode='max',
    #           merge_train_val=True, batch_size=256, comment='debug2',
    #           use_attn_pool=True, encoder_hidden_dim=32, decoder_dropout=0.4, decoder_hidden_dims=(2048, 1024, 512),
    #           use_project_head=True, use_llm=True, pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
    #           num_workers=12).run(debug=True) # mse 3.012


    Evaluator(model_name='DomainDTACL', deterministic=True, dataset_name='davis', max_epochs=200, lr=1e-3,
              use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
              monitor_metric = 'val/CI', monitor_mode='max',
              merge_train_val=True, batch_size=256, comment='debug2', decoder_hidden_dims=(2048, 512),
              use_project_head=True, use_llm=True, pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
              num_workers=12).run(debug=True) #

    Evaluator(model_name='DomainDTACL', deterministic=True, dataset_name='davis', max_epochs=200, lr=1e-3,
              use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
              monitor_metric = 'val/CI', monitor_mode='max',
              merge_train_val=True, batch_size=256, comment='debug2',
              use_attn_pool=True, encoder_hidden_dim=32, decoder_hidden_dims=(2048, 512),
              use_project_head=True, use_llm=True, pretrained_model_name_or_path="facebook/esm2_t6_8M_UR50D",
              num_workers=12).run(debug=True) #
