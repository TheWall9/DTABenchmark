import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pad_sequence
from torch_geometric import nn as gnn
from torch_geometric.utils import to_dense_batch
from repo.gvp.models import MQAModel
from repo.drugban.ban import BANLayer

from toolbox import ModelBase
from benchmark.pocketdta.dataset import PocketDTADataset, PocketDTADataset2
from benchmark.pocketdta.lookahead import Lookahead
from benchmark.pocketdta.Radam import RAdam
from benchmark.deepdta.deepdta import CombinedDecoder


class EmbeddingEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EmbeddingEncoder, self).__init__()
        self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(hidden_dim, output_dim),
                                   )
    def forward(self, x):
        return self.model(x)


class GVPEncoder(MQAModel):
    def __init__(self, node_in_dim, node_h_dim, edge_in_dim, edge_h_dim,
                 seq_in=False, num_layers=3, drop_rate=0.4):
        super(GVPEncoder, self).__init__(node_in_dim, node_h_dim, edge_in_dim, edge_h_dim, seq_in,
                                         num_layers, drop_rate)
        self.ln = nn.LayerNorm(node_h_dim[0])

    def forward(self, h_V, edge_index, h_E, seq=None, batch=None, padding=True):
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
        if padding:
            lengths = torch.bincount(batch)
            out = torch.split(out, lengths.tolist())
            out = pad_sequence(out, batch_first=True)

        return out


class ConvEncoder(nn.Module):
    """protein feature extraction.
       modified from https://github.com/lifanchen-simm/transformerCPI/blob/master/Kinase/model.py
    """
    def __init__(self, input_dim, hiden_dim, num_layers=1, kernel_size=5, dropout=0.3, max_seq_len=61):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"
        self.pos_embedding = nn.Embedding(max_seq_len, input_dim)
        self.fc = nn.Linear(input_dim, hiden_dim)
        self.ln = nn.LayerNorm(hiden_dim)
        self.register_buffer('scale', torch.sqrt(torch.FloatTensor([0.5])))
        dims_in = [hiden_dim]*num_layers
        dims_out = [hiden_dim*2]*num_layers
        model = []
        for dim_in, dim_out in zip(dims_in, dims_out):
            model.append(nn.Sequential(
                                nn.Dropout(dropout),
                                nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                                nn.GLU(dim=1)))
        self.convs = nn.ModuleList(model)


    def forward(self, x):
        pos = torch.arange(0, x.shape[1]).unsqueeze(0).to(x.device)
        x = x + self.pos_embedding(pos)
        x = self.fc(x)
        x = x.permute(0, 2, 1)
        for conv in self.convs:
            output = conv(x)
            x = (output+x)*self.scale
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        return x


class BANDecoder(BANLayer):
    def __init__(self, v_dim, q_dim, h_dim, h_out, act='ReLU', dropout=0.2, k=3, hidden_dims=(1024, 256)):
        super().__init__(v_dim, q_dim, h_dim, h_out, act, dropout, k)
        dims = [h_dim]+list(hidden_dims)
        model = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            model.append(nn.Linear(dim_in, dim_out))
            model.append(nn.ReLU(inplace=True))
            model.append(nn.Dropout(dropout))
        model.append(nn.Linear(dims[-1], 1))
        self.project = nn.Sequential(*model)

    def forward(self, v, q, softmax=False):
        x, attn = super().forward(v, q, softmax)
        x = self.project(x)
        return x.squeeze(-1), attn



class PocketDTA(ModelBase):
    dataset_cls = PocketDTADataset
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        protein_embed_dim = config['protein_embedding_dim']
        ligand_inputs_embeds_dim = config['ligand_inputs_embeds_dim']
        ligand_embed_dim = config['ligand_embedding_dim']
        protein_node_s_dim = config['protein_graph_node_s_dim']
        protein_node_v_dim = config['protein_graph_node_v_dim']
        protein_edge_s_dim = config['protein_graph_edge_s_dim']
        protein_edge_v_dim = config['protein_graph_edge_v_dim']
        ligand_max_seq_length = config['ligand_max_seq_length']

        protein_encoder_hidden_dim = config['protein_encoder_hidden_dim']
        protein_encoder_output_dim = config['protein_encoder_output_dim']

        ligand_encoder_hidden_dim = config['ligand_encoder_hidden_dim']
        ligand_encoder_output_dim = config['ligand_encoder_output_dim']
        ligand_encoder_window_size = config['ligand_encoder_window_size']
        protein_struct_encoder_dropout = config['protein_struct_encoder_dropout']
        protein_struct_encoder_num_layers = config['protein_struct_encoder_num_layers']
        protein_struct_encoder_node_h_dim = config['protein_struct_encoder_node_h_dim']
        protein_struct_encoder_edge_h_dim = config['protein_struct_encoder_edge_h_dim']
        fusion_hidden_dim = config['fusion_hidden_dim']
        fusion_heads = config['fusion_heads']
        decoder_hidden_dims = config['decoder_hidden_dims']

        self.protein_embedding_encoder = EmbeddingEncoder(protein_embed_dim, protein_encoder_hidden_dim, protein_encoder_output_dim)
        self.ligand_embedding_encoder = EmbeddingEncoder(ligand_embed_dim, ligand_encoder_hidden_dim, ligand_encoder_output_dim)

        self.ligand_struct_encoder = ConvEncoder(ligand_inputs_embeds_dim, ligand_encoder_output_dim,
                                                  kernel_size=ligand_encoder_window_size,
                                                 max_seq_len=ligand_max_seq_length)
        self.protein_struct_encoder = GVPEncoder(node_in_dim=(protein_node_s_dim, protein_node_v_dim),
                                                 node_h_dim=protein_struct_encoder_node_h_dim,
                                                 edge_in_dim=(protein_edge_s_dim, protein_edge_v_dim),
                                                 edge_h_dim=protein_struct_encoder_edge_h_dim,
                                                num_layers=protein_struct_encoder_num_layers,
                                                drop_rate=protein_struct_encoder_dropout)
        self.protein_struct_project = nn.Sequential(nn.Linear(protein_struct_encoder_node_h_dim[0], fusion_hidden_dim),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(fusion_hidden_dim, protein_encoder_output_dim))

        self.fusion_decoder = weight_norm(BANDecoder(v_dim=ligand_encoder_output_dim, q_dim=protein_encoder_output_dim,
                                                     h_dim=fusion_hidden_dim, h_out=fusion_heads, k=3,
                                                     hidden_dims=decoder_hidden_dims),
                                              name='h_mat', dim=None)


    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        parser.add_argument("--protein_encoder_hidden_dim", type=int, default=256)
        parser.add_argument("--ligand_encoder_hidden_dim", type=int, default=512)
        parser.add_argument("--protein_encoder_output_dim", type=int, default=1280)
        parser.add_argument("--ligand_encoder_output_dim", type=int, default=256)
        parser.add_argument("--ligand_encoder_window_size", type=int, default=5)
        parser.add_argument("--fusion_hidden_dim", type=int, default=128)
        parser.add_argument("--fusion_heads", type=int, default=6)

        parser.add_argument("--protein_struct_encoder_dropout", default=0.4, type=float)
        parser.add_argument("--protein_struct_encoder_num_layers", default=3, type=int)
        parser.add_argument('--protein_struct_encoder_node_h_dim', default=(128, 3), type=int, nargs='+')
        parser.add_argument('--protein_struct_encoder_edge_h_dim', default=(32, 1), type=int, nargs='+')

        parser.add_argument("--decoder_hidden_dims", type=int, nargs='+', default=[1024, 1024, 256])
        parser.add_argument("--decoder_dropout", type=float, default=0.2)
        parser.add_argument("--weight_decay", type=int, default=0.0001)
        parser.add_argument("--lr_step_size", type=int, default=5)
        parser.set_defaults(lr=0.001, max_epochs=200, accumulate_grad_batches=8, batch_size=32, lr_gamma=0.95,
                            check_val_every_n_epoch=1)

    def forward(self, ligand_embedding, ligand_inputs_embeds, protein_embedding, protein_graph):
        protein_embeds = self.protein_embedding_encoder(protein_embedding)
        ligand_embeds = self.ligand_embedding_encoder(ligand_embedding)
        ligand_hidden_embeds = self.ligand_struct_encoder(ligand_inputs_embeds)

        protein_hidden_embeds = self.protein_struct_encoder((protein_graph.node_s, protein_graph.node_v),
                                                            protein_graph.edge_index,
                                                            (protein_graph.edge_s, protein_graph.edge_v),
                                                            batch=protein_graph.batch)
        protein_hidden_embeds = self.protein_struct_project(protein_hidden_embeds)
        ligand = torch.cat([ligand_embeds.unsqueeze(1), ligand_hidden_embeds], dim=1)
        protein = torch.cat([protein_embeds.unsqueeze(1), protein_hidden_embeds], dim=1)
        predict, attn = self.fusion_decoder(ligand, protein)
        return predict, attn


    def step(self, ligand_embedding, ligand_inputs_embeds, protein_embedding, protein_graph, affinity):
        predict, attn = self.forward(ligand_embedding, ligand_inputs_embeds, protein_embedding, protein_graph)
        loss = F.mse_loss(predict, affinity)
        return {"loss": loss, "predict": predict, "attn": attn}

    def configure_optimizers(self):
        weight_p, bias_p = [], []
        for name, p in self.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer_inner = RAdam([{'params': weight_p, 'weight_decay': self.config['weight_decay']},
                                      {'params': bias_p, 'weight_decay': 0}], lr=self.config['lr'])
        optimizer = Lookahead(optimizer_inner, la_steps=5, la_alpha=0.5)
        # optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config['lr_step_size'], gamma=self.config['lr_gamma'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
                }
            }


class PocketDTA2(PocketDTA):
    dataset_cls = PocketDTADataset2
    def __init__(self, config):
        super().__init__(config)


class PocketDTA3(PocketDTA):
    def __init__(self, config):
        super().__init__(config)

    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        parser.set_defaults(lr=0.001, max_epochs=200, accumulate_grad_batches=1, batch_size=256, lr_gamma=0.95,
                            check_val_every_n_epoch=1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config['lr'])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=self.config['lr_step_size'], gamma=self.config['lr_gamma'])
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
                }
            }


if __name__ == '__main__':
    from toolbox import Evaluator
    from toolbox import Evaluator
    dataset_names = ['davis', 'kiba']
    dataset_names = ['kiba']
    for dataset_name in dataset_names:
        Evaluator(model_name="PocketDTA", dataset_name=dataset_name, use_wandb_logger=True,
                  wandb_project='DTA Benchmark', wandb_online=False,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=False, comment='exp_demo',
                  num_workers=12).run(debug=True)

        Evaluator(model_name="PocketDTA", dataset_name=dataset_name, use_wandb_logger=True,
                  wandb_project='DTA Benchmark', wandb_online=False,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=True, comment='exp_demo',
                  num_workers=12).run(debug=True)