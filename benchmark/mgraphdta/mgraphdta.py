import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn
from lightning.pytorch.callbacks import EarlyStopping

from toolbox import ModelBase
from benchmark.mgraphdta.dataset import MGraphDTADataset
from benchmark.deepdta.deepdta import CombinedDecoder

class StackCNN(nn.Module):
    def __init__(self, input_dim, output_dim=96, kernel_size=3, num_layers=3):
        super().__init__()
        dims = [input_dim]+[output_dim]*num_layers
        model = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            model.append(nn.Conv1d(dim_in, dim_out, kernel_size=kernel_size))
            model.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        """BxCxT"""
        x = self.model(x)
        x, _ = x.max(dim=-1)
        return x


class MConvEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=128, hidden_dim=96, num_layers=3, kernel_size=3):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0)
        self.blocks = nn.ModuleList([StackCNN(embedding_dim, hidden_dim, kernel_size, i+1) for i in range(num_layers)])
        self.project = nn.Linear(num_layers*hidden_dim, hidden_dim)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = x.permute(0,2,1)
        feats = [block(x) for block in self.blocks]
        x = torch.cat(feats, -1)
        x = self.project(x)
        return x


class GNNDenseLayer(nn.Module):
    def __init__(self, input_dim, output_dim=32, growth_rate=4):
        super().__init__()
        self.model = gnn.Sequential("x, edge_index", [
                    (gnn.GraphConv(input_dim, output_dim*growth_rate), "x, edge_index -> x"),
                    (gnn.BatchNorm(output_dim*growth_rate), "x -> x"),
                    (nn.ReLU(inplace=True), "x -> x"),
                    (gnn.GraphConv(output_dim*growth_rate, output_dim), "x, edge_index -> x"),
                    (gnn.BatchNorm(output_dim), "x -> x"),
                    (nn.ReLU(inplace=True), "x -> x"),
                ])

    def forward(self, xs, edge_index):
        if isinstance(xs, torch.Tensor):
            xs = [xs]
        x = torch.cat(xs, dim=-1)
        x = self.model(x, edge_index)
        return x

class GNNDenseBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, growth_rate=4, num_layers=3):
        super().__init__()
        blocks = []
        dims_in = [input_dim+hidden_dim*i for i in range(num_layers)]
        dims_out = [hidden_dim]*num_layers
        for dim_in, dim_out in zip(dims_in, dims_out):
            blocks.append(GNNDenseLayer(dim_in, dim_out, growth_rate=growth_rate))

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, edge_index):
        xs = [x]
        for block in self.blocks:
            x = block(xs, edge_index)
            xs.append(x)
        x = torch.cat(xs, dim=-1)
        return x

class DenseGNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, growth_rate=(2, 3, 4, 4), num_block_layers=(3,3,3,3)):
        super().__init__()
        model = [(gnn.Sequential("x, edge_index", [
            (gnn.GraphConv(input_dim, hidden_dim), "x, edge_index -> x"),
            (gnn.BatchNorm(hidden_dim), "x -> x")]), "x, edge_index -> x",)
                 ]
        dim_in = hidden_dim
        for i, num_layers in enumerate(num_block_layers):
            model.append((GNNDenseBlock(dim_in, hidden_dim, growth_rate[i], num_layers), 'x, edge_index -> x'))
            dim_in += hidden_dim*num_layers
            model.append((gnn.Sequential("x, edge_index", [
                (gnn.GraphConv(dim_in, dim_in//2), "x, edge_index -> x"),
                (gnn.BatchNorm(dim_in//2), "x -> x")]), "x, edge_index -> x")
                         )
            dim_in = dim_in//2
        self.model = gnn.Sequential("x, edge_index", model)
        self.project = nn.Linear(dim_in, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.model(x, edge_index)
        x = gnn.global_mean_pool(x, batch)
        x = self.project(x)
        return x


class MGraphDTA(ModelBase):
    dataset_cls = MGraphDTADataset
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        num_protein_tokens = config['num_protein_tokens']
        protein_encoder_num_layers = config['protein_encoder_num_layers']
        ligand_feat_dim = config['ligand_graph_node_features_dim']
        protein_window_size = config['protein_window_size']
        encoder_output_dim = config['encoder_output_dim']
        embedding_dim = config['encoder_embedding_dim']
        hidden_dim = config['encoder_hidden_dim']
        decoder_hidden_dims = config['decoder_hidden_dims']
        decoder_dropout = config['decoder_dropout']
        ligand_encoder_block_layers = config['ligand_encoder_block_layers']
        ligand_encoder_growth_rate = config['ligand_encoder_growth_rate']
        self.protein_encoder = MConvEncoder(num_protein_tokens, embedding_dim=embedding_dim, hidden_dim=encoder_output_dim,
                                            num_layers=protein_encoder_num_layers, kernel_size=protein_window_size)
        self.ligand_encoder = DenseGNNEncoder(ligand_feat_dim, output_dim=encoder_output_dim, hidden_dim=hidden_dim,
                                              num_block_layers=ligand_encoder_block_layers, growth_rate=ligand_encoder_growth_rate)

        self.affinity_decoder = CombinedDecoder(encoder_output_dim, encoder_output_dim,
                                                decoder_hidden_dims, decoder_dropout)


    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        parser.add_argument('--protein_window_size', type=int, default=3)
        parser.add_argument('--protein_encoder_num_layers', type=int, default=3)
        parser.add_argument('--encoder_embedding_dim', type=int, default=128)
        parser.add_argument('--encoder_hidden_dim', type=int, default=32)
        parser.add_argument('--encoder_output_dim', type=int, default=96)
        parser.add_argument('--ligand_encoder_block_layers', type=int, nargs='+', default=(8, 8, 8))
        parser.add_argument('--ligand_encoder_growth_rate', type=int, nargs='+', default=(2, 2, 2))
        parser.add_argument("--decoder_hidden_dims", type=int, nargs='+', default=[1024, 1024, 256])
        parser.add_argument("--decoder_dropout", type=float, default=0.1)
        parser.add_argument("--early_stop_patient", type=int, default=8)
        parser.set_defaults(lr=0.0005, max_epochs=600)

    def forward(self, ligand_graph, protein_input_ids):
        protein = self.protein_encoder(protein_input_ids)
        ligand = self.ligand_encoder(ligand_graph.node_features, ligand_graph.edge_index, ligand_graph.batch)
        affinity = self.affinity_decoder(protein, ligand)
        return affinity

    def step(self, ligand_graph, protein_input_ids, affinity):
        predict = self.forward(ligand_graph, protein_input_ids)
        loss = F.mse_loss(predict, affinity)
        return {"loss": loss, "predict": predict}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer

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
    Evaluator(model_name="MGraphDTA", max_epochs=600, dataset_name='davis', early_stop_patient=200).run(debug=True)