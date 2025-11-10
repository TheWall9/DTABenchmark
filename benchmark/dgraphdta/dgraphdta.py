import torch
from torch import nn, optim
import torch.nn.functional as F

import torch_geometric.nn as gnn
from toolbox.model_helper import ModelBase
from benchmark.dgraphdta.dataset import DGraphDTADataset
from benchmark.deepdta.deepdta import CombinedDecoder


class GCNConvEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=128, num_layers=3, dropout=0.2):
        super().__init__()
        dims = [input_dim]+[input_dim*pow(2, i) for i in range(num_layers)]
        model = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            model.append((gnn.Sequential("x, edge_index",[
                (gnn.GCNConv(dim_in, dim_out), "x, edge_index -> x"),
                (nn.ReLU(inplace=True), 'x -> x')
            ]), 'x, edge_index -> x'))

        self.model = gnn.Sequential("x, edge_index", model)

        self.project = nn.Sequential(nn.Linear(dims[-1], hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=dropout),
                                     nn.Linear(hidden_dim, output_dim),
                                     nn.Dropout(p=dropout),)

    def forward(self, x, edge_index, batch_idx):
        x = self.model(x, edge_index)
        x = gnn.global_mean_pool(x, batch_idx)
        x= self.project(x)
        return x


class DGraphDTA(ModelBase):
    dataset_cls = DGraphDTADataset
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        protein_feat_dim = config['protein_graph_node_features_dim']
        ligand_feat_dim = config['ligand_graph_node_features_dim']
        encoder_dropout = config['encoder_dropout']
        encoder_output_dim = config['encoder_output_dim']

        hidden_dim = config['encoder_hidden_dim']
        num_layers = config['encoder_num_layers']
        decoder_hidden_dims = config['decoder_hidden_dims']
        decoder_dropout = config['decoder_dropout']
        self.protein_encoder = GCNConvEncoder(protein_feat_dim, hidden_dim=hidden_dim, output_dim=encoder_output_dim,
                                             num_layers=num_layers, dropout=encoder_dropout)

        self.ligand_encoder = GCNConvEncoder(ligand_feat_dim, hidden_dim=hidden_dim, output_dim=encoder_output_dim,
                                             num_layers=num_layers, dropout=encoder_dropout)
        self.affinity_decoder = CombinedDecoder(encoder_output_dim, encoder_output_dim,
                                                decoder_hidden_dims, decoder_dropout)


    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        parser.add_argument('--encoder_hidden_dim', type=int, default=1024)
        parser.add_argument('--encoder_output_dim', type=int, default=128)
        parser.add_argument('--encoder_dropout', type=float, default=0.2)
        parser.add_argument('--encoder_num_layers', type=int, default=3)
        parser.add_argument("--decoder_hidden_dims", type=int, nargs='+', default=[1024, 512])
        parser.add_argument("--decoder_dropout", type=float, default=0.2)
        parser.set_defaults(lr=0.001, max_epochs=2000)

    def forward(self, ligand_graph, protein_graph):
        protein = self.protein_encoder(protein_graph.node_features, protein_graph.edge_index, protein_graph.batch)
        ligand = self.ligand_encoder(ligand_graph.node_features, ligand_graph.edge_index, ligand_graph.batch)
        affinity = self.affinity_decoder(protein, ligand)
        return affinity

    def step(self, ligand_graph, protein_graph, affinity):
        predict = self.forward(ligand_graph, protein_graph)
        loss = F.mse_loss(predict, affinity)
        return {"loss": loss, "predict": predict}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return optimizer



if __name__ == '__main__':
    from toolbox import Evaluator
    Evaluator(model_name="DGraphDTA", max_epochs=2).run(debug=True)
