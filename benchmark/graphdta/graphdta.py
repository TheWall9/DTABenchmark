import os
import abc

import torch
from torch import nn, optim
import torch.nn.functional as F

import torch_geometric.nn as gnn
from torch_geometric.loader import DataLoader
from toolbox.model_helper import ModelBase
from benchmark.graphdta.dataset import GraphDTADataset
from benchmark.deepdta.deepdta import CombinedDecoder
from toolbox import datamodule
from toolbox.featurizer.ligand import MolFeaturizerBase, SimpleSmilesTokenFeaturizer
from toolbox.featurizer.protein import SimpleProtTokenFeaturizer


class ConvEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=128, hidden_dim=128, output_dim=128, seq_length=1000, kernel_size=8, use_norm_conv=False):
        super().__init__()
        self.use_norm_conv = use_norm_conv
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0)
        if use_norm_conv:
            self.conv = nn.Conv1d(embedding_dim, hidden_dim, kernel_size)
            self.project = nn.Linear((seq_length-kernel_size+1)*hidden_dim, output_dim)
        else:
            self.conv = nn.Conv1d(seq_length, hidden_dim, kernel_size)
            self.project = nn.Linear(hidden_dim * (embedding_dim-kernel_size+1), output_dim)


    def forward(self, x):
        x = self.embedding(x)
        if self.use_norm_conv:
            x = self.conv(x.transpose(0, 2, 1))
        else:
            x = self.conv(x) # ?
        x = torch.flatten(x, start_dim=1)
        x = self.project(x)
        return x


class GINConvEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128, num_layers=5, dropout=0.2):
        super().__init__()
        dims = [input_dim]+[hidden_dim]*(num_layers-1)
        model = []
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            model.append((gnn.Sequential("x, edge_index",[
                (gnn.GINConv(nn.Sequential(nn.Linear(dim_in, dim_out),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(dim_out, dim_out))), "x, edge_index -> x"),
                (nn.Sequential(nn.ReLU(inplace=True),
                               nn.BatchNorm1d(hidden_dim)), 'x -> x')
            ]), 'x, edge_index -> x'))

        self.model = gnn.Sequential("x, edge_index", model)
        self.project = nn.Sequential(nn.Linear(dims[-1], output_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(p=dropout))

    def forward(self, x, edge_index, batch_idx):
        x = self.model(x, edge_index)
        x = gnn.global_add_pool(x, batch_idx)
        x= self.project(x)
        return x


class GraphDTA(ModelBase):
    dataset_cls = GraphDTADataset
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        num_protein_tokens = config['num_protein_tokens']
        protein_max_lengths = config['protein_max_lengths']
        ligand_feat_dim = config['ligand_graph_node_features_dim']

        protein_window_size = config['protein_window_size']
        encoder_dropout = config['encoder_dropout']
        encoder_output_dim = config['encoder_output_dim']
        encoder_use_norm_conv = config['encoder_use_norm_conv']
        embedding_dim = config['encoder_embedding_dim']
        hidden_dim = config['encoder_hidden_dim']
        num_layers = config['encoder_num_layers']
        decoder_hidden_dims = config['decoder_hidden_dims']
        decoder_dropout = config['decoder_dropout']
        self.protein_encoder = ConvEncoder(num_protein_tokens, embedding_dim=embedding_dim,
                                           hidden_dim=hidden_dim, output_dim=encoder_output_dim,
                                           kernel_size=protein_window_size, seq_length=protein_max_lengths,
                                           use_norm_conv=encoder_use_norm_conv)

        self.ligand_encoder = GINConvEncoder(ligand_feat_dim, hidden_dim=hidden_dim, output_dim=encoder_output_dim,
                                             num_layers=num_layers, dropout=encoder_dropout)
        self.affinity_decoder = CombinedDecoder(encoder_output_dim, encoder_output_dim,
                                                decoder_hidden_dims, decoder_dropout)


    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        parser.add_argument('--protein_window_size', type=int, default=8)
        parser.add_argument('--encoder_embedding_dim', type=int, default=128)
        parser.add_argument('--encoder_hidden_dim', type=int, default=32)
        parser.add_argument('--encoder_output_dim', type=int, default=128)
        parser.add_argument('--encoder_use_norm_conv', action='store_true')
        parser.add_argument('--encoder_dropout', type=float, default=0.2)
        parser.add_argument('--encoder_num_layers', type=int, default=5)
        parser.add_argument("--decoder_hidden_dims", type=int, nargs='+', default=[256, 1024, 256])
        parser.add_argument("--decoder_dropout", type=float, default=0.2)
        parser.set_defaults(lr=0.0005, max_epochs=1000)

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



if __name__ == '__main__':
    from toolbox import Evaluator
    dataset_names = ['davis', 'davis_domain', 'davis_domain2', 'kiba', 'kiba_domain2']
    for dataset_name in dataset_names:
        Evaluator(model_name="GraphDTA", dataset_name=dataset_name, use_wandb_logger=True,
                  wandb_project='DTA Benchmark', wandb_online=False,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=False, comment='exp_demo',
                  num_workers=12).run(debug=True)

        Evaluator(model_name="GraphDTA", dataset_name=dataset_name, use_wandb_logger=True,
                  wandb_project='DTA Benchmark', wandb_online=False,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=True, comment='exp_demo',
                  num_workers=12).run(debug=True)
