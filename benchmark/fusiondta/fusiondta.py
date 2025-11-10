import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
from torch_geometric import nn as gnn
from torch_geometric.utils import softmax as gnn_softmax, to_dense_batch, unbatch
from torch_scatter import scatter_add

from toolbox import ModelBase
from benchmark.fusiondta.dataset import FusionDTADataset
from benchmark.deepdta.deepdta import CombinedDecoder

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


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_heads, n_layers=1, bidirectional=False, embedding_dim=None, num_embeddings=None, dropout=0.1):
        super(LSTMEncoder, self).__init__()
        # self.sentence_input_fc = nn.Linear(embedding_dim, hidden_dim)
        if num_embeddings is not None:
            self.embedding = nn.Sequential(nn.Embedding(num_embeddings, embedding_dim, padding_idx=0),
                                           nn.Linear(embedding_dim, hidden_dim))
        else:
            self.embedding = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout, num_layers=n_layers)
        self.attn = LinkAttention(2*hidden_dim if bidirectional else hidden_dim, n_heads)
        # self.output_project = nn.Linear(2*hidden_dim if bidirectional else hidden_dim, hidden_dim)


    def pack_sequence(self, inputs, batch):
        seq_len = torch.bincount(batch)
        seq_lens_sorted, sort_idx = seq_len.sort(descending=True)

        batch_map = torch.cat([(batch == b).nonzero(as_tuple=True)[0] for b in sort_idx])
        inputs_sorted = inputs[batch_map]
        max_len = seq_lens_sorted.max()
        batch_sizes = torch.tensor([(seq_lens_sorted > i).sum().item() for i in range(max_len)])
        packed = PackedSequence(data=inputs_sorted, batch_sizes=batch_sizes,
                                # sorted_indices=sort_idx, unsorted_indices=torch.argsort(sort_idx)
                                )
        return packed, batch_map

    def unpack_sequence(self, packed, sorted_idx):
        output_flat = packed.data.new_empty(packed.data.size())
        output_flat[sorted_idx] = packed.data
        return output_flat

    def forward(self, inputs, inputs_batch):
        """inputs: (BxL)xD
           inputs_batch: (BxL)"""
        embedding = self.embedding(inputs)
        packed_embedding, sorted_idx = self.pack_sequence(embedding, inputs_batch)
        packed_output, (h, c) = self.lstm(packed_embedding)
        output = self.unpack_sequence(packed_output, sorted_idx)
        value, attn = self.attn(output, inputs_batch)
        return value, output


class LSTMPaddedEncoder(LSTMEncoder):
    def __init__(self, *args, **kwargs):
        super(LSTMPaddedEncoder, self).__init__(*args, **kwargs)

    def forward(self, inputs, inputs_batch):
        embedding = self.embedding(inputs)
        padded_embedding, mask = to_dense_batch(embedding, inputs_batch)
        packed_output, (h, c) = self.lstm(padded_embedding)
        output = packed_output[mask]
        value, attn = self.attn(output, inputs_batch)
        return value, output



class FusionDTA(ModelBase):
    dataset_cls = FusionDTADataset
    def __init__(self, config):
        super(FusionDTA, self).__init__(config)
        embedding_dim = config['encoder_embedding_dim']
        hidden_dim = config['encoder_hidden_dim']
        num_layers = config['encoder_num_layers']
        dropout = config['dropout']
        decoder_hidden_dims = config['decoder_hidden_dims']

        n_heads = config['n_heads']
        bidirectional = config['bidirectional']
        self.ligand_encoder = LSTMEncoder(input_dim=None, num_embeddings=config['num_ligand_tokens'],
                                          embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                          bidirectional=bidirectional, n_heads=n_heads,
                                          n_layers=num_layers, dropout=dropout)
        self.protein_encoder = LSTMEncoder(input_dim=config['protein_graph_inputs_embeds_dim'],
                                          embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                          bidirectional=bidirectional, n_heads=n_heads,
                                          n_layers=num_layers, dropout=dropout)
        encoder_output_dim = hidden_dim*2 if bidirectional else hidden_dim
        self.attn = LinkAttention(encoder_output_dim, n_heads)
        self.decoder = CombinedDecoder(encoder_output_dim, encoder_output_dim, hidden_dims=decoder_hidden_dims, dropout=dropout, input_dim3=encoder_output_dim)
        self.register_buffer('mean', torch.tensor(config['affinity_mean']))
        self.register_buffer('std', torch.tensor(config['affinity_std']))
        total_params = sum(p.numel() for p in self.parameters())
        print(f"总参数个数：{total_params}")

    def forward(self, ligand_graph, protein_graph):
        ligand, ligand_hiddens = self.ligand_encoder(ligand_graph.input_ids, ligand_graph.input_ids_batch)
        protein, protein_hiddens = self.protein_encoder(protein_graph.inputs_embeds, protein_graph.inputs_embeds_batch)
        fusion = torch.cat([ligand_hiddens, protein_hiddens], dim=0)
        fusion_batch = torch.cat([ligand_graph.input_ids_batch, protein_graph.inputs_embeds_batch], dim=0)
        fusion_embedding, score = self.attn(fusion, fusion_batch)
        predict = self.decoder(protein, ligand, fusion_embedding)
        return predict

    def step(self, ligand_graph, protein_graph, affinity):
        predict = self.forward(ligand_graph, protein_graph)
        # loss = F.mse_loss(predict, affinity)
        # return {"loss": loss, "predict": predict}
        loss = F.mse_loss(predict, (affinity-self.mean)/self.std)
        return {"loss": loss, "predict": predict*self.std + self.mean}

    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        parser.add_argument('--encoder_embedding_dim', default=256, type=int)
        parser.add_argument('--encoder_hidden_dim', default=128, type=int)
        parser.add_argument('--encoder_num_layers', default=2, type=int)
        parser.add_argument('--bidirectional', action='store_true', default=False)
        parser.add_argument('--dropout', default=0.3, type=float)
        parser.add_argument('--decoder_hidden_dims', default=(2048, 512), nargs='+', type=int)
        parser.add_argument('--n_heads', default=8, type=int)
        parser.set_defaults(bidirectional=True, max_epochs=1000)


class FusionDTAPad(FusionDTA):
    def __init__(self, config):
        super(FusionDTAPad, self).__init__(config)
        embedding_dim = config['encoder_embedding_dim']
        hidden_dim = config['encoder_hidden_dim']
        num_layers = config['encoder_num_layers']
        dropout = config['dropout']

        n_heads = config['n_heads']
        bidirectional = config['bidirectional']
        self.ligand_encoder = LSTMPaddedEncoder(input_dim=None, num_embeddings=config['num_ligand_tokens'],
                                          embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                          bidirectional=bidirectional, n_heads=n_heads,
                                          n_layers=num_layers, dropout=dropout)
        self.protein_encoder = LSTMPaddedEncoder(input_dim=config['protein_graph_inputs_embeds_dim'],
                                           embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                           bidirectional=bidirectional, n_heads=n_heads,
                                           n_layers=num_layers, dropout=dropout)
        # self.register_buffer('mean', torch.tensor(5.4515))
        # self.register_buffer('std', torch.tensor(0.8947))


    def step(self, ligand_graph, protein_graph, affinity):
        predict = self.forward(ligand_graph, protein_graph)
        loss = F.mse_loss(predict, (affinity-self.mean)/self.std)
        return {"loss": loss, "predict": predict*self.std + self.mean}

if __name__=='__main__':
    from toolbox import Evaluator
    # Evaluator(model_name='FusionDTA', max_epochs=1000, dataset_name='davis',).run(debug=True)  # 0.8203641176223755 0.24 100epoch
    # Evaluator(model_name='FusionDTA', max_epochs=200, batch_size=128, merge_train_val=False, dataset_name='davis',).run(debug=True)  #
    # Evaluator(model_name='FusionDTAPad', max_epochs=150, batch_size=128, merge_train_val=True, dataset_name='davis',).run(debug=True)  #
    # Evaluator(model_name='FusionDTAPad', max_epochs=150, batch_size=128, merge_train_val=False, dataset_name='davis',).run(debug=True)  #
    # Evaluator(model_name='FusionDTAPad', max_epochs=150, batch_size=128, merge_train_val=True, dataset_name='davis',).run(debug=True)  #
    Evaluator(model_name='FusionDTA', max_epochs=150, batch_size=8, merge_train_val=True, dataset_name='davis',).run(debug=True)  #

