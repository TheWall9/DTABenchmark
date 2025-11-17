import torch
from torch import nn, optim
from torch.nn import functional as F

from torch_geometric.utils import to_dense_batch
from lightning.pytorch.callbacks import EarlyStopping
from toolbox import ModelBase
from benchmark.deepdta.dataset import DeepDTADataset


class ConvEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=128, hidden_dim=32, num_layers=3, kernel_size=8):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, padding_idx=0)
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size,]*num_layers

        dims = [embedding_dim]+[hidden_dim*i for i in range(1, num_layers+1)]
        model = []
        for dim_in, dim_out, window_size in zip(dims[:-1], dims[1:], kernel_size):
            model.append(nn.Conv1d(dim_in, dim_out, window_size, stride=1, padding=0))
            model.append(nn.ReLU(inplace=True))
        self.model = nn.Sequential(*model)
        # self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, input_ids, pool=True):
        x = self.embedding(input_ids)
        x = x.permute(0,2,1)
        x = self.model(x)
        if pool:
            x, _ = x.max(dim=-1)
        # x = self.pool(x).squeeze(-1)
        return x


class CombinedDecoder(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dims=(1024, 1024, 512), dropout=0.1, norm=False, input_dim3=0):
        super().__init__()
        dims = [input_dim1+input_dim2+input_dim3]+list(hidden_dims)
        model = []
        for i, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            model.append(nn.Linear(dim_in, dim_out))
            model.append(nn.ReLU(inplace=True))
            if i!=len(hidden_dims)-1:
                if norm:
                    model.append(nn.LayerNorm(dim_out))
                model.append(nn.Dropout(dropout))
        model.append(nn.Linear(dims[-1], 1))
        self.model = nn.Sequential(*model)

    def forward(self, x1, x2, x3=None):
        if x3 is None:
            x = torch.cat([x1, x2], dim=-1)
        else:
            x = torch.cat([x1, x2, x3], dim=-1)
        output = self.model(x).squeeze(-1)
        return output


class DeepDTA(ModelBase):
    dataset_cls = DeepDTADataset
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        """CombinedCategoricalModel"""
        num_protein_tokens = config['num_protein_tokens']
        num_ligand_tokens = config['num_ligand_tokens']
        embedding_dim = config['encoder_embedding_dim']
        hidden_dim = config['encoder_hidden_dim']
        num_layers = config['encoder_num_layers']
        protein_window_size = config['protein_window_size']
        ligand_window_size = config['ligand_window_size']
        decoder_hidden_dims = config['decoder_hidden_dims']
        decoder_dropout = config['decoder_dropout']
        self.protein_encoder = ConvEncoder(num_protein_tokens, embedding_dim=embedding_dim,
                                           hidden_dim=hidden_dim, num_layers=num_layers,
                                           kernel_size=protein_window_size)
        self.ligand_encoder = ConvEncoder(num_ligand_tokens, embedding_dim=embedding_dim,
                                          hidden_dim=hidden_dim, num_layers=num_layers,
                                          kernel_size=ligand_window_size)
        self.affinity_decoder = CombinedDecoder(hidden_dim*num_layers, hidden_dim*num_layers,
                                                decoder_hidden_dims, decoder_dropout, norm=True)


    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        parser.add_argument("--protein_window_size", type=int, default=8)
        parser.add_argument("--ligand_window_size", type=int, default=8)
        parser.add_argument("--encoder_embedding_dim", type=int, default=128)
        parser.add_argument("--encoder_hidden_dim", type=int, default=32)
        parser.add_argument("--encoder_num_layers", type=int, default=3)
        parser.add_argument("--decoder_hidden_dims", type=int, nargs='+', default=[1024, 1024, 512])
        parser.add_argument("--decoder_dropout", type=float, default=0.1)
        parser.add_argument("--early_stop_patient", type=int, default=15)

    @classmethod
    def generate_optuna_params(cls, trial):
        protein_window_size = trial.suggest_categorical("protein_window_size", [4, 8, 12])
        ligand_window_size = trial.suggest_categorical("ligand_window_size", [4, 6, 8])
        return {"protein_window_size": protein_window_size,
                "ligand_window_size": ligand_window_size}

    def forward(self, ligand_input_ids, protein_input_ids):
        protein = self.protein_encoder(protein_input_ids)
        ligand = self.ligand_encoder(ligand_input_ids)
        predict = self.affinity_decoder(protein, ligand)
        return predict

    def step(self, ligand_input_ids, protein_input_ids, affinity):
        predict = self.forward(ligand_input_ids, protein_input_ids)
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
    dataset_names = ['davis', 'davis_domain', 'davis_domain2', 'kiba', 'kiba_domain2']
    dataset_names = ['davis_domain2', 'kiba_domain2']
    for dataset_name in dataset_names:
        Evaluator(model_name='DeepDTA', deterministic=True, dataset_name=dataset_name, max_epochs=200, lr=1e-3,
                  use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=False, early_stop_patient=100, batch_size=256, comment='exp',
                  num_workers=12).run()

        Evaluator(model_name='DeepDTA', deterministic=True, dataset_name=dataset_name, max_epochs=200, lr=1e-3,
                  use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=False,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=True, early_stop_patient=100, batch_size=256, comment='exp',
                  num_workers=12).run()
