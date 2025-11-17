from torch import nn
from torch.nn import functional as F
from toolbox import datamodule, ModelBase
from toolbox.featurizer.protein import SimpleProtTokenFeaturizer
from benchmark.deepdta.deepdta import DeepDTA


class DeepDTANoProteins(DeepDTA):
    def __init__(self, config):
        super().__init__(config)
        hidden_dim = config['encoder_hidden_dim']
        num_layers = config['encoder_num_layers']
        self.protein_encoder = nn.Embedding(config['num_proteins'], hidden_dim*num_layers)

    def step(self, ligand_input_ids, PID, affinity):
        predict = self.forward(ligand_input_ids, PID)
        loss = F.mse_loss(predict, affinity)
        return {"loss": loss, "predict": predict}


from toolbox.featurizer.ligand import MorganFeaturizer
class MorganDataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = MorganFeaturizer
    protein_featurizer_cls = SimpleProtTokenFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.to_pandas()
        self.to_tensor()


class MorganDTANoProtein(DeepDTA):
    dataset_cls = MorganDataset
    def __init__(self, config):
        config['num_ligand_tokens'] = 1
        super().__init__(config)
        self.ligand_encoder = nn.Sequential(nn.Linear(config['ligand_embedding_dim'], config['encoder_hidden_dim']*3),
                                            # nn.ReLU(inplace=True),
                                            # nn.Linear(config['encoder_embedding_dim'], config['encoder_hidden_dim']),
                                            # nn.ReLU(inplace=True),
                                            # nn.Linear(config['encoder_hidden_dim'], config['encoder_hidden_dim']*2),
                                            nn.ReLU(inplace=True),
                                            # nn.Linear(config['encoder_embedding_dim'], config['encoder_hidden_dim']*3),
                                            )

    def step(self, ligand_embedding, protein_input_ids, affinity):
        predict = self.forward(ligand_embedding, protein_input_ids)
        loss = F.mse_loss(predict, affinity)
        return {"loss": loss, "predict": predict}


class MorganDTA(DeepDTA):
    dataset_cls = MorganDataset
    def __init__(self, config):
        config['num_ligand_tokens'] = 1
        super().__init__(config)
        self.ligand_encoder = nn.Sequential(nn.Linear(config['ligand_embedding_dim'], config['encoder_embedding_dim'*3]),
                                            # nn.ReLU(inplace=True),
                                            # nn.Linear(config['encoder_embedding_dim'], config['encoder_hidden_dim']),
                                            # nn.ReLU(inplace=True),
                                            # nn.Linear(config['encoder_hidden_dim'], config['encoder_hidden_dim']*2),
                                            nn.ReLU(inplace=True),
                                            # nn.Linear(config['encoder_hidden_dim']*2, config['encoder_hidden_dim']*3),
                                            )
    def step(self, ligand_embedding, protein_input_ids, affinity):
        predict = self.forward(ligand_embedding, protein_input_ids)
        loss = F.mse_loss(predict, affinity)
        return {"loss": loss, "predict": predict}


if __name__ == '__main__':
    from toolbox import Evaluator
    Evaluator(model_name='MorganDTA', deterministic=True, dataset_name='davis_domain_ncbi', max_epochs=200, lr=1e-3,
              use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=True,
              monitor_metric = 'val/CI', monitor_mode='max',
              merge_train_val=False, early_stop_patient=100, batch_size=256, comment='exp',
              num_workers=12).run()
    Evaluator(model_name='MorganDTA', deterministic=True, dataset_name='davis_domain_ncbi', max_epochs=200, lr=1e-3,
              use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=True,
              monitor_metric = 'val/CI', monitor_mode='max',
              merge_train_val=True, early_stop_patient=100, batch_size=256, comment='exp',
              num_workers=12).run()
    exit()


    dataset_names = ['davis', 'kiba']
    for dataset_name in dataset_names:
        Evaluator(model_name='DeepDTANoProteins', deterministic=True, dataset_name=dataset_name, max_epochs=200, lr=1e-3,
                  use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=True,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=False, early_stop_patient=100, batch_size=256, comment='exp',
                  num_workers=12).run()

        Evaluator(model_name='DeepDTANoProteins', deterministic=True, dataset_name=dataset_name, max_epochs=200, lr=1e-3,
                  use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=True,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=True, early_stop_patient=100, batch_size=256, comment='exp',
                  num_workers=12).run()

    dataset_names = ['davis', 'kiba']
    for dataset_name in dataset_names:
        Evaluator(model_name='MorganDTANoProtein', deterministic=True, dataset_name=dataset_name, max_epochs=200, lr=1e-3,
                  use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=True,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=False, early_stop_patient=100, batch_size=256, comment='exp',
                  num_workers=12).run()

        Evaluator(model_name='MorganDTANoProtein', deterministic=True, dataset_name=dataset_name, max_epochs=200, lr=1e-3,
                  use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=True,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=True, early_stop_patient=100, batch_size=256, comment='exp',
                  num_workers=12).run()

    dataset_names =  ['davis', 'davis_domain', 'davis_domain2', 'kiba', 'kiba_domain2']
    for dataset_name in dataset_names:
        Evaluator(model_name='MorganDTA', deterministic=True, dataset_name=dataset_name, max_epochs=200, lr=1e-3,
                  use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=True,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=False, early_stop_patient=100, batch_size=256, comment='exp',
                  num_workers=12).run()
        Evaluator(model_name='MorganDTA', deterministic=True, dataset_name=dataset_name, max_epochs=200, lr=1e-3,
                  use_wandb_logger=True, wandb_project='DTA Benchmark', wandb_online=True,
                  monitor_metric = 'val/CI', monitor_mode='max',
                  merge_train_val=True, early_stop_patient=100, batch_size=256, comment='exp',
                  num_workers=12).run()


