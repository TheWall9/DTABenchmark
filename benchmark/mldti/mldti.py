from torch.nn import functional as F

from toolbox import ModelBase
from benchmark.deepdta.dataset import DeepDTADataset
from benchmark.mldti.ML_DTI import DTImodel
from benchmark.mldti.LSTM_DTI import DTImodel as LSTMDTI



class ML_DTI(ModelBase):
    dataset_cls = DeepDTADataset
    def __init__(self, config):
        super(ML_DTI, self).__init__(config)
        num_protein_tokens = config['num_protein_tokens']
        num_ligand_tokens = config['num_ligand_tokens']
        embedding_dim = config['encoder_embedding_dim']
        hidden_dim = config['encoder_hidden_dim']
        self.model = DTImodel(vocab_prot_size=num_protein_tokens,
                              vocab_drug_size=num_ligand_tokens,
                              embedding_size=embedding_dim,
                              filter_num=hidden_dim,)

    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        parser.add_argument("--encoder_embedding_dim", type=int, default=128)
        parser.add_argument("--encoder_hidden_dim", type=int, default=32)
        parser.set_defaults(lr=1e-3, max_epochs=1000)

    def forward(self, ligand_input_ids, protein_input_ids):
        predict = self.model(protein_input_ids, ligand_input_ids)
        return predict.squeeze(-1)

    def step(self, ligand_input_ids, protein_input_ids, affinity):
        predict = self.forward(ligand_input_ids, protein_input_ids)
        loss = F.mse_loss(predict, affinity)
        return {"loss": loss, "predict": predict}



class LSTM_DTI(ML_DTI):
    def __init__(self, config):
        super(LSTM_DTI, self).__init__(config)
        num_protein_tokens = config['num_protein_tokens']
        num_ligand_tokens = config['num_ligand_tokens']
        embedding_dim = config['encoder_embedding_dim']
        hidden_dim = config['encoder_hidden_dim']
        self.model = LSTMDTI(vocab_protein_size=num_protein_tokens,
                              vocab_ligand_size=num_ligand_tokens,
                              embedding_size=embedding_dim,
                              filter_num=hidden_dim,)


if __name__ == "__main__":
    from toolbox import Evaluator
    Evaluator(model_name='ML_DTI', max_epochs=2).run(debug=True)