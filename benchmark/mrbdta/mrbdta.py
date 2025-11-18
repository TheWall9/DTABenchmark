from torch.nn import functional as F

from benchmark.deepdta import DeepDTADataset
from toolbox import ModelBase
from benchmark.mrbdta.model import Transformer






class MRBDTA(ModelBase):
    dataset_cls = DeepDTADataset
    def __init__(self, config):
        super().__init__(config)
        num_protein_tokens = config['num_protein_tokens']
        num_ligand_tokens = config['num_ligand_tokens']

        embedding_dim = config['encoder_embedding_dim']
        n_heads = config['encoder_n_heads']
        num_layers = config['encoder_num_layers']

        self.model = Transformer(drug_vocab_size=num_ligand_tokens,
                                 target_vocab_size=num_protein_tokens,
                                 d_model=embedding_dim, n_heads=n_heads,
                                 d_k=embedding_dim//n_heads, d_v=embedding_dim//n_heads,
                                 d_ff=embedding_dim*4, n_layers=num_layers)

    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        parser.add_argument("--encoder_embedding_dim", type=int, default=128)
        parser.add_argument("--encoder_num_layers", type=int, default=1)
        parser.add_argument("--encoder_n_heads", type=int, default=8)

    def forward(self, ligand_input_ids, protein_input_ids):
        ans = self.model(ligand_input_ids, protein_input_ids)
        return ans[0]

    def step(self, ligand_input_ids, protein_input_ids, affinity):
        predict = self.forward(ligand_input_ids, protein_input_ids)
        loss = F.mse_loss(predict, affinity)
        return {"loss": loss, "predict": predict}


if __name__=="__main__":
    from toolbox import Evaluator
    Evaluator(model_name='MRBDTA', max_epochs=2).run(debug=True)