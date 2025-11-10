import torch
from torch.nn import functional as F
from toolbox import datamodule
from toolbox.featurizer.ligand import SimpleSmilesTokenFeaturizer
from toolbox.featurizer.protein import LLMFeaturizer, LLMStructFeaturizer
from benchmark.fusiondta.fusiondta import FusionDTA

class FusionDTADataset2(datamodule.DTADatasetBase):
    ligand_featurizer_cls = SimpleSmilesTokenFeaturizer
    protein_featurizer_cls = LLMStructFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.to_pandas()
        self.to_tensor()

    @classmethod
    def add_parser_arguments(cls, parser):
        super().add_parser_arguments(parser)
        parser.set_defaults(pretrained_model_name_or_path='facebook/esm2_t6_8M_UR50D',
                            smiles_max_lengths=None, feat_type='full', pocket_type='fpocket')

class FusionDTA2(FusionDTA):
    dataset_cls = FusionDTADataset2
    def __init__(self, config):
        config['protein_graph_inputs_embeds_dim'] = config['protein_graph_node_features_dim']
        super().__init__(config)

    def forward(self, ligand_graph, protein_graph):
        ligand, ligand_hiddens = self.ligand_encoder(ligand_graph.input_ids, ligand_graph.input_ids_batch)
        protein, protein_hiddens = self.protein_encoder(protein_graph.node_features, protein_graph.batch)
        fusion = torch.cat([ligand_hiddens, protein_hiddens], dim=0)
        fusion_batch = torch.cat([ligand_graph.input_ids_batch, protein_graph.batch], dim=0)
        fusion_embedding, score = self.attn(fusion, fusion_batch)
        predict = self.decoder(protein, ligand, fusion_embedding)
        return predict

    def step(self, ligand_graph, protein_graph, affinity):
        predict = self.forward(ligand_graph, protein_graph)
        loss = F.mse_loss(predict, affinity)
        return {"loss": loss, "predict": predict}

if __name__ == '__main__':
    from toolbox import Evaluator
    Evaluator(model_name='FusionDTA2', max_epochs=100).run(debug=True)  # 0.8014177083969116 0.30 100epoch