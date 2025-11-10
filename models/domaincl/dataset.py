import os
import json
import numpy as np
from toolbox.featurizer.ligand import MorganFeaturizer, SimpleSmilesTokenFeaturizer
from toolbox import datamodule, FeaturizerBase, FeatData, GraphData
from toolbox.config import STRUCT_ROOT_DIR
from toolbox.featurizer.protein import LLMFeaturizer, SimpleProtTokenFeaturizer, LLMStructFeaturizer, PocketGVPFeaturizer

class ProteinDomainFeaturizer(FeaturizerBase):
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR, use_llm=False,
                 pretrained_model_name_or_path="Rostlab/prot_bert", feat_type='seq'):
        super().__init__()
        self.struct_root_dir = struct_root_dir
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.feat_type = feat_type
        if use_llm:
            self.seq_featurizer = LLMFeaturizer(pretrained_model_name_or_path=pretrained_model_name_or_path, feat_type=feat_type)
        else:
            self.seq_featurizer = SimpleProtTokenFeaturizer(protein_max_lengths=None)
        self.use_llm = use_llm
        self.seq_struct_map_file = os.path.join(struct_root_dir, 'seq_struct_esmfold_map.json')
        with open(self.seq_struct_map_file) as f:
            self.seq_struct_map = json.load(f)

    def _featurize(self, datapoint, **kwargs):
        cls = self.seq_featurizer.featurize([datapoint])[0]
        seq_len = len(cls.graph.inputs_embeds) if self.use_llm else len(cls.graph.input_ids)
        domain_loc = None
        info = self.seq_struct_map.get(datapoint, None)
        if info is not None:
            domain_loc = info.get('domain', [{"loc":None}])[0]['loc']
        if domain_loc is None:
            domain_loc = [1, seq_len]
        domain_mask = np.zeros(seq_len, dtype=int)
        domain_mask[domain_loc[0]-1:domain_loc[1]] = 1
        graph_domain = self.seq_featurizer([datapoint[domain_loc[0]-1:domain_loc[1]]])[0].graph

        if self.use_llm:
            graph = GraphData(domain_mask=domain_mask, **cls.graph.items())
        else:
            onehot = np.eye(self.seq_featurizer.tokenizer.get_vocab_size())
            input_embeds = onehot[cls.graph.input_ids]
            graph = GraphData(inputs_embeds=input_embeds, inputs_embeds_batch=cls.graph.input_ids_batch,
                              domain_mask=domain_mask)
            graph_domain = GraphData(inputs_embeds=onehot[graph_domain.input_ids], inputs_embeds_batch=graph_domain.input_ids_batch)
        return FeatData(graph=graph, graph_domain=graph_domain, prefix='protein')


class MorganDomainDataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = MorganFeaturizer
    protein_featurizer_cls = ProteinDomainFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.to_pandas()
        self.to_tensor()


class MorganGVPDataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = MorganFeaturizer
    protein_featurizer_cls = PocketGVPFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.to_pandas()
        self.to_tensor()


class DomainDataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = SimpleSmilesTokenFeaturizer
    protein_featurizer_cls = ProteinDomainFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.to_pandas()
        self.to_tensor()


class ProteinPocketDomainFeaturizer(ProteinDomainFeaturizer):
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR,
                 pretrained_model_name_or_path="Rostlab/prot_bert", feat_type='seq',
                 pocket_type='fpocket', pocket_tops=3):
        use_llm = True
        super().__init__(struct_root_dir, use_llm, pretrained_model_name_or_path, feat_type)
        self.struct_featurizer = LLMStructFeaturizer(pocket_type=pocket_type, pretrained_model_name_or_path=pretrained_model_name_or_path,
                                                     pocket_tops=pocket_tops)
        self.pocket_type = pocket_type
        self.pocket_tops = pocket_tops

    def _featurize(self, datapoint, **kwargs):
        cls = self.seq_featurizer.featurize([datapoint])[0]
        pocket = self.struct_featurizer.featurize([datapoint])[0]
        seq_len = len(cls.graph.inputs_embeds) if self.use_llm else len(cls.graph.input_ids)
        domain_loc = None
        info = self.seq_struct_map.get(datapoint, None)
        if info is not None:
            domain_loc = info.get('domain', [{"loc":None}])[0]['loc']
        if domain_loc is None:
            domain_loc = [1, seq_len]
        domain_mask = np.zeros(seq_len, dtype=int)
        domain_mask[domain_loc[0]-1:domain_loc[1]] = 1
        graph_domain = self.seq_featurizer([datapoint[domain_loc[0]-1:domain_loc[1]]])[0].graph

        inputs_embeds=pocket.graph.node_features
        graph = GraphData(inputs_embeds=inputs_embeds, inputs_embeds_batch=np.zeros(len(inputs_embeds), dtype=int))
        return FeatData(graph=graph, graph_domain=graph_domain, prefix='protein')


class PocketDomainDataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = MorganFeaturizer
    protein_featurizer_cls = ProteinPocketDomainFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.to_pandas()
        self.to_tensor()

if __name__=='__main__':
    import argparse
    from toolbox import DataModule
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    dataset_cls = MorganGVPDataset
    dataset_cls.add_parser_arguments(parser)
    DataModule.add_parser_arguments(parser)
    args = parser.parse_args()
    args.dataset_name='davis'
    args.edge_num = 10
    datamodule = DataModule(vars(args), dataset_cls=dataset_cls)
    datamodule.prepare_data()
    datamodule.setup()
    for batch in tqdm(datamodule.train_dataloader()):
        pass