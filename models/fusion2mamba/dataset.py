import os
from typing import Any

import numpy as np

import torch
from rdkit import Chem
from rdkit.Chem import ValenceType, AllChem
from deepchem.feat.graph_features import one_of_k_encoding_unk, one_of_k_encoding

from toolbox import datamodule
from toolbox.featurizer.ligand import SimpleSmilesTokenFeaturizer, Mol3dGraphFeaturizer, MorganFeaturizer, \
    GraphMVPFeaturizer, GraphMVP3dPointFeaturizer, MolFeaturizerBase, UnimolFeaturizer
from toolbox.featurizer.protein import SimpleProtTokenFeaturizer, LLMStructFeaturizer, STRUCT_ROOT_DIR, LLMFeaturizer, PocketSurfaceNormalFeaturizer
from toolbox.featurizer.tools import FeaturizerBase, GraphData, FeatData
from toolbox.utils import disk_cache, serialize3d


def reorder_graph(graph, node_order, reverse_order, attr_keys=None):
    ans = {}
    attr_keys = graph.attr_keys if attr_keys is None else attr_keys
    for key in attr_keys:
        data = getattr(graph, key, None)
        if data is None:
            continue
        if 'edge_index' in key:
            index_map = {o:i for i, o in enumerate(node_order)}
            new_edge_index = np.array([index_map[item] for item in data.flatten()]).reshape(data.shape)
            ans[key] = new_edge_index
        elif 'edge' in key:
            ans[key] = data
        elif len(node_order)==len(data):
            if isinstance(data, np.ndarray):
                data = data[node_order]
            else:
                data = [data[i] for i in node_order]
            ans[key] = data
        else:
            ans[key] = data
    ans['reverse_order_batch'] = reverse_order
    return GraphData(**ans)

def serailize_graph(graph, batch_key, keys):
    pos = getattr(graph, 'pos')
    batch_idx = getattr(graph, batch_key, None) if batch_key is not None else None
    forward_order, forward_inverse_order = serialize3d(pos, batch_idx, order='hilbert')
    backward_order, backward_inverse_order = serialize3d(pos, batch_idx, order='hilbert-trans')
    forward_graph = reorder_graph(graph, forward_order, forward_inverse_order, attr_keys=keys)
    backward_graph = reorder_graph(graph, backward_order, backward_inverse_order, attr_keys=keys)
    return forward_graph, backward_graph


class FusionLigandFeaturizer(FeaturizerBase):
    def __init__(self, smiles_max_lengths=100, unimol_model_name='unimolv1', unimol_model_size='84m'):
        super().__init__()
        self.smiles_max_lengths = smiles_max_lengths
        self.unimol_model_name = unimol_model_name
        self.unimol_model_size = unimol_model_size
        self.morgan_featurizer = MorganFeaturizer()
        self.smiles_featurizer = SimpleSmilesTokenFeaturizer(smiles_max_lengths=smiles_max_lengths)
        self.struct_featurizer = UnimolFeaturizer(unimol_model_size=unimol_model_size, unimol_model_name=unimol_model_name)

    def _featurize(self, datapoint: Any, **kwargs):
        morgan = self.morgan_featurizer.featurize([datapoint])[0]
        tokens = self.smiles_featurizer.featurize([datapoint])[0]
        struct = self.struct_featurizer.featurize([datapoint])[0]
        forward_graph, backward_graph = serailize_graph(struct.graph, batch_key=None, keys=['node_features', 'graph_features', 'pos'])
        ans = FeatData(embedding=morgan.embedding, input_ids=tokens.input_ids,
                       graph=forward_graph, prefix='ligand')
        return ans

    def get_feat_info(self, data=None):
        ans = super().get_feat_info(data=data)
        ans['num_ligand_tokens'] = self.smiles_featurizer.tokenizer.get_vocab_size()
        return ans


class FusionLigandFeaturizer2(FeaturizerBase):
    def __init__(self, smiles_max_lengths=100):
        super().__init__()
        self.smiles_max_lengths = smiles_max_lengths
        self.morgan_featurizer = MorganFeaturizer()
        self.smiles_featurizer = SimpleSmilesTokenFeaturizer(smiles_max_lengths=smiles_max_lengths)
        self.struct_featurizer = GraphMVP3dPointFeaturizer()

    def _featurize(self, datapoint: Any, **kwargs):
        morgan = self.morgan_featurizer.featurize([datapoint])[0]
        tokens = self.smiles_featurizer.featurize([datapoint])[0]
        struct = self.struct_featurizer.featurize([datapoint])[0]
        forward_graph, backward_graph = serailize_graph(struct.graph, batch_key=None, keys=['node_features', 'pos'])
        ans = FeatData(embedding=morgan.embedding, input_ids=tokens.input_ids,
                       graph=forward_graph, prefix='ligand')
        return ans

    def get_feat_info(self, data=None):
        ans = super().get_feat_info(data=data)
        ans['num_ligand_tokens'] = self.smiles_featurizer.tokenizer.get_vocab_size()
        return ans

class FusionLigandFeaturizer3(FusionLigandFeaturizer2):
    def __init__(self, smiles_max_lengths=100):
        super().__init__(smiles_max_lengths=smiles_max_lengths)

    def _featurize(self, datapoint: Any, **kwargs):
        morgan = self.morgan_featurizer.featurize([datapoint])[0]
        tokens = self.smiles_featurizer.featurize([datapoint])[0]
        struct = self.struct_featurizer.featurize([datapoint])[0]
        forward_graph, backward_graph = serailize_graph(struct.graph, batch_key=None, keys=['raw_features', 'pos'])
        forward_graph.node_features = forward_graph.raw_features
        ans = FeatData(embedding=morgan.embedding, input_ids=tokens.input_ids,
                       graph=forward_graph, prefix='ligand')
        return ans

class FusionProteinFeaturizer(FeaturizerBase):
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR, struct_type='esmfold', pocket_type='fpocket', pocket_tops=3,
                 pretrained_model_name_or_path="Rostlab/prot_bert", feat_type='cls'):
        super().__init__()
        self.struct_root_dir = struct_root_dir
        self.struct_type = struct_type
        self.pocket_type = pocket_type
        self.pocket_tops = pocket_tops
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.feat_type = feat_type
        self.struct_featurizer = LLMStructFeaturizer(struct_root_dir=struct_root_dir, struct_type=struct_type, pocket_type=pocket_type,
                                                     pocket_tops=pocket_tops, pretrained_model_name_or_path=pretrained_model_name_or_path)
        self.seq_featurizer = LLMFeaturizer(pretrained_model_name_or_path=pretrained_model_name_or_path, feat_type=feat_type)

    def _featurize(self, datapoint, **kwargs):
        struct = self.struct_featurizer.featurize([datapoint])[0]
        cls = self.seq_featurizer.featurize([datapoint])[0]
        graph = struct.graph
        graph.pos = graph.pos[:,1]
        forward_graph, backward_graph = serailize_graph(graph, batch_key='node_batch', keys=['node_features', 'pos', 'node_batch', 'pocket_batch'])
        return FeatData(embedding=cls.embedding, graph=forward_graph, prefix='protein')



class FusionDataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = FusionLigandFeaturizer
    protein_featurizer_cls = FusionProteinFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.to_pandas()
        self.to_tensor()

class FusionDataset2(datamodule.DTADatasetBase):
    ligand_featurizer_cls = FusionLigandFeaturizer2
    protein_featurizer_cls = FusionProteinFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.to_pandas()
        self.to_tensor()

class FusionDataset3(datamodule.DTADatasetBase):
    ligand_featurizer_cls = FusionLigandFeaturizer3
    protein_featurizer_cls = FusionProteinFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.to_pandas()
        self.to_tensor()

if __name__ == '__main__':
    from toolbox.datamodule import DataModule
    from tqdm import tqdm
    from lightning.pytorch.utilities import move_data_to_device
    import argparse

    dataset_cls = FusionDataset
    parser = argparse.ArgumentParser()
    DataModule.add_parser_arguments(parser)
    dataset_cls.add_parser_arguments(parser)
    args = parser.parse_args()
    args.dataset_name = 'kiba'
    args.dataset_name = 'kiba_pocketdta'
    args.dataset_name = 'davis'
    args.pocket_tops = 2
    datamodule = DataModule(vars(args), dataset_cls=dataset_cls)
    datamodule.prepare_data()
    datamodule.setup()
    # data = datamodule.train_dataloader().dataset
    # data.affinities.data.combine_chunks()
    # data.input_columns = ['PID', 'ligand_input_ids']
    # import datasets
    # datasets.Dataset.to

    for batch in tqdm(datamodule.train_dataloader()):
        move_data_to_device(batch, 'cuda')
        pass

    for dataset in datamodule.split_datasets():
        for batch in tqdm(dataset.train_dataloader()):
            pass