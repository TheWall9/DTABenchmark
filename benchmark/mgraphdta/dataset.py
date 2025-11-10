import os
import abc

import torch
from enum import Enum
import numpy as np
import networkx as nx
import tokenizers as tk

import rdkit
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import ValenceType
from rdkit import RDConfig

from torch_geometric.loader import DataLoader

import deepchem as dc
from deepchem.feat import MolecularFeaturizer
from deepchem.feat.graph_data import GraphData
from deepchem.feat.graph_features import one_of_k_encoding_unk, one_of_k_encoding

from toolbox import datamodule
from toolbox.featurizer.ligand import MolFeaturizerBase
from toolbox.featurizer.protein import SimpleProtTokenFeaturizer

from deepchem.feat.graph_features import atom_features


class MGraphDTAMolFeaturizer(MolFeaturizerBase):
    def __init__(self, use_original_atoms_order=False, atom_feature_families=('Donor', 'Acceptor')):
        super().__init__(use_original_atoms_order, atom_feature_families=atom_feature_families)

    def get_atom_features_fn(self, atom):
        feat = [int(atom.GetSymbol() in ['H', 'C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I', ]),
                atom.GetAtomicNum(), atom.GetIsAromatic(),
                int(atom.GetHybridization() in (Chem.rdchem.HybridizationType.SP,
                                                Chem.rdchem.HybridizationType.SP2,
                                                Chem.rdchem.HybridizationType.SP3)),
                atom.GetTotalNumHs(),
                atom.GetValence(ValenceType.EXPLICIT), atom.GetFormalCharge(), atom.GetValence(ValenceType.IMPLICIT),
                atom.GetNumExplicitHs(), atom.GetNumRadicalElectrons(),
                ]
        feat = np.asarray(feat, dtype=np.float32)
        feat = (feat-feat.min())/(feat.max()-feat.min())
        return feat

    def get_atom_features_fn2(self, atom):
        feat = atom_features(atom, bool_id_feat=True, explicit_H=True, use_chirality=True)
        return feat/feat.sum()

    def get_bond_features_fn(self, bond):
        feat = [int(bond.GetBondType() in (Chem.rdchem.BondType.SINGLE,
                                           Chem.rdchem.BondType.DOUBLE,
                                           Chem.rdchem.BondType.TRIPLE,
                                           Chem.rdchem.BondType.AROMATIC)),
                int(bond.GetIsConjugated())]
        return feat


class MGraphDTADataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = MGraphDTAMolFeaturizer
    protein_featurizer_cls = SimpleProtTokenFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.dataloader_cls = DataLoader
        self.to_pandas()
        self.to_tensor()

    # @classmethod
    # def preprocess_proteins(cls, proteins, config):
    #     protein_featurizer = cls.protein_featurizer_cls.build_featurizer(**config)
    #     key = "Protein_Sequence"
    #     protein_featurizer.sanitize(proteins[0][key])
    #     proteins = proteins.map(protein_featurizer.datasets_map_fn, batched=True, fn_kwargs={"col":key},
    #                             desc='tokenize proteins')
    #     return proteins, protein_featurizer.get_feat_info()

    # @classmethod
    # def preprocess_ligands(cls, ligands, config):
    #     ligand_featurizer = cls.ligand_featurizer_cls.build_featurizer(**config)
    #     key = 'ISO_SMILES'
    #     ligand_featurizer.sanitize(ligands[0][key])
    #     smiles = ligands.map(ligand_featurizer.datasets_map_fn, batched=True, fn_kwargs={"col":key,
    #                                                                                      "feat_name":"ligand_graph"},
    #                          desc='tokenize ligands')
    #     return smiles, ligand_featurizer.get_feat_info()

    # @classmethod
    # def preprocess_affinities(cls, ligands, proteins, affinities, config):
    #     ligands_key_map = {"ligand_graph": "ligand_graph"}
    #     proteins_key_map = {"protein_input_ids": "protein_input_ids"}
    #     affinities = cls.merge_datasets(ligands, proteins, affinities, ligands_key_map, proteins_key_map)
    #     affinities = affinities.to_pandas()
    #     return affinities, {}

    @classmethod
    def add_parser_arguments(cls, parser):
        cls.ligand_featurizer_cls.add_argparser_args(parser)
        cls.protein_featurizer_cls.add_argparser_args(parser)
        parser.set_defaults(batch_size=512, protein_max_lengths=1200)

    # def __getitem__(self, idx):
    #     affinity = self.select_affinity(idx)
    #     affinity = self.to_pgy_data(affinity)
    #     if self.input_columns is not None:
    #         affinity = {key: affinity[key] for key in self.input_columns}
    #     return affinity


if __name__=="__main__":
    from toolbox.datamodule import DataModule
    from tqdm import tqdm
    import argparse
    dataset_cls = MGraphDTADataset
    parser = argparse.ArgumentParser()
    DataModule.add_parser_arguments(parser)
    dataset_cls.add_parser_arguments(parser)
    args = parser.parse_args()
    datamodule = DataModule(vars(args), dataset_cls=dataset_cls)
    datamodule.prepare_data()
    datamodule.setup()
    for batch in tqdm(datamodule.train_dataloader()):
        pass

    for dataset in datamodule.split_datasets():
        for batch in tqdm(dataset.train_dataloader()):
            pass

    """5s"""