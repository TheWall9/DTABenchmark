import os
import copy
import logging
from typing import Dict, List, Any
from collections import defaultdict
import numpy as np

import torch
import tokenizers as tk
from rdkit import RDConfig, Chem, DataStructs
from rdkit.Chem import ChemicalFeatures, rdFingerprintGenerator
from rdkit.Chem import ValenceType, AllChem
from deepchem.feat.graph_features import one_of_k_encoding_unk, one_of_k_encoding

from deepchem import feat
from toolbox.featurizer.tools import FeaturizerBase, SmilesFeaturizerBase, FeaturizerWrapper, FeatData, GraphData
from toolbox.utils import Hasher, disk_cache
from toolbox.config import THIRD_PARTY_CHECKPOINTS_DIR

from repo.graphmvp.molecule_gnn_model import GNNComplete
from repo.graphmvp.molecule_datasets import mol_to_graph_data_obj_simple

CHEM_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SimpleSmilesTokenFeaturizer(SmilesFeaturizerBase):
    def __init__(self, use_original_atoms_order=False, isomeric_smiles=True, smiles_max_lengths=100):
        super().__init__(use_original_atoms_order)
        tokenizer_file = os.path.join(self.tokenizer_dir, f"DeepDTA_{'ISO' if isomeric_smiles else 'CAN'}SMILES_tokenizer.json")
        tokenizer = tk.Tokenizer.from_file(tokenizer_file)
        if smiles_max_lengths is not None:
            tokenizer.enable_truncation(max_length=smiles_max_lengths)
            tokenizer.enable_padding(length=smiles_max_lengths)
        else:
            tokenizer.no_padding()
            tokenizer.no_truncation()
        self.isometric_smiles = isomeric_smiles
        self.smiles_max_lengths = smiles_max_lengths
        self.tokenizer = tokenizer

    def _featurize(self, datapoint: Any, **kwargs):
        smiles = Chem.MolToSmiles(datapoint, isomericSmiles=True)
        if self.smiles_max_lengths is not None:
            return FeatData(input_ids=self.tokenizer.encode(smiles).ids, prefix='ligand')
        else:
            tokens = self.tokenizer.encode(smiles)
            input_ids_batch = np.zeros(len(tokens.ids), dtype=int)
            return FeatData(graph=GraphData(input_ids=tokens.ids, input_ids_batch=input_ids_batch), prefix='ligand')

    def get_feat_info(self, data=None):
        return {"num_ligand_tokens": self.tokenizer.get_vocab_size()}


class MolFeaturizerBase(SmilesFeaturizerBase):
    tokenizer_dir = os.path.join(os.path.dirname(__file__), "tokenizer")
    def __init__(self, use_original_atoms_order=False, atom_feature_families=tuple(),
                 node_feat_fn=None, edge_feat_fn=None, graph_consructor_fn=None,
                 get_atom_feat_fn=None, get_bond_feat_fn=None, get_graph_feat_fn=None,):
        super().__init__(use_original_atoms_order)
        self.atom_feature_families = atom_feature_families
        self.node_featurizer = node_feat_fn or self._node_featurizer_fn
        self.edge_featurizer = edge_feat_fn or self._edge_featurizer_fn
        self.graph_constructor = graph_consructor_fn or self._digraph_constructor_fn
        self.get_atom_features = get_atom_feat_fn or self.get_atom_features_fn
        self.get_bond_features = get_bond_feat_fn or self.get_bond_features_fn
        self.get_graph_features = get_graph_feat_fn or self.get_graph_features_fn


    def get_atom_features_fn(self, atom):
        pass

    def get_bond_features_fn(self, bond):
        pass

    def get_graph_features_fn(self, mol):
        pass

    def _node_featurizer_fn(self, mol, **kwargs):
        if self.get_atom_features(next(iter(mol.GetAtoms()))) is None:
            return np.empty((mol.GetNumAtoms(), 0))
        feats = []
        for atom in mol.GetAtoms():
            feats.append(self.get_atom_features(atom))
        feats = np.asarray(feats, dtype=float)
        extra_feats = self._get_mol_family_features(mol, self.atom_feature_families)
        if extra_feats is not None:
            feats = np.concatenate((feats, extra_feats), axis=-1)
        return feats

    def _edge_featurizer_fn(self, mol, **kwargs):
        if self.get_bond_features(next(iter(mol.GetBonds()))) is None:
            return None
        feats = []
        for bond in mol.GetBonds():
            feats.append(self.get_bond_features(bond))
        feats = np.asarray(feats, dtype=float)
        return feats

    def _digraph_constructor_fn(self, mol, **kwargs):
        edge_index = []
        for bond in mol.GetBonds():
            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_index = np.asarray(edge_index).transpose()
        node_feature = np.empty((mol.GetNumAtoms(), 0))
        extra_feats = {}
        graph_features = self.get_graph_features(mol)
        if graph_features is not None:
            if isinstance(graph_features, Dict):
                extra_feats.update(graph_features)
            else:
                extra_feats['graph_features'] = np.asarray(graph_features, dtype=float)
        return mol, GraphData(node_features=node_feature, edge_index=edge_index, **extra_feats)

    def _get_mol_family_features(self, mol, valid_family=tuple()):
        if valid_family is not None:
            features = CHEM_FEATURE_FACTORY.GetFeaturesForMol(mol)
            extra_atom_feature = np.zeros((mol.GetNumAtoms(), len(valid_family)), dtype=float)
            for feature in features:
                family = feature.GetFamily()
                if family in valid_family:
                    idx = valid_family.index(family)
                    node_list = feature.GetAtomIds()
                    extra_atom_feature[node_list, idx] = 1
            return extra_atom_feature

    def _featurize(self, mol, **kwargs):
        mol, g = self.graph_constructor(mol, **kwargs)
        if self.node_featurizer is not None:
            node_features = self.node_featurizer(mol, **kwargs)
            if g.node_features is not None:
                g.node_features = np.concatenate((g.node_features, node_features), axis=-1)
            else:
                g.node_features = node_features
        if self.edge_featurizer is not None:
            edge_features = self.edge_featurizer(mol, **kwargs)
            if g.edge_features is not None:
                g.edge_features = np.concatenate((g.edge_features, edge_features), axis=-1)
            else:
                g.edge_features = self.edge_featurizer(mol, **kwargs)
        return FeatData(graph=g, prefix='ligand')

    def mol_to_3d(self, mol, optimize=True):
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        cid = AllChem.EmbedMolecule(mol, params=params)
        if cid==-1:
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol, params=params)
        if cid == -1:
            raise RuntimeError("无法生成3D构象")
        if optimize:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol)
            ff.Initialize()
            ff.Minimize(maxIts=1000)
        return mol

class Mol2VecFeaturizer(feat.Mol2VecFingerprint):
    def __init__(self, pretrain_model_path=None, radius=1, unseen='UNK'):
        super().__init__(pretrain_model_path, radius=radius, unseen=unseen)
        self.pretrain_model_path = pretrain_model_path
        self.feat_info = {}

    def _featurize(self, datapoint, **kwargs):
        if isinstance(datapoint, str):
            smiles = Chem.MolToSmiles(Chem.MolFromSmiles(datapoint), isomericSmiles=False)
            datapoint = Chem.MolFromSmiles(smiles)
        ans = super()._featurize(datapoint, **kwargs)
        return FeatData(inputs_embeds=ans, prefix='ligand')

    def sanitize(self, item):
        ans = self.featurize([item])[0]
        self.feat_info = self.get_feat_info(ans)
        return ans

    def get_feat_info(self, data=None):
        if data is None:
            return self.feat_info
        else:
            return data.get_feat_info()



class GraphMVPFeaturizer(SmilesFeaturizerBase):
    def __init__(self, use_original_atoms_order=False, device='cuda', ligand_max_seq_length=100):
        super().__init__(use_original_atoms_order)
        self.device = device
        self.pretrain_model_path = os.path.join(THIRD_PARTY_CHECKPOINTS_DIR, "GraphMVP", "GraphMVP_complate_features_for_regression",
                                                "GraphMVP", "pretraining_model.pth")
        self.ligand_max_seq_length = ligand_max_seq_length
        self.emb_dim = 300
        self.model = None
        self.disk_cache_dir = os.path.join(self.FEATURIZER_OUTPUT_TEMP_DIR, "GraphMVPFeaturizer", f'max_seq_{ligand_max_seq_length}')
        os.makedirs(self.disk_cache_dir, exist_ok=True)


    def load_model(self, pretrain_model_path):
        assert os.path.exists(pretrain_model_path)
        model = GNNComplete(num_layer=5, emb_dim=self.emb_dim)
        model.load_state_dict(torch.load(pretrain_model_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        return model

    @disk_cache
    def _featurize(self, datapoint, **kwargs):
        if not hasattr(self, 'model') or self.model is None:
            self.model = self.load_model(self.pretrain_model_path)
        data = mol_to_graph_data_obj_simple(datapoint)
        feats = torch.zeros((self.ligand_max_seq_length,self.emb_dim))
        valid_mask = torch.zeros(self.ligand_max_seq_length, dtype=torch.int)
        with torch.no_grad():
            graph_embedding = self.model(data.to(self.device)).cpu()
        if graph_embedding.shape[0] >= self.ligand_max_seq_length:
            feats = graph_embedding[:self.ligand_max_seq_length,:]
            valid_mask[:self.ligand_max_seq_length] = 1
        else:
            feats[:graph_embedding.shape[0],:] = graph_embedding  #36*300
            valid_mask[:graph_embedding.shape[0]] = 1
        feats = feats.numpy()
        return FeatData(inputs_embeds=feats, attention_mask=valid_mask.numpy(), prefix='ligand')




class MorganFeaturizer(SmilesFeaturizerBase):
    def __init__(self, use_original_atoms_order=False, radius=2, fpSize=2048):
        super().__init__(use_original_atoms_order)
        self.radius = radius
        self.fpSize = fpSize

    def _featurize(self, datapoint, **kwargs):
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.fpSize)
        fp = mfpgen.GetFingerprint(datapoint)
        feat = np.zeros((fp.GetNumBits(),))
        DataStructs.ConvertToNumpyArray(fp, feat)
        return FeatData(embedding=feat, prefix='ligand')



class UnimolFeaturizer(MolFeaturizerBase):
    def __init__(self, use_original_atoms_order=False, atom_feature_families=tuple(),
                 node_feat_fn=None, edge_feat_fn=None, graph_consructor_fn=None,
                 get_atom_feat_fn=None, get_bond_feat_fn=None, get_graph_feat_fn=None,
                 unimol_model_name='unimolv1', unimol_model_size='84m'):
        super().__init__(use_original_atoms_order, atom_feature_families, node_feat_fn, edge_feat_fn, graph_consructor_fn,
                         get_atom_feat_fn, get_bond_feat_fn, get_graph_feat_fn)
        self.model = None
        # self.enable_cache(self.cached_root_output_dir, overwrite=False)
        self.disk_cache_dir = os.path.join(self.FEATURIZER_OUTPUT_TEMP_DIR, 'UnimolFeaturizer')
        self.unimol_model_name = unimol_model_name
        self.unimol_model_size = unimol_model_size
        os.makedirs(self.disk_cache_dir, exist_ok=True)

    def load_model(self):
        from toolbox.config import UNIMOL_WEIGHT_DIR
        from unimol_tools import UniMolRepr
        from unimol_tools.utils import logger
        logger.setLevel(logging.ERROR)
        clf = UniMolRepr(data_type='molecule', remove_hs=False, model_name=self.unimol_model_name, model_size=self.unimol_model_size)
        clf.params['max_atoms'] = 1000
        clf.params['multi_process'] = False
        return clf

    @disk_cache
    def _digraph_constructor_fn(self, mol, **kwargs):
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol = Chem.MolFromSmiles(smiles)
        mol = self.mol_to_3d(mol)
        mol, g = super()._digraph_constructor_fn(mol, **kwargs)

        if not hasattr(self, 'model') or self.model is None:
            self.model = self.load_model()
        unimol_repr = self.model.get_repr([smiles], return_atomic_reprs=True)
        assert "".join(unimol_repr['atomic_symbol'][0])=="".join(atom.GetSymbol() for atom in mol.GetAtoms())

        graph = GraphData(node_features=unimol_repr['atomic_reprs'][0],
                          pos=unimol_repr['atomic_coords'][0],
                          graph_features=np.array(unimol_repr['cls_repr']),
                          edge_index=g.edge_index,
                          atom_type=unimol_repr['atomic_symbol'][0],
                          atomic_num=np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
                           **g.kwargs)
        return mol, graph


class Mol3dGraphFeaturizer(MolFeaturizerBase):
    def __init__(self, use_original_atoms_order=False):
        super().__init__(use_original_atoms_order)
        self.disk_cache_dir = os.path.join(self.FEATURIZER_OUTPUT_TEMP_DIR, "Mol3dGraphFeaturizer")
        os.makedirs(self.disk_cache_dir, exist_ok=True)

    def get_atom_features_fn(self, atom, **kwargs):
        feat = np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                              ['H', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                                               'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                               'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                               'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetValence(ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])
        return feat / sum(feat)

    @disk_cache
    def _digraph_constructor_fn(self, mol, **kwargs):
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol = Chem.MolFromSmiles(smiles)
        mol, coordinates = self.mol_to_3d(mol)
        mol, g = super()._digraph_constructor_fn(mol, **kwargs)
        g.pos = coordinates
        g.edge_index = np.concatenate([g.edge_index, g.edge_index[::-1]], axis=0)
        return mol, g

    def mol_to_3d(self, mol, optimize=True):
        mol = Chem.AddHs(mol)
        cid = -1
        for seed in list(range(100))+[9999, ]:
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol, params=params)
            if cid!=-1:
                break

        if cid == -1:
            raise RuntimeError("无法生成3D构象")
        # 力场优化（仅针对这一个构象）
        if optimize:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol)
            ff.Initialize()
            ff.Minimize(maxIts=1000)
        conf = mol.GetConformer(0)
        coordinates = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        return mol, coordinates


class GraphMVP3dPointFeaturizer(SmilesFeaturizerBase):
    def __init__(self, use_original_atoms_order=False, device='cuda'):
        super().__init__(use_original_atoms_order)
        self.device = device
        self.pretrain_model_path = os.path.join(THIRD_PARTY_CHECKPOINTS_DIR, "GraphMVP",
                                                "GraphMVP_complate_features_for_regression",
                                                "GraphMVP", "pretraining_model.pth")
        self.emb_dim = 300
        self.model = None
        self.disk_cache_dir = os.path.join(self.FEATURIZER_OUTPUT_TEMP_DIR, "GraphMVP3dPointFeaturizer")
        os.makedirs(self.disk_cache_dir, exist_ok=True)

    def load_model(self, pretrain_model_path):
        assert os.path.exists(pretrain_model_path)
        model = GNNComplete(num_layer=5, emb_dim=self.emb_dim)
        model.load_state_dict(torch.load(pretrain_model_path, map_location=self.device, weights_only=True))
        model.to(self.device)
        model.eval()
        return model

    @disk_cache
    def _featurize(self, datapoint, **kwargs):
        if not hasattr(self, 'model') or self.model is None:
            self.model = self.load_model(self.pretrain_model_path)
        mol, coordinates = self.mol_to_3d(datapoint)
        raw_features = []
        for atom in mol.GetAtoms():
            raw_features.append(self.get_atom_features_fn(atom))
        raw_features = np.stack(raw_features)
        data = mol_to_graph_data_obj_simple(mol)
        with torch.no_grad():
            graph_embedding = self.model(data.to(self.device)).cpu()
        mvp_graph = GraphData(node_features=graph_embedding.cpu().numpy(), edge_index=data.edge_index.cpu().numpy(),
                              edge_features=data.edge_attr.cpu().numpy(), pos=coordinates,
                              raw_features=raw_features)
        return FeatData(graph=mvp_graph, prefix='ligand')

    def mol_to_3d(self, mol, optimize=True):
        mol = Chem.AddHs(mol)
        cid = -1
        for seed in list(range(100))+[9999, ]:
            params = AllChem.ETKDGv3()
            params.randomSeed = seed
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol, params=params)
            if cid!=-1:
                break

        if cid == -1:
            raise RuntimeError("无法生成3D构象")
        # 力场优化（仅针对这一个构象）
        if optimize:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
            else:
                ff = AllChem.UFFGetMoleculeForceField(mol)
            ff.Initialize()
            ff.Minimize(maxIts=1000)
        conf = mol.GetConformer(0)
        coordinates = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
        return mol, coordinates

    def get_atom_features_fn(self, atom, **kwargs):
        feat = np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                              ['H', 'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                                               'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                               'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                               'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetValence(ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])
        return feat / sum(feat)