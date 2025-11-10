import os
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from typing import List, Mapping, Iterable, Dict

import torch
from torch.utils.data import default_collate
from torch_geometric import data as gdata
from torch_geometric.loader import DataLoader

import datasets

from toolbox.datamodule.data_loader import load_data


class TorchFormatter:

    def recursive_tensorize(self, data_struct):
        if isinstance(data_struct, np.ndarray):
            if data_struct.dtype==object:
                demo = data_struct[0]
                if isinstance(demo, np.ndarray):
                    if np.issubdtype(demo.dtype, np.number):
                        return self._tensorize(np.stack(data_struct))
                    elif isinstance(demo[0], np.ndarray) and np.issubdtype(demo[0].dtype, np.number):
                        return self._tensorize(np.stack([np.stack(item) for item in data_struct]))
                elif isinstance(demo, (str,)):
                    return data_struct.tolist()
                elif isinstance(demo, Dict):
                    return [self.recursive_tensorize(substruct) for substruct in data_struct]
                return self._consolidate([self.recursive_tensorize(item) for item in data_struct]) # skip
        elif isinstance(data_struct, Dict):
            return {key:self.recursive_tensorize(value) for key, value in data_struct.items()}
        elif isinstance(data_struct, (list, tuple)): # skip
            return self._consolidate([self.recursive_tensorize(substruct) for substruct in data_struct])
        elif isinstance(data_struct, (gdata.Data, torch.Tensor)):
            return data_struct
        return self._tensorize(data_struct)

    def _consolidate(self, column):
        if isinstance(column, list) and column:
            if not isinstance(column[0], torch.Tensor):
                return column
            if all(
                isinstance(x, torch.Tensor) and x.shape == column[0].shape and x.dtype == column[0].dtype
                for x in column
            ):
                return torch.stack(column)
        return column

    def _tensorize(self, value):
        if isinstance(value, (str, bytes, type(None))):
            return value
        elif isinstance(value, (np.character, np.ndarray)) and np.issubdtype(value.dtype, np.character):
            return value.tolist()
        default_dtype = {}
        if isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.integer):
            default_dtype = {"dtype": torch.int64}
            # Convert dtype to np.int64 if it's either np.uint16 or np.uint32 to ensure compatibility.
            # np.uint64 is excluded from this conversion as there is no compatible PyTorch dtype that can handle it without loss.
            if value.dtype in [np.uint16, np.uint32]:
                value = value.astype(np.int64)
        elif isinstance(value, (np.number, np.ndarray)) and np.issubdtype(value.dtype, np.floating):
            default_dtype = {"dtype": torch.float32}
        return torch.tensor(value, **default_dtype)

    def to_pgy_data(self, data: Dict, to_list=False):
        for col in data:
            tmp = data[col]
            if isinstance(tmp, List) and isinstance(next(iter(tmp)), Dict):
            # if "graph" in col:
            #     tmp = data[col]
                if isinstance(tmp, Mapping):
                    tmp = gdata.Data(**tmp)
                elif isinstance(tmp, pd.Series):
                    tmp = np.array([gdata.Data(**{key:np.asarray(graph[key].tolist()) for key in graph}) for graph in tmp], dtype=object)
                elif isinstance(tmp, (List, np.ndarray)):
                    if isinstance(tmp[0], gdata.Data):
                        continue
                    ans = [None]*len(tmp) if to_list else np.empty(len(tmp), dtype=object)
                    for i, graph in enumerate(tmp):
                        ans[i] = gdata.Data(**graph)
                    tmp = ans
                data[col] = tmp
        return data



class DatasetsMixin(TorchFormatter):
    def __init__(self, **datasets):
        for key, value in datasets.items():
            setattr(self, key, value)
        self._data_keys = list(datasets.keys())

    def save_to_disk(self, save_path, type='arrow'):
        assert type in ['arrow', 'torch']
        os.makedirs(save_path, exist_ok=True)
        for k in self._data_keys:
            save_file = os.path.join(save_path, k if type == 'arrow' else f'{k}.pt')
            data = getattr(self, k)
            if isinstance(data, datasets.Dataset):
                if type == 'arrow':
                    data.save_to_disk(save_file)
                elif type == 'torch':
                    data = data.with_format(type='torch')
                    data = {key:data[key] for key in data.column_names}
                    for key, value in data.items():
                        if not isinstance(value, torch.Tensor):
                            data[key] = np.asarray(value)
                    torch.save(data, save_file)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

    def load_from_disk(self, load_path, keep_in_memory=False, type='arrow'):
        assert type in ['arrow', 'torch']
        ans = {}
        for file in os.listdir(load_path):
            load_file = os.path.join(load_path, file)
            if 'torch'==type and load_file.endswith('.pt'):
                ans[file.replace('.pt', '', 1)] = torch.load(load_file, weights_only=True)
            elif 'arrow'==type and os.path.exists(os.path.join(load_file, 'dataset_info.json')):
                ans[file] = datasets.Dataset.load_from_disk(load_file, keep_in_memory=keep_in_memory)
        for k in ans:
            delattr(self, k)
        self._data_keys = list(ans.keys())
        for k, v in ans.items():
            setattr(self, k, v)

    def select_data(self, data, idx):
        if isinstance(idx, int):
            idx = [idx]
        elif isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        if isinstance(data, datasets.Dataset):
            dataset = data.select(idx)
        elif isinstance(data, pd.DataFrame):
            dataset = data.iloc[idx]
        elif isinstance(data, List):
            dataset = [data[i] for i in idx]
        elif isinstance(data, Mapping):
            dataset = {key:data[key][idx] for key in data}
        else:
            raise NotImplementedError
        return dataset

    def select_columns(self, data, cols):
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(data, datasets.Dataset):
            ans = data.select_columns(cols)
        elif isinstance(data, pd.DataFrame):
            ans = data.loc[:, cols]
        elif isinstance(data, Mapping):
            ans = {key:data[key] for key in cols}
        else:
            raise NotImplementedError
        return ans

    def to_tensor(self):
        for k in tqdm(self._data_keys, desc="to_tensor"):
            data = getattr(self, k)
            if isinstance(data, datasets.Dataset):
                data = data.with_format(type='torch')
                data = {key:data[key] for key in data.column_names}
                for key, value in data.items():
                    if not isinstance(value, torch.Tensor):
                        data[key] = np.asarray(value)
                data = self.to_pgy_data(data)
                setattr(self, k, data)
            elif isinstance(data, pd.DataFrame):
                data = {key:data[key].values for key in data.columns}
                data = self.recursive_tensorize(data)
                data = self.to_pgy_data(data)
                for key, value in data.items():
                    if not isinstance(value, torch.Tensor):
                        data[key] = np.asarray(value)
                setattr(self, k, data)

    def to_pandas(self):
        for k in self._data_keys:
            data = getattr(self, k)
            if isinstance(data, datasets.Dataset):
                data = data.to_pandas()
                setattr(self, k, data)
            elif not isinstance(data, pd.DataFrame):
                raise NotImplementedError

    def to_datasets(self):
        for k in self._data_keys:
            data = getattr(self, k)
            if isinstance(data, pd.DataFrame):
                data = datasets.Dataset.from_pandas(data)
                setattr(self, k, data)
            elif not isinstance(data, datasets.Dataset):
                raise NotImplementedError

    def get_data_columns(self):
        columns = {}
        for k in self._data_keys:
            data = getattr(self, k)
            if isinstance(data, datasets.Dataset):
                cols = list(data.column_names)
            elif isinstance(data, pd.DataFrame):
                cols = list(data.columns)
            elif isinstance(data, Mapping):
                cols = list(data.keys())
            else:
                raise NotImplementedError
            columns[k] = cols
        return columns




class BindingDatasetMixin(DatasetsMixin):
    def __init__(self, datasets, relation_dataset_name="affinities",
                 foreign_key_map={"ligands":"LID", "proteins":"PID"},
                 input_columns=None):
        super().__init__(**datasets)
        assert set(foreign_key_map.keys()).issubset(set(datasets.keys()))
        assert relation_dataset_name in datasets
        self.relation_dataset_name = relation_dataset_name
        self.foreign_key_map = foreign_key_map
        self.input_columns = input_columns
        self.ordered_dataset_names = [relation_dataset_name]+list(foreign_key_map.keys())
        # print("keep columns:", self.keep_columns)
        for name, cols in self.keep_columns.items():
            data = self.select_columns(getattr(self, name), cols)
            setattr(self, name, data)

    @property
    def keep_columns(self):
        return self.get_keep_columns(self.input_columns)

    def get_keep_columns(self, columns=None):
        columns_map = self.get_data_columns()
        if columns is None:
            keep_columns = []
            for col in columns_map.values():
                keep_columns.extend(col)
            keep_columns = set(keep_columns)
        else:
            keep_columns = set(list(self.foreign_key_map.values())+list(columns))
        ans = {}
        for name in self.ordered_dataset_names:
            ans[name] = sorted(keep_columns.intersection(columns_map[name]))
            keep_columns = keep_columns.difference(columns_map[name])
        return ans

    def index_select(self, indices):
        if indices is None:
            return self
        data = getattr(self, self.relation_dataset_name)
        new_data = self.select_data(data, indices)

        datasets = {key:getattr(self, key) for key in self._data_keys}
        datasets[self.relation_dataset_name] = new_data
        ans =  self.__class__(datasets, relation_dataset_name=self.relation_dataset_name,
                              foreign_key_map=self.foreign_key_map, input_columns=self.input_columns)
        return ans

    def select(self, indices):
        return self.index_select(indices)

    def filter_cols(self, data_list, cols):
        ans = {}
        for col in cols:
            for data in data_list:
                if col in data and col not in ans:
                    ans[col] = data[col]
        return ans


    def __len__(self):
        data = getattr(self, self.relation_dataset_name)
        if isinstance(data, Mapping):
            return len(next(iter(data.values())))
        return len(data)

    def get(self, idx):
        r = self.select_data(getattr(self, self.relation_dataset_name), idx)
        data_list = [r]
        for name, key in self.foreign_key_map.items():
            data = self.select_data(getattr(self, name), r[key])
            data_list.append(data)
        ans = self.merge_data(data_list)
        return ans

    def __getitem__(self, idx):
        return self.get(idx)

    def merge_data(self, data_list):
        ans = {}
        for data in data_list:
            if isinstance(data, datasets.Dataset):
               for col in data.column_names:
                   ans[col] = data[col]
            elif isinstance(data, pd.DataFrame):
                for col in data.columns:
                    ans[col] = data[col].values
            elif isinstance(data, Mapping):
                for col in data.keys():
                    ans[col] = data[col]
            else:
                raise NotImplementedError
        return ans


class DTAProcessMixin():
    protein_featurizer_cls = None
    ligand_featurizer_cls = None

    @classmethod
    def add_parser_arguments(cls, parser):
        cls.ligand_featurizer_cls.add_argparser_args(parser)
        cls.protein_featurizer_cls.add_argparser_args(parser)


    @classmethod
    def datasets_map_fn(cls, examples, featurizer, col=None, feat_name=None):
        key = col or featurizer.example_key
        feats = featurizer.featurize(examples[key])
        ans = featurizer.to_datasets_example(feats, feat_name)
        return ans

    @classmethod
    def preprocess_ligands(cls, ligands, config):
        return ligands, {}

    @classmethod
    def preprocess_proteins(cls, proteins, config):
        return proteins, {}

    @classmethod
    def preprocess_affinities(cls, ligands, proteins, affinities, config):
        return affinities, {}

    @classmethod
    def preprocess(cls, ligands, proteins, affinities, config, **kwargs):
        info = {}
        ligands, ligand_info = cls.preprocess_ligands(ligands, config)
        proteins, protein_info = cls.preprocess_proteins(proteins, config)
        affinities, affinity_info = cls.preprocess_affinities(ligands, proteins, affinities, config)
        info.update(ligand_info)
        info.update(protein_info)
        info.update(affinity_info)
        info["num_proteins"] = len(proteins)
        info["num_ligands"] = len(ligands)
        info['affinity_mean'] = np.mean(affinities['affinity'])
        info['affinity_std'] = np.std(affinities['affinity'])
        info['affinity_max'] = np.max(affinities['affinity'])
        info['affinity_min'] = np.min(affinities['affinity'])
        return {"ligands": ligands,
                "proteins": proteins,
                "affinities": affinities}, info



class DTADatasetBase(BindingDatasetMixin, DTAProcessMixin):
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(datasets={"proteins": proteins,
                                   "ligands": ligands,
                                   "affinities": affinities},
                         foreign_key_map={"proteins":"PID", "ligands":"LID"},
                         relation_dataset_name='affinities',
                         input_columns=input_columns)
        self.collate_fn = default_collate
        self.dataloader_cls = DataLoader


    def index_select(self, indices):
        if indices is None:
            return self
        data = getattr(self, self.relation_dataset_name)
        new_data = self.select_data(data, indices)
        # datasets = {key:getattr(self, key) for key in self._data_keys}
        # datasets[self.relation_dataset_name] = new_data
        # ans =  self.__class__(**datasets, input_columns=self.input_columns)
        ans = copy.copy(self)
        setattr(ans, self.relation_dataset_name, new_data)
        return ans


    @classmethod
    def preprocess_proteins(cls, proteins, config, seq_col='Protein_Sequence'):
        if cls.protein_featurizer_cls is not None:
            protein_featurizer = cls.protein_featurizer_cls.build_featurizer(**config)
            protein_featurizer.sanitize(proteins[0][seq_col])
            proteins = proteins.map(cls.datasets_map_fn, batched=True, fn_kwargs={"col":seq_col,
                                                                                  "featurizer": protein_featurizer},
                                    desc='tokenize proteins')
            return proteins, protein_featurizer.get_feat_info()
        return super(DTADatasetBase, cls).preprocess_proteins(proteins, config)


    @classmethod
    def preprocess_ligands(cls, ligands, config, seq_col='ISO_SMILES'):
        if cls.ligand_featurizer_cls is not None:
            ligand_featurizer = cls.ligand_featurizer_cls.build_featurizer(**config)
            ligand_featurizer.sanitize(ligands[0][seq_col])
            smiles = ligands.map(cls.datasets_map_fn, batched=True, fn_kwargs={"col":seq_col,
                                                                               'featurizer':ligand_featurizer},
                                 desc='tokenize ligands', )
            return smiles, ligand_featurizer.get_feat_info()
        return super(DTADatasetBase, cls).preprocess_ligands(ligands, config)


    def __repr__(self):
        keys = self._data_keys
        ans = []
        for key in keys:
            data = getattr(self, key)
            if isinstance(data, (datasets.Dataset, pd.DataFrame)):
               ans.append(f"{key}={data.shape}")
            elif isinstance(data, Dict):
                ans.append(f"{key}=({len(next(iter(data.values())))}, {len(data)})")
        ans = ", ".join(ans)
        return f"{self.__class__.__name__}({ans})"


    @classmethod
    def load_data(cls, root_data_dir, dataset_name):
        ligands, proteins, affinities, splits = load_data(root_data_dir, dataset_name)
        data_size = len(affinities)
        return {"ligands": ligands,
                "proteins": proteins,
                "affinities": affinities,
                "cv_splits": splits,
                'data_size': data_size}


    def __getitem__(self, idx):
        ans = self.get(idx)
        ans = self.recursive_tensorize(ans)
        ans = self.to_pgy_data(ans)
        return ans


    def __getitems__(self, idx):
        ans = self.get(idx)
        if isinstance(ans, Dict):
            data = self.recursive_tensorize(ans)
            data = self.to_pgy_data(data)
            demo = next(iter(data.values()))
            ans = [{key:item[i] for key, item in data.items()} for i in range(len(demo))]
            return ans