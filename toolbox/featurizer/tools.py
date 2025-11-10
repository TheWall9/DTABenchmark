import os
import copy
import pickle
from typing import Iterable, List, Any, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops

from deepchem import feat

from toolbox.utils import auto_argparser, get_func_arguments, Hasher
from toolbox.config import FEATURIZER_INPUT_TEMP_DIR, FEATURIZER_OUTPUT_TEMP_DIR


class GraphData():
    def __init__(self, node_features=None, edge_index=None, edge_features=None, pos=None, prefix='', **kwargs):
        self.node_features = node_features
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.pos = pos
        self.attr_keys = ['node_features', 'edge_index', 'edge_features', 'pos', *kwargs.keys()]
        self.kwargs = kwargs
        self.prefix = f"{prefix}_" if prefix else ""
        for key, value in kwargs.items():
            setattr(self, key, value)

    def items(self):
        ans = {}
        for key in self.attr_keys:
            ans[key] = getattr(self, key, None)
        return ans

    def to_dict(self, prefix=''):
        ans = {}
        prefix = prefix or self.prefix
        for key in self.attr_keys:
            value = getattr(self, key, None)
            if value is None:
                continue
            if isinstance(value, GraphData):
                ans[f'{prefix}{key}'] = value.to_dict()
            else:
                ans[f'{prefix}{key}'] = value
        return ans

    def get_feat_info(self, prefix=''):
        ans = {}
        prefix = prefix or self.prefix
        for key in self.attr_keys:
            if 'edge_index' in key or 'batch' in key or 'ids' in key:
                continue
            value = getattr(self, key, None)
            if value is None:
                continue
            if isinstance(value, np.ndarray) and value.size==0:
                continue
            if isinstance(value, GraphData):
                ans.update(value.get_feat_info(prefix=f'{prefix}{key}_'))
            elif key in ['node_v', 'edge_v']:
                ans[f'{prefix}{key}_dim'] = value.shape[1]
            elif isinstance(value, (np.ndarray, pd.DataFrame)):
                ans[f'{prefix}{key}_dim'] = value.shape[-1]
        return ans

    def get_feat_keys(self, prefix=''):
        ans = []
        prefix = prefix or self.prefix
        keys = ['x', 'edge_index', 'edge_attr', 'pos']
        for key in keys:
            value = getattr(self, key, None)
            if value is not None:
                ans.append(f"{prefix}{key}")
        for key, value in self.kwargs.items():
            if isinstance(value, GraphData):
                ans.extend(value.get_feat_keys(prefix=f'{prefix}{key}_'))
            else:
                ans.append(f"{self.prefix}{key}")
        return ans


class FeatData(GraphData):
    def __init__(self, embedding=None, input_ids=None, inputs_embeds=None, prefix='', **kwargs):
        super().__init__(embedding=embedding, prefix=prefix, input_ids=input_ids, inputs_embeds=inputs_embeds, **kwargs)
        # other_kwargs = {}
        # if inputs_embeds is not None:
        #     other_kwargs['inputs_embeds'] = inputs_embeds
        #     other_kwargs['inputs_embeds_batch'] = np.zeros(len(inputs_embeds), dtype=int)
        # if len(other_kwargs)!=0:
        #     self.inputs = GraphData(**other_kwargs)
        #     self.kwargs['inputs'] = self.inputs
        #     self.attr_keys.append('inputs')




class FeatData2():
    def __init__(self, embedding=None, input_ids=None, inputs_embeds=None, graph=None, prefix='', **kwargs):
        self.embedding = embedding
        self.input_ids = input_ids
        self.inputs_embeds = inputs_embeds
        self.graph = graph
        self.prefix = f"{prefix}_" if prefix else ""
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self, prefix=''):
        ans = {}
        prefix = prefix or self.prefix
        if self.embedding is not None:
            ans[f"{prefix}embedding"] = self.embedding
        if self.input_ids is not None:
            ans[f"{prefix}input_ids"] = self.input_ids
        if self.inputs_embeds is not None:
            ans[f"{prefix}inputs_embeds"] = [item for item in self.inputs_embeds]
        for key, value in self.kwargs.items():
            if isinstance(value, feat.GraphData):
                ans[f"{prefix}{key}"] = self.convert_graph_to_dict(value)
        if self.graph is not None:
            ans[f'{prefix}graph'] = self.convert_graph_to_dict(self.graph)
        return ans

    def convert_graph_to_dict(self, graph):
        tmp = {}
        if graph.node_features is not None and len(graph.node_features) != 0:
            tmp['node_features'] = graph.node_features
        if graph.edge_features is not None and len(graph.edge_features) != 0:
            tmp['edge_features'] = graph.edge_features
        if graph.edge_index is not None and len(graph.edge_index) != 0:
            tmp['edge_index'] = graph.edge_index
        if graph.node_pos_features is not None and len(graph.node_pos_features) != 0:
            tmp['node_pos_features'] = graph.node_pos_features
        for key in graph.kwargs:
            tmp[key] = graph.kwargs[key]
        return tmp

    def get_feat_info(self):
        ans = {}
        prefix = self.prefix
        if self.embedding is not None:
            ans[f"{prefix}embedding_dim"] = self.embedding.shape[-1]
        if self.inputs_embeds is not None:
            ans[f"{prefix}inputs_embeds_dim"] = self.inputs_embeds.shape[-1]
        if self.graph is not None:
            if self.graph.node_features is not None:
                ans[f'{prefix}node_features_dim'] = self.graph.node_features.shape[-1]
            if self.graph.edge_features is not None:
                ans[f'{prefix}edge_features_dim'] = self.graph.edge_features.shape[-1]
            if self.graph.node_pos_features is not None:
                ans[f'{prefix}node_pos_features_dim'] = self.graph.node_pos_features.shape[-1]
            if hasattr(self.graph, 'node_s'):
                ans[f'{prefix}node_s_dim'] = self.graph.node_s.shape[-1]
            if hasattr(self.graph, 'edge_s'):
                ans[f'{prefix}edge_s_dim'] = self.graph.edge_s.shape[-1]
            if hasattr(self.graph, 'node_v'):
                ans[f'{prefix}node_v_dim'] = self.graph.node_v.shape[1]
            if hasattr(self.graph, 'edge_v'):
                ans[f'{prefix}edge_v_dim'] = self.graph.edge_v.shape[1]
            for key2, value2 in self.graph.kwargs.items():
                if "feat" in key2:
                    ans[f'{prefix}{key2}_dim'] = value2.shape[-1]
        for key, value in self.kwargs.items():
            if isinstance(value, feat.GraphData):
                if value.node_features is not None:
                    ans[f'{prefix}{key}_node_features_dim'] = value.node_features.shape[-1]
                if value.edge_features is not None:
                    ans[f'{prefix}{key}_edge_features_dim'] = value.edge_features.shape[-1]
                if value.node_pos_features is not None:
                    ans[f'{prefix}{key}_node_pos_features_dim'] = value.node_pos_features.shape[-1]
                if hasattr(value, 'node_s'):
                    ans[f'{prefix}{key}_node_s_dim'] = value.node_s.shape[-1]
                if hasattr(value, 'edge_s'):
                    ans[f'{prefix}{key}_edge_s_dim'] = value.edge_s.shape[-1]
                if hasattr(value, 'node_v'):
                    ans[f'{prefix}{key}_node_v_dim'] = value.node_v.shape[1]
                if hasattr(value, 'edge_v'):
                    ans[f'{prefix}{key}_edge_v_dim'] = value.edge_v.shape[1]
                for key2, value2 in value.kwargs.items():
                    if "feat" in key2:
                        ans[f'{prefix}{key}_{key2}_dim'] = value2.shape[-1]

        return ans

    def get_feat_keys(self):
        ans = []
        if self.embedding is not None:
            ans.append(f"{self.prefix}embedding")
        if self.inputs_embeds is not None:
            ans.append(f"{self.prefix}inputs_embeds")
        if self.inputs_embeds is not None:
            ans.append(f"{self.prefix}input_ids")
        if self.graph is not None:
            ans.append(f"{self.prefix}graph")
        for key in self.kwargs:
            ans.append(f"{self.prefix}{key}")
        return ans



class ToDatasetsMixin:
    def __init__(self):
        self.example_key = None
        self._enable_cache_status = False
        self._overwrite_cache = False
        self._cache_save_dir = None
        self.tokenizer_dir = os.path.join(os.path.dirname(__file__), 'tokenizer')

    def to_datasets_example(self, datapoints, feat_name=None):
        item = next(iter(datapoints))
        if isinstance(item, FeatData):
            items = [datapoint.to_dict(prefix=feat_name) for datapoint in datapoints]
            ans = {key:[datapoint[key] for datapoint in items] for key in items[0].keys()}
            return ans
        elif isinstance(item, Dict):
            ans = {}
            for k, v in item.items():
                data = [datapoint[k] for datapoint in datapoints]
                ans[k] = data
            return ans
        return {feat_name: datapoints}

    def datasets_map_fn(self, examples, col=None, feat_name=None):
        key = col or self.example_key
        feats = self.featurize(examples[key])
        ans = self.to_datasets_example(feats, feat_name)
        return ans

    def build_featurizer(self, **kwargs):
        params = get_func_arguments(self.__init__)
        kwargs = {key:kwargs[key] for key in params if key in kwargs}
        return self.__class__(**kwargs)

    @classmethod
    def add_argparser_args(cls, parser):
        auto_argparser(cls.__init__, parser)

    def enable_cache(self, cache_dir=None, overwrite=False):
        self._enable_cache_status = True
        self._cache_save_dir = cache_dir or self._cache_save_dir
        self._overwrite_cache = overwrite
        assert self._cache_save_dir is not None
        os.makedirs(self._cache_save_dir, exist_ok=True)


class FeaturizerWrapper(ToDatasetsMixin):
    def __init__(self, feat_name=None, featurizer_cls=None, example_key=None):
        super().__init__()
        self.feat_name = feat_name or (featurizer_cls.__name__ if featurizer_cls is not None else self.__class__.__name__)
        self.featurizer_cls = featurizer_cls
        self.example_key = example_key

    def __call__(self, *args, **kwargs):
        instance_kwargs = get_func_arguments(self.featurizer_cls.__init__, *args, **kwargs)
        featurizer = self.featurizer_cls(*args, **kwargs)
        featurizer.example_key = self.example_key
        featurizer.feat_name = self.feat_name
        featurizer.to_datasets_example = self.to_datasets_example
        featurizer.datasets_map_fn = self.datasets_map_fn
        featurizer.add_argparser_args = self.add_argparser_args
        featurizer.build_featurizer = self.build_featurizer
        for key, value in instance_kwargs.items():
            if not hasattr(featurizer, key):
                setattr(featurizer, key, value)
        return featurizer


class FeaturizerBase(feat.Featurizer, ToDatasetsMixin):
    FEATURIZER_OUTPUT_TEMP_DIR = FEATURIZER_OUTPUT_TEMP_DIR
    FEATURIZER_INPUT_TEMP_DIR = FEATURIZER_INPUT_TEMP_DIR
    def __init__(self, feat_name=None, example_key=None, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.feat_name = feat_name or self.__class__.__name__
        self.example_key = example_key
        self.feat_info = {}
        self.model = None

    @property
    def cached_root_input_dir(self):
        return os.path.join(self.FEATURIZER_INPUT_TEMP_DIR, self.feat_name)

    @property
    def cached_root_output_dir(self):
        return os.path.join(self.FEATURIZER_OUTPUT_TEMP_DIR, self.feat_name)

    def featurize(self,
                  datapoints: Iterable[Any],
                  verbose: bool = False,
                  **kwargs) -> np.ndarray:
        datapoints = datapoints if isinstance(datapoints, (List, np.ndarray)) else [datapoints]
        features = []
        if verbose:
            datapoints = tqdm(datapoints, desc=f"{self.__class__.__name__}")
        for i, point in enumerate(datapoints):
            features.append(self._featurize_(str(point), **kwargs))
        try:
            ans = np.asarray(features)
        except:
            ans = np.asarray(features, dtype=object)
        return ans

    def __call__(self, datapoints: Iterable[Any], **kwargs):
        return self.featurize(datapoints, **kwargs)

    @classmethod
    def add_argparser_args(cls, parser):
        auto_argparser(cls.__init__, parser)

    @classmethod
    def build_featurizer(cls, **kwargs):
        params = get_func_arguments(cls.__init__)
        kwargs = {key:kwargs[key] for key in params if key in kwargs}
        return cls(**kwargs)

    def get_feat_info(self, data=None):
        if data is None:
            return self.feat_info
        else:
            return data.get_feat_info()

    def sanitize(self, item):
        ans = self.featurize([item])[0]
        self.feat_info = self.get_feat_info(ans)
        self.model = None
        return ans

    def _featurize_(self, datapoint, **kwargs):
        if self._enable_cache_status:
            hash = Hasher.hash(tuple([datapoint, *kwargs.items()]))
            save_file = os.path.join(self._cache_save_dir, f'{hash}.pickle')
            if not os.path.exists(save_file) or self._overwrite_cache:
                ans = self._featurize(datapoint, **kwargs)
                with open(save_file, 'wb') as f:
                    pickle.dump(ans, f)
            else:
                with open(save_file, 'rb') as f:
                    ans = pickle.load(f)
            return ans
        else:
            return self._featurize(datapoint, **kwargs)



class SmilesFeaturizerBase(FeaturizerBase):
    def __init__(self, use_original_atoms_order=False):
        super(SmilesFeaturizerBase, self).__init__()
        self.use_original_atoms_order = use_original_atoms_order

    def featurize(self, datapoints, verbose=False, **kwargs) -> np.ndarray:
        datapoints = datapoints if isinstance(datapoints, Iterable) else [datapoints]
        if verbose:
            datapoints = tqdm(datapoints, desc=f"{self.__class__.__name__}")
        features = []
        for i, seq in enumerate(datapoints):
            if isinstance(seq, str):
                if self.use_original_atoms_order:
                    mol = Chem.MolFromSmiles(seq)
                else:
                    mol = Chem.MolFromSmiles(seq)
                    new_order = rdmolfiles.CanonicalRankAtoms(mol)
                    mol = rdmolops.RenumberAtoms(mol, new_order)
            elif isinstance(seq, Chem.rdchem.Mol):
                mol = seq
            else:
                raise NotImplementedError
            kwargs_per_datapoint = {}
            for key in kwargs.keys():
                kwargs_per_datapoint[key] = kwargs[key][i]
            features.append(self._featurize_(mol, **kwargs_per_datapoint))
        return np.asarray(features, dtype=object)

class ConcatFeaturizer():
    def __init__(self, featurizers):
        self.featurizers = featurizers

    def featurize(self, *args, **kwargs):
        return [featurizer.featurize(*args, **kwargs) for featurizer in self.featurizers]

    def __call__(self, *args, **kwargs):
        return self.featurize(*args, **kwargs)

    def to_datasets_example(self, datapoints):
        return {featurizer.feature_name:value for featurizer, value in zip(self.featurizers, datapoints)}

    def datasets_map_fn(self, examples, col=None, feat_name=None):
        keys = col if isinstance(col, (List, np.ndarray)) else [col]*len(self.featurizers)
        feat_names = feat_name if isinstance(feat_name, (List, np.ndarray)) else [feat_name]*len(self.featurizers)
        ans = {}
        for i, featurizer in enumerate(self.featurizers):
            item = featurizer.datasets_map_fn(examples, col=keys[i], feat_name=feat_names[i])
            ans.update(item)
        return ans

    def get_feat_info(self):
        ans = {}
        for featurizer in self.featurizers:
            ans.update(featurizer.get_feat_info())
        return ans


