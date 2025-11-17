import os
import json

import inspect
import copy
import numpy as np
from tqdm import tqdm
from lightning import LightningDataModule
from sklearn.model_selection import KFold
import datasets

from toolbox.utils import load_class
from toolbox.config import DATASET_TEMP_DIR, ROOT_DATA_DIR

from toolbox.datamodule.dataset import DTADatasetBase


class CrossValidationMixin():
    def __init__(self):
        self.cv_train_val_test_index = [None, None, None]
        self.cv_splits_file = None
        self.cv_fold_id = None

    def generate_cv_splits(self, dataset, split_type='predetermined', cv_n_splits=5, seed=666,
                           splits=None, overwrite=False, cached_data_dir=None):
        assert split_type in ['cv', 'cv_no_test', 'predetermined']
        hasher = datasets.fingerprint.Hasher()
        fingerprint = hasher.hash((dataset, split_type, cv_n_splits, seed))
        if cached_data_dir is not None:
            cached_dir = os.path.join(cached_data_dir, 'splits', split_type, f"{cv_n_splits}_{seed}")
            splits_file = os.path.join(cached_dir, f"{fingerprint}.json")
            self.cv_splits_file = splits_file
            if not os.path.exists(cached_dir):
                os.makedirs(cached_dir)
            if os.path.exists(splits_file) and not overwrite:
                with open(splits_file, 'r') as f:
                    splits = json.load(f)
            else:
                if split_type in ['cv', 'cv_no_test']:
                    kfold = KFold(n_splits=cv_n_splits+1 if split_type=='cv' else cv_n_splits, shuffle=True, random_state=seed)
                    folds = []
                    for train_idx, valid_idx in kfold.split(dataset):
                        folds.append(valid_idx.tolist())
                    splits = {"train_folds": folds[1:] if split_type=='cv' else folds,
                              "test_fold": folds[0] if split_type=='cv' else []}
                elif split_type=='predetermined':
                    pass
                else:
                    raise NotImplementedError
                if os.path.exists(cached_dir):
                    with open(splits_file, 'w') as f:
                        json.dump(splits, f)
        return splits

    def generate_cold_splits(self, pairs, split_type='cold_drug', cv_n_splits=5, seed=666, splits=None, overwrite=False, cached_data_dir=None):
        hasher = datasets.fingerprint.Hasher()
        fingerprint = hasher.hash((pairs, split_type, cv_n_splits, seed))
        if cached_data_dir is not None:
            cached_dir = os.path.join(cached_data_dir, 'splits', split_type, f"{cv_n_splits}_{seed}")
            splits_file = os.path.join(cached_dir, f"{fingerprint}.json")
            self.cv_splits_file = splits_file
            if not os.path.exists(cached_dir):
                os.makedirs(cached_dir)
            if os.path.exists(splits_file) and not overwrite:
                with open(splits_file, 'r') as f:
                    splits = json.load(f)
            else:
                if split_type in ['cold_drug', 'cold_target']:
                    folds = []
                    offset = 0 if split_type=='cold_drug' else 1
                    items = np.unique(pairs[:, offset])
                    kfold = KFold(n_splits=cv_n_splits+1, shuffle=True, random_state=seed)
                    for train_idx, valid_idx in kfold.split(items):
                        valid_id = items[valid_idx]
                        valid_fold = np.where(np.isin(pairs[:, offset], valid_id))[0].tolist()
                        folds.append(valid_fold)
                    splits = {"train_folds": folds[1:],
                              "test_fold": folds[-1]}
                elif split_type=='all_cold':
                    drugs = np.unique(pairs[:, 0])
                    targets = np.unique(pairs[:, 1])
                    kfold = KFold(n_splits=cv_n_splits+1, shuffle=True, random_state=seed)
                    train_folds, val_folds, test_folds = [], [], []
                    for drug_idx, target_idx in zip(kfold.split(drugs), kfold.split(targets)):
                        test_drug = drugs[drug_idx[1]]
                        test_target = targets[target_idx[1]]
                        test_mask = np.isin(pairs[:, 0], test_drug) & np.isin(pairs[:, 1], test_target)
                        val_mask = ~test_mask & (np.isin(pairs[:, 0], test_drug) | np.isin(pairs[:, 1], test_target))
                        train_mask = ~test_mask & ~val_mask
                        train_folds.append(np.where(train_mask)[0].tolist())
                        val_folds.append(np.where(val_mask)[0].tolist())
                        test_folds.append(np.where(test_mask)[0].tolist())
                    splits = {"train_folds":train_folds,
                              "val_folds": [None for _ in val_folds],
                              "test_folds":test_folds}
                else:
                    raise NotImplementedError
                if os.path.exists(cached_dir):
                    with open(splits_file, 'w') as f:
                        json.dump(splits, f)
        return splits


    def split_datasets(self, cv_splits=None, deepcopy=False, merge_train_val=False):
        if cv_splits is None:
            with open(self.cv_splits_file, 'r') as f:
                splits = json.load(f)
        data_size = max(map(lambda x:max(x), splits['train_folds']))
        if "test_folds" in splits:
            data_size = max(data_size, max(map(lambda x:max(x) if x is not None else 0, splits['test_folds'])))
            data_size = max(data_size, max(map(lambda x: max(x) if x is not None else 0, splits['val_folds'])))
        else:
            data_size = max(data_size, max(splits['test_fold'])) if len(splits['test_fold']) > 0 else data_size

        mask = np.ones(data_size+1, dtype=bool)
        folds = []
        if "test_folds" in splits:
            for train_fold, val_fold, test_fold in zip(splits['train_folds'], splits['val_folds'], splits['test_folds']):
                val_fold = val_fold if val_fold is not None else []
                test_fold = test_fold if test_fold is not None else []
                if merge_train_val:
                    train_fold.extend(val_fold)
                    folds.append((train_fold, test_fold, test_fold))
                else:
                    folds.append((train_fold, val_fold, test_fold))
        else:
            if merge_train_val:
                train_mask = mask.copy()
                train_mask[splits['test_fold']] = False
                folds.append([np.where(train_mask)[0].tolist(), splits['test_fold'], splits['test_fold']])
                for i, fold in enumerate(splits['train_folds']):
                    train_mask = mask.copy()
                    train_mask[fold] = False
                    folds.append([np.where(train_mask)[0].tolist(), fold, fold])
            else:
                mask[splits['test_fold']] = False
                for i, fold in enumerate(splits['train_folds']):
                    train_mask = mask.copy()
                    train_mask[fold] = False
                    folds.append([np.where(train_mask)[0].tolist(), fold, splits['test_fold']])

        for i, fold in enumerate(folds):
            dataset = copy.deepcopy(self) if deepcopy else copy.copy(self)
            dataset.cv_train_val_test_index = fold
            dataset.cv_fold_id = i
            yield dataset



class DataModule(LightningDataModule, CrossValidationMixin):
    def __init__(self, config, dataset_cls=None):
        super().__init__()
        self.config = config
        self.root_data_dir = config['root_data_dir']
        self.dataset_name = config['dataset_name']
        self.cached_data_dir = os.path.join(config.get('cached_data_dir', DATASET_TEMP_DIR), config['dataset_name'])
        self.preprocessed_data = None
        self.dataset_info = None
        self.model_cls = self.__load_model_cls(config)
        self.dataset_cls = dataset_cls or (DTADatasetBase if self.model_cls is None else self.model_cls.dataset_cls)
        self.input_columns = inspect.getfullargspec(self.model_cls.step).args[1:] if self.model_cls is not None else None
        self.prepare_data()

    def __load_model_cls(self, config):
        model_name = config.get('model_name')
        if model_name is not None:
            try:
                model_cls = load_class("benchmark", model_name)
            except:
                model_cls = load_class("models", model_name)
            return model_cls


    @classmethod
    def add_parser_arguments(cls, parser):
        parser.add_argument('--root_data_dir', type=str, default=ROOT_DATA_DIR)
        parser.add_argument('--dataset_name', type=str, default='kiba')
        parser.add_argument('--cached_data_dir', type=str, default=DATASET_TEMP_DIR)
        parser.add_argument('--cv_split_type', type=str, default='predetermined', choices=['predetermined', 'cv_no_test', 'cv'])
        parser.add_argument('--cv_n_splits', type=int, default=5)
        parser.add_argument('--cv_split_seed', type=int, default=666)
        parser.add_argument("--merge_train_val", action="store_true")
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--num_workers', type=int, default=0)
        parser.add_argument('--prefetch_factor', type=int, default=None)
        parser.add_argument('--overwrite', action='store_true')

    @property
    def dataset_full_name(self):
        return os.path.join(self.dataset_name, f"{self.config['cv_split_type']}",
                            f"{self.config['cv_n_splits']}_{self.config['cv_split_seed']}_{self.cv_fold_id}")

    def generate_splits(self, dataset, split_type='predetermined', cv_n_splits=5, seed=666,
                           splits=None, overwrite=False, cached_data_dir=None):
        if split_type in ['cv', 'cv_no_test', 'predetermined']:
            return self.generate_cv_splits(np.arange(len(dataset)), split_type, cv_n_splits, seed, splits, overwrite, cached_data_dir)
        elif split_type in ['cold_drug', 'cold_target', 'all_cold']:
            return self.generate_cold_splits(dataset[['LID', 'PID']].values, split_type, cv_n_splits, seed, splits, overwrite, cached_data_dir)
        else:
            raise NotImplementedError

    def prepare_data(self):
        split_type = self.config['cv_split_type']
        cv_n_splits = self.config['cv_n_splits']
        seed = self.config['cv_split_seed']
        overwrite = self.config['overwrite']
        if self.preprocessed_data is None:
            raw_data = DTADatasetBase.load_data(self.root_data_dir, dataset_name=self.dataset_name)
            splits = self.generate_splits(raw_data['affinities'].to_pandas(), split_type=split_type, cv_n_splits=cv_n_splits, seed=seed,
                                             splits=raw_data.get("cv_splits"), overwrite=overwrite, cached_data_dir=self.cached_data_dir)
            preprocessed_data, self.dataset_info = self.dataset_cls.preprocess(config=self.config, **raw_data)
            self.preprocessed_data = self.dataset_cls(input_columns=self.input_columns, **preprocessed_data)


    @property
    def data_info(self):
        basic = {"cv_fold_id": self.cv_fold_id,
                "dataset_full_name": self.dataset_full_name,
                }
        basic.update(self.dataset_info)
        return basic

    def setup(self, stage=None):
        self.prepare_data()


    def train_dataloader(self):
        dataset = self.preprocessed_data.select(self.cv_train_val_test_index[0])
        collate_fn = dataset.collate_fn
        dataloader_cls = dataset.dataloader_cls
        if len(dataset)!=0:
            dataloader = dataloader_cls(dataset, batch_size=self.config['batch_size'], collate_fn=collate_fn, num_workers=self.config['num_workers'],
                                        prefetch_factor=self.config['prefetch_factor'], shuffle=True)
            return dataloader
        else:
            return []

    def val_dataloader(self):
        dataset = self.preprocessed_data.select(self.cv_train_val_test_index[1])
        collate_fn = dataset.collate_fn
        dataloader_cls = dataset.dataloader_cls
        if len(dataset)!=0:
            dataloader = dataloader_cls(dataset, batch_size=self.config['batch_size'], shuffle=False, collate_fn=collate_fn)
            return dataloader
        else:
            return []

    def test_dataloader(self):
        dataset = self.preprocessed_data.select(self.cv_train_val_test_index[2])
        collate_fn = dataset.collate_fn
        dataloader_cls = dataset.dataloader_cls
        if len(dataset)!=0:
            dataloader = dataloader_cls(dataset, batch_size=self.config['batch_size'], shuffle=False, collate_fn=collate_fn)
            return dataloader
        else:
            return []

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join([f'{key}={value}' for key, value in self.config.items()])})"

    def __len__(self):
        return len(self.preprocessed_data)

    def select(self, index):
        dataset = copy.copy(self)
        dataset.preprocessed_data = dataset.preprocessed_data.select(index)
        return dataset



if __name__ == "__main__":

    import time
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    DataModule.add_parser_arguments(parser)
    args = parser.parse_args()
    start_time = time.time()
    # args.dataset_name = 'davis_refine'
    datamodule = DataModule(vars(args))
    datamodule.prepare_data()
    datamodule.setup()
    # datamodule.preprocessed_data.to_pandas()

    for batch in tqdm(datamodule.train_dataloader()):
        pass

    for dataset in datamodule.split_datasets():
        for batch in tqdm(dataset.train_dataloader()):
            pass





