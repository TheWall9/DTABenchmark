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
                        valid_fold = np.where(pairs[:, offset].isin(valid_id))[0].tolist()
                        folds.append(valid_fold)
                    splits = {"train_folds": folds[1:],
                              "test_fold": folds[-1]}
                elif split_type=='all_cold':
                    drugs = np.unique(pairs[:, 0])
                    targets = np.unique(pairs[:, 1])
                    np.random.seed(seed)
                    valid_drugs = np.random.choice(drugs, size=drugs//cv_n_splits, replace=False)
                    valid_targets = np.random.choice(targets, size=targets//cv_n_splits, replace=False)
                    mask = pairs[:, 0].isin(valid_drugs) & pairs[:, 1].isin(valid_targets)
                    valid_fold = np.where(mask)[0].tolist()
                    train_fold = np.where(~mask)[0].tolist()
                    splits = {"train_folds":[train_fold],
                              "test_fold":valid_fold}
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
        data_size = max(data_size, max(splits['test_fold'])) if len(splits['test_fold']) > 0 else data_size
        mask = np.ones(data_size+1, dtype=bool)
        if merge_train_val:
            train_mask = mask.copy()
            train_mask[splits['test_fold']] = False
            dataset = copy.deepcopy(self) if deepcopy else copy.copy(self)
            dataset.cv_train_val_test_index = [np.where(train_mask)[0].tolist(), splits['test_fold'], splits['test_fold']]
            dataset.cv_fold_id = 0
            yield dataset
            for i, fold in enumerate(splits['train_folds']):
                train_mask = mask.copy()
                train_mask[fold] = False
                dataset = copy.deepcopy(self) if deepcopy else copy.copy(self)
                dataset.cv_train_val_test_index = [np.where(train_mask)[0].tolist(), fold, fold]
                dataset.cv_fold_id = i+1
                yield dataset
        else:
            mask[splits['test_fold']] = False
            for i, fold in enumerate(splits['train_folds']):
                train_mask = mask.copy()
                train_mask[fold] = False
                dataset = copy.deepcopy(self) if deepcopy else copy.copy(self)
                dataset.cv_train_val_test_index = [np.where(train_mask)[0].tolist(), fold, splits['test_fold']]
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

    def prepare_data(self):
        split_type = self.config['cv_split_type']
        cv_n_splits = self.config['cv_n_splits']
        seed = self.config['cv_split_seed']
        overwrite = self.config['overwrite']
        if self.preprocessed_data is None:
            raw_data = DTADatasetBase.load_data(self.root_data_dir, dataset_name=self.dataset_name)
            splits = self.generate_cv_splits(np.arange(raw_data['data_size']), split_type=split_type, cv_n_splits=cv_n_splits, seed=seed,
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





