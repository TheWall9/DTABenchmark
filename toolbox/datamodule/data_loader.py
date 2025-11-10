import os
import json
import pickle
import copy
import numpy as np
import pandas as pd
from collections import OrderedDict
import datasets

from toolbox.config import DATASET_TEMP_DIR

def load_davis_data(data_dir, log_affinity=True, threshold=7.0):
    "https://github.com/hkmztrk/DeepDTA/tree/master/data/davis"
    ligands, proteins, affinity, splits = load_kiba_data(data_dir, log_affinity=log_affinity, threshold=threshold)
    ligands.rename(columns={"CHEMBL_ID": "PubChem_CID", "Uniprot_ID": "Gene_Name"}, inplace=True)
    return ligands, proteins, affinity, splits

def load_davis_refine_data(data_dir, log_affinity=True, threshold=7.0):
    ligands, proteins, affinity, splits = load_davis_data(data_dir, log_affinity=log_affinity, threshold=threshold)
    return ligands, proteins, affinity, splits


def load_kiba_data(data_dir, log_affinity=False, threshold=12.1):
    "https://github.com/hkmztrk/DeepDTA/tree/master/data/kiba"
    can_ligands_file = os.path.join(data_dir, f"ligands_can.txt")
    iso_ligands_file = os.path.join(data_dir, f"ligands_iso.txt")
    proteins_file = os.path.join(data_dir, 'proteins.txt')
    affinities_file = os.path.join(data_dir, 'Y')
    with open(can_ligands_file) as f:
        ligands = json.load(f, object_pairs_hook=OrderedDict)
    with open(iso_ligands_file) as f:
        iso_ligands = json.load(f, object_pairs_hook=OrderedDict)
    with open(proteins_file) as f:
        proteins = json.load(f, object_pairs_hook=OrderedDict)
    with open(affinities_file, 'rb') as f:
        affinity = pickle.load(f, encoding='latin1')

    train_fold_file = os.path.join(data_dir, 'folds', 'train_fold_setting1.txt')
    test_fold_file = os.path.join(data_dir, 'folds', 'test_fold_setting1.txt')
    with open(train_fold_file) as f:
        folds = json.load(f)
    with open(test_fold_file) as f:
        test_fold = json.load(f)
    valid_row, valid_col = np.where(~np.isnan(affinity))
    # row:drug  col:target
    ligands = pd.DataFrame(ligands.items(), columns=['CHEMBL_ID', 'SMILES'])
    ligands.index.name = 'LID'
    ligands.reset_index(inplace=True)
    iso_ligands = pd.DataFrame(iso_ligands.items(), columns=['CHEMBL_ID', 'ISO_SMILES'])
    ligands = pd.merge(ligands, iso_ligands, on='CHEMBL_ID')
    ligands.set_index('LID', inplace=True)
    proteins = pd.DataFrame(proteins.items(), columns=['Uniprot_ID', 'Protein_Sequence'])
    proteins.index.name = 'PID'
    proteins.reset_index(inplace=True)
    proteins.set_index('PID', inplace=True)
    affinities = pd.DataFrame({"LID": valid_row, 'PID': valid_col, 'affinity': affinity[valid_row, valid_col]})
    if log_affinity:
        affinities['affinity'] = -np.log10(affinities['affinity'] / 1e9)
    affinities['Label'] = (affinities['affinity'] > threshold).astype(int)

    splits = {"train_folds": folds, "test_fold": test_fold}
    return ligands, proteins, affinities, splits

def load_kiba_pocketdta_data(data_dir, log_affinity=False, threshold=12.1):
    data = pd.read_csv(os.path.join(data_dir, 'process.csv'), index_col=0)
    data.reset_index(inplace=True)

    splits = {"train_folds":[data[data['val']==1].index.to_list()],
              "test_fold":data[data['test']==1].index.to_list()}

    data = data.rename(columns={"Target":"Protein_Sequence", "target_key":"Uniprot_ID", "Y":"affinity"})
    proteins = data[['PID', 'Protein_Sequence', 'Uniprot_ID']].drop_duplicates().copy()
    ligands = data[['LID', 'ISO_SMILES']].drop_duplicates().copy()
    ligands.set_index('LID', inplace=True)
    ligands.reset_index(inplace=True)
    proteins.set_index('PID', inplace=True)
    proteins.reset_index(inplace=True)
    affinities = data[['PID', 'LID', 'affinity']].copy()
    if log_affinity:
        affinities['affinity'] = -np.log10(affinities['affinity'] / 1e9)
    affinities['Label'] = (affinities['affinity'] > threshold).astype(int)
    return ligands, proteins, affinities, splits

def load_davis_pocketdta_data(data_dir, log_affinity=True, threshold=7.0):
    ligands, proteins, affinity, splits = load_kiba_pocketdta_data(data_dir, log_affinity=log_affinity, threshold=threshold)
    return ligands, proteins, affinity, splits

def load_data(root_data_dir, dataset_name, overwrite=False):
    data_dir = os.path.join(root_data_dir, dataset_name)
    cached_data_dir = os.path.join(DATASET_TEMP_DIR, dataset_name)
    ligands_file = os.path.join(cached_data_dir, 'liqands')
    proteins_file = os.path.join(cached_data_dir, 'proteins')
    affinities_file = os.path.join(cached_data_dir, 'affinities')
    splits_file = os.path.join(cached_data_dir, 'splits.json')
    if not (os.path.exists(affinities_file) and os.path.exists(ligands_file) and os.path.exists(proteins_file)) or overwrite:
        if dataset_name == 'davis':
            ligands, proteins, affinities, splits = load_davis_data(data_dir)
        elif dataset_name == 'davis2':
            ligands, proteins, affinities, splits = load_davis_data(data_dir.replace("davis2", 'davis'), log_affinity=False)
        elif dataset_name == 'kiba':
            ligands, proteins, affinities, splits = load_kiba_data(data_dir)
        elif dataset_name == 'davis_refine':
            ligands, proteins, affinities, splits = load_davis_refine_data(data_dir)
        elif dataset_name in ['davis_domain', 'davis_domain2', 'davis_domain_ncbi']:
            ligands, proteins, affinities, splits = load_davis_refine_data(data_dir)
        elif dataset_name == 'kiba_domain2':
            ligands, proteins, affinities, splits = load_kiba_data(data_dir)
        elif dataset_name == 'kiba_pocketdta':
            ligands, proteins, affinities, splits = load_kiba_pocketdta_data(data_dir)
        elif dataset_name == 'davis_pocketdta':
            ligands, proteins, affinities, splits = load_davis_pocketdta_data(data_dir, log_affinity=False)
        else:
            raise NotImplementedError
        if not os.path.exists(cached_data_dir):
            os.makedirs(cached_data_dir)
        if not os.path.exists(ligands_file) or overwrite:
            datasets.Dataset.from_pandas(ligands).with_format('np').save_to_disk(ligands_file)
        if not os.path.exists(proteins_file) or overwrite:
            datasets.Dataset.from_pandas(proteins).with_format('np').save_to_disk(proteins_file)
        if not os.path.exists(affinities_file) or overwrite:
            datasets.Dataset.from_pandas(affinities).with_format('np').save_to_disk(affinities_file)
        if not os.path.exists(splits_file) or overwrite:
            with open(splits_file, 'w') as f:
                json.dump(splits, f)
    ligands = datasets.load_from_disk(ligands_file)
    proteins = datasets.load_from_disk(proteins_file)
    affinities = datasets.load_from_disk(affinities_file)
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    return ligands, proteins, affinities, splits
