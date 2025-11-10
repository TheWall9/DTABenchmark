import os
import json
import time
from typing import Any

import numpy as np
import requests
import pandas as pd
from tqdm import tqdm
from Bio import UniProt
from collections import OrderedDict

from toolbox import FeaturizerBase
from toolbox.utils import read_mmcif_structure


def kiba_download_uniprot_and_alphafold():
    with open("../data/kiba/proteins.txt") as f:
        proteins = json.load(f, object_hook=OrderedDict)
    proteins = pd.DataFrame(proteins.items(), columns=['final_uniprot_id', 'DeepDTA_seq'])
    tmp_dir = f'download_tmp/kiba/final_uniprot'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    task = proteins['final_uniprot_id'].unique().tolist()
    for idx in tqdm(task, 'crawl uniprot'):
        save_file = os.path.join(tmp_dir, f"{idx}.json")
        if os.path.exists(save_file):
            continue
        # idx = idx.split(".")[0]
        res = UniProt.search(f"(accession:{idx})")
        ans = [entry for entry in res]
        if len(ans) > 0:
            with open(save_file, "w") as f:
                json.dump(ans, f)
        else:
            print(idx)
        time.sleep(0.5)
    for idx in tqdm(task, 'crawl alphafold2'):
        file = f'AF-{idx}-F1-model_v4.cif'
        save_file = os.path.join(tmp_dir, file)
        if os.path.exists(save_file):
            continue
        url = f"https://alphafold.ebi.ac.uk/files/{file}"
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                with open(save_file, "w") as f:
                    f.write(response.text)
            else:
                print("fail", idx)
            time.sleep(0.5)
        except Exception as e:
            print(idx, e)
    from Bio.PDB import PPBuilder
    builder = PPBuilder()
    seqs = {}
    for idx in tqdm(task):
        # if idx not in ["O14965", "Q00532", "P49761", "Q9Y4K4", "O75747", "Q9Y2K2"]:
        #     continue
        file = os.path.join(tmp_dir, f'AF-{idx}-F1-model_v4.cif')
        if idx=="P78527":
            file = os.path.join(tmp_dir, "7otw.cif")
        structure = read_mmcif_structure(file)
        pps = list(builder.build_peptides(structure))
        if len(pps)>1:
            pass
        if len(list(structure.get_chains()))>1:
            print(idx)
        pp_seq = "X".join([str(pp.get_sequence()) for pp in pps])

        uniprot_file = os.path.join(tmp_dir, f"{idx}.json")
        with open(uniprot_file) as f:
            data = json.load(f)
        if len(data)>1:
            print(uniprot_file)
        entry = data[0]
        prot_seq = entry.get('sequence', {}).get('value')

        tmp_seq = list(prot_seq)
        for pp in pps:
            start_idx = pp[0].get_id()[1]-1
            end_idx = pp[-1].get_id()[1]
            pp_tmp_seq = list(pp.get_sequence())
            if pp_tmp_seq!=tmp_seq[start_idx:end_idx]:
                print("".join(pp_tmp_seq))
                print("".join(tmp_seq[start_idx:end_idx]))
            tmp_seq[start_idx:end_idx] = pp_tmp_seq
        pp_seq2 = "".join(tmp_seq)
        pp_seq = get_sequence(structure)
        if prot_seq!=pp_seq:
            print("mismatch", file, uniprot_file, len(prot_seq), pps)

        seqs[idx] = (pp_seq, os.path.basename(file), prot_seq)

    proteins['final_alphafold_seq'] = None
    for idx, row in proteins.iterrows():
        proteins.loc[idx, 'final_uniprot_seq'] = seqs[row['final_uniprot_id']][2]
        proteins.loc[idx, 'final_alphafold_seq'] = seqs[row['final_uniprot_id']][0]
        proteins.loc[idx, 'struct_file'] = seqs[row['final_uniprot_id']][1]
        # proteins.loc[idx, 'alphafold_seq_identity'] = int(seqs[row['final_uniprot_id']][0]==row['DeepDTA_seq'])

        # proteins.loc[idx, 'uniprot_seq_identity'] = int(seqs[row['final_uniprot_id']][2]==row['DeepDTA_seq'])
    proteins.to_excel("kiba_proteins.xlsx", index=False)



def get_sequence(struct):
    from Bio import PDB
    from Bio.Data.IUPACData import protein_letters_3to1
    chains = list(struct.get_chains())
    assert len(chains)==1
    chain = chains[0]
    residues = []
    for residue in chain:
        residue_name = 'X'
        if PDB.is_aa(residue):
            residue_name = protein_letters_3to1[residue.get_resname().capitalize()]
        else:
            print(struct)
        residues.append(residue_name)
    assert len(residues)==len(chain)
    return "".join(residues)

def extract_uniprot_domain():
    proteins = pd.read_excel("kiba_proteins.xlsx")
    tmp_dir = f'download_tmp/kiba/final_uniprot'
    proteins['final_uniprot_domain_loc'] = None
    proteins['final_uniprot_domain'] = None
    for idx, row in proteins.iterrows():
        json_file = os.path.join(tmp_dir, f"{row['final_uniprot_id']}.json")
        with open(os.path.join(json_file)) as f:
            data = json.load(f)
        assert len(data)==1
        seq = data[0]['sequence']['value']
        domains = [feature for feature in data[0]['features'] if feature['type']=='Domain']
        features = []
        for domain in domains:
            if 'Protein kinase' in domain['description']:
                features.append(domain)
            elif "catalytic" in domain['description']:
                features.append(domain)
            elif len(domains)==1:
                features.append(domain)
                # print(domain)
        if len(features)>2:
            print(features)
        "O60674 O75582 "
        # try:
        assert proteins.loc[idx, 'final_uniprot_seq']==seq
        # except:
        #     print(row)
        if 0<len(features)<=2:
            feature = features[0]
            start = feature['location']['start']['value']
            end = feature['location']['end']['value']
            domain = seq[start-1:end]
            proteins.loc[idx, 'final_uniprot_domain_loc'] = f"{start}-{end}"
            proteins.loc[idx, 'final_uniprot_domain'] = domain
    with pd.ExcelWriter('kiba_proteins.xlsx', mode='a', if_sheet_exists='replace') as writer:
        proteins.to_excel(writer, sheet_name='uniprot', index=False)


def extract_deepdta_domain():
    proteins = pd.read_excel("kiba_proteins.xlsx")
    uniprot_proteins = pd.read_excel("kiba_proteins.xlsx", sheet_name='uniprot')
    proteins['DeepDTA_domain_loc'] = None
    proteins['DeepDTA_domain'] = None
    for idx, row in proteins.iterrows():
        start_idx, end_idx = None, None
        uniprot_row = uniprot_proteins.loc[idx]
        uniprot_seq = uniprot_row['final_uniprot_seq']
        seq = row['DeepDTA_seq']
        uniprot_domain = uniprot_row['final_uniprot_domain']
        if pd.isna(uniprot_domain):
            continue
        if uniprot_seq==seq:
            proteins.loc[idx, 'DeepDTA_domain_loc'] = uniprot_row['final_uniprot_domain_loc']
            proteins.loc[idx, 'DeepDTA_domain'] = uniprot_row['final_uniprot_domain']
            start_idx, end_idx = map(int, uniprot_row['final_uniprot_domain_loc'].split('-'))
            start_idx = start_idx-1
        else:
            start_idx = seq.find(uniprot_domain)
            if start_idx!=-1:
                end_idx = min(start_idx+len(uniprot_domain), len(seq))
                proteins.loc[idx, 'DeepDTA_domain_loc'] = f"{start_idx+1}-{end_idx}"
                proteins.loc[idx, 'DeepDTA_domain'] = uniprot_domain

            else:
                from Bio import pairwise2
                alignments = pairwise2.align.localms(uniprot_domain, seq, 2, -1, -3, -3)
                start_idx = alignments[0].start
                end_idx = alignments[0].end
                match_len = end_idx-start_idx
                if match_len==len(uniprot_domain):
                    proteins.loc[idx, 'DeepDTA_domain_loc'] = f"{start_idx + 1}-{end_idx}"
                    proteins.loc[idx, 'DeepDTA_domain'] = seq[start_idx:end_idx]
                # elif row['Accession Number'] in ['NP_001706.2','NP_000052.1', 'NP_003709.3', 'NP_004435.3', 'NP_002750.1', 'NP_006276.2']:
                #     proteins.loc[idx, 'DeepDTA_domain_loc'] = f"{start_idx + 1}-{end_idx}"
                #     proteins.loc[idx, 'DeepDTA_domain'] = seq[start_idx:end_idx]
                else:
                    print(row['Entrez Gene Symbol'], row['Kinase'], row['Accession Number'], uniprot_row['final_uniprot_id'])
                    print(len(alignments))
                    print(pairwise2.format_alignment(*alignments[0]))

    with pd.ExcelWriter('kiba_proteins.xlsx', mode='a', if_sheet_exists='replace') as writer:
        proteins.to_excel(writer, sheet_name='DeepDTA', index=False)


def extract_alphafold_domain():
    proteins = pd.read_excel("kiba_proteins.xlsx", sheet_name='uniprot')
    proteins['final_alphafold_domain_loc'] = None
    proteins['final_alphafold_domain'] = None
    proteins['final_alphafold_seq_truncated'] = None
    for idx, row in proteins.iterrows():
        start_idx, end_idx = None, None
        uniprot_seq = row['final_uniprot_seq']
        seq = row['final_alphafold_seq']
        uniprot_domain = row['final_uniprot_domain']
        if pd.isna(uniprot_domain):
            continue
        if uniprot_seq == seq:
            proteins.loc[idx, 'final_alphafold_domain_loc'] = row['final_uniprot_domain_loc']
            proteins.loc[idx, 'final_alphafold_domain'] = row['final_uniprot_domain']
            start_idx, end_idx = map(int, row['final_uniprot_domain_loc'].split('-'))
            start_idx = start_idx - 1
        else:
            start_idx = seq.find(uniprot_domain)
            if start_idx != -1:
                end_idx = min(start_idx + len(uniprot_domain), len(seq))
                proteins.loc[idx, 'final_alphafold_domain_loc'] = f"{start_idx + 1}-{end_idx}"
                proteins.loc[idx, 'final_alphafold_domain'] = uniprot_domain

            else:
                from Bio import pairwise2
                alignments = pairwise2.align.localms(uniprot_domain, seq, 2, -1, -3, -3)
                start_idx = alignments[0].start
                end_idx = alignments[0].end
                match_len = end_idx - start_idx
                if match_len == len(uniprot_domain):
                    proteins.loc[idx, 'final_alphafold_domain_loc'] = f"{start_idx + 1}-{end_idx}"
                    proteins.loc[idx, 'final_alphafold_domain'] = seq[start_idx:end_idx]
                else:
                    print(row['Entrez Gene Symbol'], row['Kinase'], row['Accession Number'],
                          row['final_uniprot_id'])
                    print(alignments)
    with pd.ExcelWriter('kiba_proteins.xlsx', mode='a', if_sheet_exists='replace') as writer:
        proteins.to_excel(writer, sheet_name='uniprot', index=False)



if __name__ == '__main__':
    """
    O14965
    Q9Y4K4
    
    Q04912
    Q9HBY8
    """
    # kiba_download_uniprot_and_alphafold()
    extract_uniprot_domain()
    extract_alphafold_domain()
    extract_deepdta_domain()

    from datasets.fingerprint import Hasher
    from toolbox.config import DATASET_TEMP_DIR
    from toolbox.utils import exec_command
    import subprocess


    from deepchem.feat import Featurizer




