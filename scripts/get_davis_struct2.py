import os
import re
import json
import time
from functools import partial
import requests
import numpy as np
from Bio import UniProt
import pandas as pd
from collections import OrderedDict
from Bio import pairwise2
from Bio import SeqIO
from tqdm import tqdm
import datasets


from toolbox.utils import get_uniprot_info, MultiThreadProcessor, get_ncbi_info, read_mmcif_structure

def align_domain(seq, domain):
    alignments = pairwise2.align.localms(domain, seq, 2, -1, -3, -3)
    start_idx = alignments[0].start
    end_idx = alignments[0].end
    return alignments, start_idx, end_idx

def apply_mutation(seq, mutation_str, replace_x=False, uniprot_json=None):
    """
    Apply mutations to a protein sequence.

    Args:
        seq (str): original amino acid sequence
        mutation_str (str): mutation info in parentheses, e.g. "E255K", "L747-E749del", "L747-E749del,A750P"

    Returns:
        str: mutated sequence
    """

    match = re.match(r'([A-Za-z0-9]+)\(([^)]*)\)', mutation_str)
    if not match:
        return seq, True
    raw_seq = seq
    seq = list(seq)  # 转成列表方便修改
    flag = True
    for tag in ['Kin.Dom.1', 'JH1domain']:
        if tag in mutation_str:
            with open(uniprot_json, 'r') as f:
                data = json.load(f)[0]
                for feature in data['features']:
                    if feature['description']=='Protein kinase 1':
                        start = feature['location']['start']['value']
                        end = feature['location']['end']['value']
                        domain = data['sequence']['value'][start - 1:end]
                        alignments, start_idx, end_idx = align_domain("".join(seq), domain)
                        assert len(alignments)==1
                        for i in range(start_idx):
                            seq[i] = 'X'
                        for j in range(end_idx, len(seq)):
                            seq[j] = 'X'
                        break
            break
    for tag in ['Kin.Dom.2', 'JH2domain']:
        if tag in mutation_str:
            with open(uniprot_json, 'r') as f:
                data = json.load(f)[0]
                for feature in data['features']:
                    if feature['description']=='Protein kinase 2':
                        start = feature['location']['start']['value']
                        end = feature['location']['end']['value']
                        domain = data['sequence']['value'][start - 1:end]
                        alignments, start_idx, end_idx = align_domain("".join(seq), domain)
                        assert len(alignments)==1
                        for i in range(start_idx):
                            seq[i] = 'X'
                        for j in range(end_idx, len(seq)):
                            seq[j] = 'X'
                        break
            break
    mutations = match.group(2).split(',')  # 支持多个突变
    for mut in mutations:
        mut = mut.strip()
        # 点突变 e.g., E255K
        m_point = re.match(r'([A-Z])(\d+)([A-Z])', mut)
        if m_point:
            aa_from, pos, aa_to = m_point.groups()
            pos = int(pos) - 1  # 0-based index
            if seq[pos] != aa_from:
                print(f"Warning: expected {aa_from} at position {pos + 1}, found {seq[pos]}")
            seq[pos] = aa_to
            flag = False
            continue

        # 缺失突变 e.g., L747-E749del
        m_del = re.match(r'([A-Z])(\d+)-([A-Z])(\d+)del', mut)
        if m_del:
            aa_start, pos_start, aa_end, pos_end = m_del.groups()
            pos_start = int(pos_start) - 1
            pos_end = int(pos_end)  # 切片末端本身不减 1
            # 可选验证起止残基
            if seq[pos_start] != aa_start or seq[pos_end - 1] != aa_end:
                print(f"Warning: deletion boundaries don't match: {mut}")
            for i in range(pos_start, pos_end):
                if replace_x:
                    seq[i] = 'X'
                else:
                    seq[i] = ''  # 删除
            flag = False
            continue
    return ''.join(seq), flag


def davis_info_merge():
    paper_protein_file = "davis_41587_2011_BFnbt1990_MOESM3_ESM.xls"
    raw_proteins = pd.read_excel(paper_protein_file)
    raw_proteins = raw_proteins[['Accession Number','Entrez Gene Symbol', 'Kinase', 'Mutant']].copy()
    raw_proteins['DeepDTA_gene_name'] = None
    raw_proteins['DeepDTA_seq'] = None
    protein_file = "../data/davis/proteins.txt"
    with open(protein_file) as f:
        proteins = json.load(f, object_pairs_hook=OrderedDict)
    for (idx, row), key in zip(raw_proteins.iterrows(), proteins.keys()):
        raw_proteins.loc[idx, 'DeepDTA_gene_name'] = key
        raw_proteins.loc[idx, 'DeepDTA_seq'] = proteins[key]
        assert row['Kinase'][:4]==key[:4]
    with pd.ExcelWriter('davis_proteins.xlsx') as writer:
        pd.read_excel(paper_protein_file).to_excel(writer, sheet_name='davis_paper_info', index=False)
        raw_proteins.to_excel(writer, sheet_name='DeepDTA', index=False)

def davis_download_from_uniprot(reviewed=False):
    tmp_dir = f'download_tmp/davis/uniprot{"_reviewed" if reviewed else ""}'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    data_file = pd.read_excel('davis_proteins.xlsx', sheet_name='davis_paper_info')
    ref_ids = data_file['Accession Number'].unique().tolist()
    processor = MultiThreadProcessor(max_workers=1)
    task_fn = partial(get_uniprot_info, save_dir=tmp_dir, return_all=True, reviewed=reviewed)
    processor.process_batch(ref_ids, task_fn)
    print(processor.get_failed_tasks())

def davis_download_from_ncbi():
    tmp_dir = 'download_tmp/davis/ncbi'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    data_file = pd.read_excel('davis_proteins.xlsx', sheet_name='davis_paper_info')
    ref_ids = data_file['Accession Number'].unique().tolist()
    processor = MultiThreadProcessor(max_workers=2)
    task_fn = partial(get_ncbi_info, save_dir=tmp_dir)
    processor.process_batch(ref_ids, task_fn)
    print(processor.get_failed_tasks())

def davis_ncbi_bind():
    ncbi_dir = 'download_tmp/davis/ncbi'
    proteins = pd.read_excel('davis_proteins.xlsx', sheet_name='davis_paper_info')
    proteins = proteins[['Accession Number','Entrez Gene Symbol', 'Kinase', 'Mutant']].copy()
    proteins['raw_ncbi_seq'] = None
    proteins['ncbi_to_UniprotId'] = None
    for idx, row in proteins.iterrows():
        raw_seq = SeqIO.read(os.path.join(ncbi_dir, f"{row['Accession Number']}.gb"), 'gb')
        ncbi_seq = str(raw_seq.seq)
        proteins.loc[idx, 'raw_ncbi_seq'] = ncbi_seq
        uniprot_id = []
        for feat in raw_seq.features:
            for note in feat.qualifiers.get('note', []):
                if "UniProtKB" in note:
                    ids = note.split("propagated from UniProtKB/Swiss-Prot")[-1].strip()[1:-1]
                    if len(ids)>8:
                        print(note)
                    else:
                        uniprot_id.append(ids)
        proteins.loc[idx, 'ncbi_to_UniprotId'] = ",".join(set(uniprot_id))
    with pd.ExcelWriter('davis_proteins.xlsx', mode='a', if_sheet_exists='replace') as writer:
        proteins.to_excel(writer, sheet_name='ncbi_info', index=False)


def davis_uniprot_bind():
    uniprot_dir = "download_tmp/davis/uniprot"
    proteins = pd.read_excel("davis_proteins.xlsx", sheet_name='davis_paper_info')
    ncbi_proteins = pd.read_excel("davis_proteins.xlsx", sheet_name='ncbi_info')
    proteins = proteins[['Accession Number','Entrez Gene Symbol', 'Kinase', 'Mutant']].copy()
    proteins['ncbi_to_UniprotId'] = ncbi_proteins['ncbi_to_UniprotId']
    proteins['uniprot_id'] = None
    proteins['id_identity'] = None
    proteins['uniprot_entry_name'] = None
    proteins['uniprot_gene_name'] = None
    proteins['uniprot_seq'] = None
    proteins['uniprot_organism'] = None
    proteins['uniprot_reviewed'] = None
    for idx, row in proteins.iterrows():
        file = os.path.join(uniprot_dir, f"{row['Accession Number']}.json")
        if "NP_001010938.1" in file:
            pass
        if os.path.exists(file):
            with open(file) as f:
                data = json.load(f)
            temp = [entry for entry in data if " reviewed " in entry["entryType"]]
            if len(data)==1:
                entry = data[0]
            elif len(temp)==1:
                entry = temp[0]
            else:
                temp = [entry for entry in data if row['Entrez Gene Symbol'] in entry.get("genes", [{}])[0].get("geneName", {}).get("value", "")]
                if len(temp)==1:
                    entry = temp[0]
                else:
                    entry = data[0]
                    print("mismatch", file)
                    # continue
            proteins.loc[idx, 'ncbi_to_UniprotId'] = ncbi_proteins.loc[idx, 'ncbi_to_UniprotId']
            if idx==137 or proteins.loc[idx, 'Accession Number'].strip()=="NP_004436.2":
                print('hello')
            assert proteins.loc[idx, 'Kinase'] == ncbi_proteins.loc[idx, 'Kinase']
            proteins.loc[idx, 'uniprot_id'] = entry.get("primaryAccession")
            if not pd.isna(ncbi_proteins.loc[idx, 'ncbi_to_UniprotId']):
                proteins.loc[idx, 'id_identity'] = int(ncbi_proteins.loc[idx, 'ncbi_to_UniprotId'].startswith(entry.get("primaryAccession")))
            proteins.loc[idx, 'uniprot_entry_name'] = entry.get("uniProtkbId")
            proteins.loc[idx, 'uniprot_gene_name'] = entry.get("genes", [{}])[0].get("geneName", {}).get("value")
            proteins.loc[idx, 'uniprot_seq'] = entry.get('sequence', {}).get('value')
            proteins.loc[idx, 'uniprot_organism'] = entry.get("organism", {}).get("scientificName")
            proteins.loc[idx, 'uniprot_reviewed'] = int(" reviewed " in entry['entryType'])

    with pd.ExcelWriter('davis_proteins.xlsx', mode='a', if_sheet_exists='replace') as writer:
        proteins.to_excel(writer, sheet_name='uniprot_retrieve_by_ncbi', index=False)


def davis_manual_update():
    manual_file = "manual_anno.json"
    with open(manual_file) as f:
        id_map = json.load(f)
    proteins = pd.read_excel("davis_proteins.xlsx", sheet_name='uniprot_retrieve_by_ncbi')
    proteins = proteins[['Accession Number','Entrez Gene Symbol', 'Kinase', 'Mutant', 'uniprot_id', 'ncbi_to_UniprotId']].copy()
    # ncbi_proteins = pd.read_excel("davis_proteins.xlsx", sheet_name='ncbi_info')
    tmp_dir = f'download_tmp/davis/uniprot_manual'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    for key, idx in tqdm(id_map.items(), 'crawl'):
        save_file = os.path.join(tmp_dir, f"{key}.json")
        if os.path.exists(save_file):
            continue
        idx = idx.split(".")[0]
        res = UniProt.search(f"(accession:{idx})")
        ans = [entry for entry in res]
        if len(ans)>0:
            with open(save_file, "w") as f:
                json.dump(ans, f)
        else:
            print(key, idx)
        time.sleep(0.5)
    proteins['final_uniprot_id'] = None
    for idx, row in proteins.iterrows():
        uniprot_id = str(row['uniprot_id'])
        refseq_uniprot_id = str(row['ncbi_to_UniprotId'])
        if (uniprot_id not in ['None', 'nan']) and (refseq_uniprot_id not in ['None', 'nan']):
            if refseq_uniprot_id[:len(uniprot_id)]!=uniprot_id:
                print("missmatch", refseq_uniprot_id, uniprot_id, row['Accession Number'])
        if row['Accession Number'] in id_map:
            proteins.loc[idx, 'final_uniprot_id'] = id_map[row['Accession Number']].split(".")[0]
        elif refseq_uniprot_id not in ['None', 'nan']:
            proteins.loc[idx, 'final_uniprot_id'] = refseq_uniprot_id.split(".")[0]
        elif uniprot_id not in ['None', 'nan']:
            proteins.loc[idx, 'final_uniprot_id'] = uniprot_id
        else:
            print("couldn't find", row['Accession Number'])
    with pd.ExcelWriter('davis_proteins.xlsx', mode='a', if_sheet_exists='replace') as writer:
        proteins.to_excel(writer, sheet_name='uniprot', index=False)



def davis_download_uniprot_and_alphafold():
    proteins = pd.read_excel("davis_proteins.xlsx", sheet_name='uniprot')
    tmp_dir = f'download_tmp/davis/final_uniprot'
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
        # if idx not in ['Q9Y2K2']:
        #     continue
        file = os.path.join(tmp_dir, f'AF-{idx}-F1-model_v4.cif')
        structure = read_mmcif_structure(file)
        pps = list(builder.build_peptides(structure))
        if len(list(structure.get_chains()))>1:
            print(idx)
        if len(pps)>1:
            print(pps)

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
    proteins['final_uniprot_seq'] = None
    proteins['final_alphafold_seq'] = None
    for idx, row in proteins.iterrows():
        proteins.loc[idx, 'final_uniprot_seq'] = seqs[row['final_uniprot_id']][2]
        proteins.loc[idx, 'final_alphafold_seq'] = seqs[row['final_uniprot_id']][0]
        proteins.loc[idx, 'struct_file'] = seqs[row['final_uniprot_id']][1]
    with pd.ExcelWriter('davis_proteins.xlsx', mode='a', if_sheet_exists='replace') as writer:
        proteins.to_excel(writer, sheet_name='uniprot', index=False)


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
    proteins = pd.read_excel("davis_proteins.xlsx", sheet_name='uniprot')
    tmp_dir = f'download_tmp/davis/final_uniprot'
    proteins['final_uniprot_domain_loc'] = None
    proteins['final_uniprot_domain'] = None
    proteins['final_uniprot_domain_desc'] = None
    for idx, row in proteins.iterrows():
        json_file = os.path.join(tmp_dir, f"{row['final_uniprot_id']}.json")
        with open(os.path.join(json_file)) as f:
            data = json.load(f)
        assert len(data)==1
        seq = data[0]['sequence']['value']
        domains = [feature for feature in data[0]['features'] if feature['type']=='Domain']
        features = []
        name = row['Kinase']
        for domain in domains:
            if ("Kin.Dom.1" in name) or ("JH1domain" in name):
                if domain['description'] == 'Protein kinase 1':
                    features.append(domain)
                    break
                continue
            if ("Kin.Dom.2" in name) or ("JH2domain" in name):
                if domain['description'] == 'Protein kinase 2':
                    features.append(domain)
                    break
                continue
            if 'Protein kinase' in domain['description']:
                features.append(domain)
            elif "catalytic" in domain['description']:
                features.append(domain)
            elif len(domains)==1:
                features.append(domain)
                # print(domain)
        if len(features)==2:
            print(features)
        # try:
        assert proteins.loc[idx, 'final_uniprot_seq']==seq
        # except:
        #     print(row)
        if len(features)==1:
            feature = features[0]
            start = feature['location']['start']['value']
            end = feature['location']['end']['value']
            domain = seq[start-1:end]
            proteins.loc[idx, 'final_uniprot_domain_loc'] = f"{start}-{end}"
            proteins.loc[idx, 'final_uniprot_domain'] = domain
            proteins.loc[idx, 'final_uniprot_domain_desc'] = feature['description']
    with pd.ExcelWriter('davis_proteins.xlsx', mode='a', if_sheet_exists='replace') as writer:
        proteins.to_excel(writer, sheet_name='uniprot', index=False)

def extract_ncbi_domain():
    tmp_dir = f'download_tmp/davis/final_uniprot'
    proteins = pd.read_excel("davis_proteins.xlsx", sheet_name='ncbi_info')
    uniprot_proteins = pd.read_excel("davis_proteins.xlsx", sheet_name='uniprot')
    proteins['ncbi_seq_mutant'] = None
    proteins['ncbi_seq_truncated'] = None
    proteins['ncbi_seq'] = None
    proteins['ncbi_domain_loc'] = None
    proteins['ncbi_domain'] = None
    for idx, row in proteins.iterrows():
        start_idx, end_idx = None, None
        uniprot_row = uniprot_proteins.loc[idx]
        uniprot_seq = uniprot_row['final_uniprot_seq']
        uniprot_json = os.path.join(tmp_dir, f"{uniprot_row['final_uniprot_id']}.json")
        name = row['Kinase']
        seq = row['raw_ncbi_seq']
        mutant_seq, mutant_flag = apply_mutation(seq, name, replace_x=True, uniprot_json=uniprot_json)
        if mutant_seq!=seq:
            mutant_seq = mutant_seq.replace("X", "")
            proteins.loc[idx, 'ncbi_seq_mutant'] = mutant_seq
            proteins.loc[idx, 'ncbi_seq'] = mutant_seq
            seq = mutant_seq
        else:
            proteins.loc[idx, 'ncbi_seq'] = seq
        uniprot_domain = uniprot_row['final_uniprot_domain']
        start_idx = seq.find(uniprot_domain)
        if start_idx!=-1:
            end_idx = min(start_idx+len(uniprot_domain), len(seq))
            proteins.loc[idx, 'ncbi_domain_loc'] = f"{start_idx+1}-{end_idx}"
            proteins.loc[idx, 'ncbi_domain'] = uniprot_domain
        else:
            alignments, start_idx, end_idx = align_domain(seq, uniprot_domain)
            match_len = end_idx-start_idx
            if match_len==len(uniprot_domain):
                proteins.loc[idx, 'ncbi_domain_loc'] = f"{start_idx + 1}-{end_idx}"
                proteins.loc[idx, 'ncbi_domain'] = seq[start_idx:end_idx]
            elif row['Accession Number'] in ['NP_002084.2', 'NP_060042.2']:
                proteins.loc[idx, 'ncbi_domain_loc'] = f"{start_idx + 1}-{end_idx}"
                proteins.loc[idx, 'ncbi_domain'] = seq[start_idx:end_idx]
            else:
                print(row['Entrez Gene Symbol'], row['Kinase'], row['Accession Number'], uniprot_row['final_uniprot_id'])
                print(alignments)
        if len(seq)>1370:
            proteins.loc[idx, 'ncbi_seq_truncated'] = proteins.loc[idx, 'ncbi_domain']
            proteins.loc[idx, 'ncbi_seq'] = proteins.loc[idx, 'ncbi_domain']
            proteins.loc[idx, 'ncbi_domain_loc'] = f"1-{len(proteins.loc[idx, 'ncbi_domain'])}"

    with pd.ExcelWriter('davis_proteins.xlsx', mode='a', if_sheet_exists='replace') as writer:
        proteins.to_excel(writer, sheet_name='ncbi_info', index=False)

def extract_deepdta_domain():
    proteins = pd.read_excel("davis_proteins.xlsx", sheet_name='DeepDTA')
    uniprot_proteins = pd.read_excel("davis_proteins.xlsx", sheet_name='uniprot')
    proteins['DeepDTA_domain_loc'] = None
    proteins['DeepDTA_domain'] = None
    for idx, row in proteins.iterrows():
        start_idx, end_idx = None, None
        uniprot_row = uniprot_proteins.loc[idx]
        uniprot_seq = uniprot_row['final_uniprot_seq']
        seq = row['DeepDTA_seq']
        uniprot_domain = uniprot_row['final_uniprot_domain']
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
                alignments, start_idx, end_idx = align_domain(seq, uniprot_domain)
                match_len = end_idx-start_idx
                if match_len==len(uniprot_domain):
                    proteins.loc[idx, 'DeepDTA_domain_loc'] = f"{start_idx + 1}-{end_idx}"
                    proteins.loc[idx, 'DeepDTA_domain'] = seq[start_idx:end_idx]
                elif row['Accession Number'] in ['NP_001706.2','NP_000052.1', 'NP_003709.3', 'NP_004435.3', 'NP_002750.1', 'NP_006276.2']:
                    proteins.loc[idx, 'DeepDTA_domain_loc'] = f"{start_idx + 1}-{end_idx}"
                    proteins.loc[idx, 'DeepDTA_domain'] = seq[start_idx:end_idx]
                else:
                    print(row['Entrez Gene Symbol'], row['Kinase'], row['Accession Number'], uniprot_row['final_uniprot_id'])
                    print(len(alignments))
                    print(pairwise2.format_alignment(*alignments[0]))
                    print('fail\n\n\n')

    # with pd.ExcelWriter('davis_proteins.xlsx', mode='a', if_sheet_exists='replace') as writer:
    #     proteins.to_excel(writer, sheet_name='DeepDTA', index=False)


def extract_alphafold_domain():
    proteins = pd.read_excel("davis_proteins.xlsx", sheet_name='uniprot')
    proteins['final_alphafold_domain_loc'] = None
    proteins['final_alphafold_domain'] = None
    for idx, row in proteins.iterrows():
        start_idx, end_idx = None, None
        uniprot_seq = row['final_uniprot_seq']
        seq = row['final_alphafold_seq']
        uniprot_domain = row['final_uniprot_domain']
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
                alignments, start_idx, end_idx = align_domain(seq, uniprot_domain)
                match_len = end_idx - start_idx
                if match_len == len(uniprot_domain):
                    proteins.loc[idx, 'final_alphafold_domain_loc'] = f"{start_idx + 1}-{end_idx}"
                    proteins.loc[idx, 'final_alphafold_domain'] = seq[start_idx:end_idx]
                else:
                    print(row['Entrez Gene Symbol'], row['Kinase'], row['Accession Number'],
                          row['final_uniprot_id'])
                    print(alignments)
    with pd.ExcelWriter('davis_proteins.xlsx', mode='a', if_sheet_exists='replace') as writer:
        proteins.to_excel(writer, sheet_name='uniprot', index=False)


if __name__=="__main__":
    """
    O14965
    Q00532
    P49761
    Q9Y4K4
    O75747
    """
    # davis_info_merge()
    # davis_download_from_uniprot(reviewed=False)
    """
    任务处理失败: NP_005149.3, 错误: Remote end closed connection without response
    任务处理失败: NP_079055.2, 错误: Remote end closed connection without response
    """
    # davis_download_from_ncbi()

    # davis_ncbi_bind()
    # davis_uniprot_bind()
    # davis_manual_update()
    # davis_download_uniprot_and_alphafold()
    #
    # extract_uniprot_domain()
    # extract_ncbi_domain()
    # extract_alphafold_domain()
    extract_deepdta_domain()
'''
    ans = """
    NP_005149.3	ABL2	P42684.2
    NP_000011.1	ACVRL1	P37023
    BAA36547.1	PRKAA1	Q13131
    AAH00442.2	AURKB	Q96GD4
    AAC77369.1	AURKC	Q9UQB9
    AAD20442.1	CAMK2D	Q13557
    NP_277023.1	CDC2L1	A4VCI5
    NP_076916.1	CDC2L2	Q4VBY6 
    AAA61480.1	CLK1	P49759.1
    NP_660204.1	CSNK1A1L	Q8N752
    NP_001310.2	CSNK1G2	P78368
    NP_001035351.3	DCLK2	Q8N568.2
    CAA52777.1	DDR2	Q16832
    NP_006292.2	MAP3K12	Q6ZN16
    NP_004751.1	STK17A	Q9UEE5
    NP_569121.1	DYRK1A	Q13627.4
    NP_005223.3	EPHA1	P21709
    NP_002011.1	FLT4	P35916
    NP_009130.1	IRAK3	Q9Y616
    NP_002218.1	JAK1	P23458
    NP_055387.1	LATS2	Q9NRM7
    NP_001001671.2	MAP3K15	Q6ZN16.3
    NP_663719.1	MAP4K4	O95819.3
    NP_002367.4	MARK3	P27448.3
    AAB60430.1	MERTK	Q12866
    NP_006272.1	STK3	Q13188
    NP_001012418.1	MYLK4	Q86YV6
    NP_001077084.1	MYO3B	Q8WXR4.4
    NP_149107.3	NEK9	Q8TD19
    AAB40118.1	MAPK12	P53778
    CAA47004.1	PCTK2	Q00537
    XP_001349680.1	PFB0815w	P62344
    XP_001350280.1	MAL13P1.279	P61075
    NP_002637.2	PIK3C2B	A2RUF7
    NP_004561.2	PIK3C2G	O75747
    NP_005017.2	PIK3CD	O00329
    NP_001001852.1	PIM3	K7BYJ6 
    NP_079055.2	PIP4K2C	Q8TBX8
    NP_214528.1	pknB	P9WI81
    NP_055079.2	PLK4	O00444
    NP_002731.3	PRKCI	P41743
    BAA76843.2	KIAA0999	Q9Y2K2
    NP_060813.1	RIOK2	Q9BVS4
    Q6XUX3.1	DSTKY	Q6XUX3
    NP_001006933.1	RPS6KA2	Q15349.3
    P0C264	SgK110	P0C264
    AAC05299.1	SRPK2	P78362
    CAA06700.1	STK16	O75716
    NP_543026.1	STK35	Q8TDR2
    NP_056505.1	STK36	Q9NRP7
    NP_057365.2	TAOK3	A0A6D2VXT9
    NP_003206.1	TEC	P42680
    AAF03095.1	TLK2	Q86UE8
    AAA75374.1	NTRK3	Q16288
    NP_003322.2	TYK2	P29597
    NP_055498.2	ULK2	Q8IYT8 
    P0C1S8	WEE2	P0C1S8
    """.split()
    items = {}
    for idx, name, pid in zip(ans[::3], ans[1::3], ans[2::3]):
        items[idx] = pid
    with open("manual_anno.json", "w") as f:
        json.dump(items, f, indent=2)
'''

