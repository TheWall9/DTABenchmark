import os
import json

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from tqdm import tqdm
import pandas as pd
from toolbox.utils import Hasher, read_mmcif_structure
from Bio import pairwise2, PDB
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain

def align_domain(seq, domain):
    alignments = pairwise2.align.localms(domain, seq, 2, -1, -3, -3)
    start_idx = alignments[0].start
    end_idx = alignments[0].end
    return alignments, start_idx, end_idx

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

def parse_struct(file):
    assert os.path.exists(file)
    structure_id = os.path.splitext(os.path.basename(file))[0]
    if file.endswith(".pdb"):
        parser = PDB.PDBParser(QUIET=True)
    elif file.endswith(".cif"):
        parser = PDB.MMCIFParser(QUIET=True)
    else:
        raise NotImplementedError
    struct = parser.get_structure(structure_id, file)
    return struct

if __name__=="__main__":
    # file = '../data/uniprot_alphafold_struct/seq_struct_map.json'
    struct_dir = "../data/uniprot_alphafold_struct"
    new_map_file = "../data/uniprot_alphafold_struct/seq_struct_esmfold_map4.json"
    save_dir = "../data/uniprot_alphafold_struct/esmfold_v1"
    os.makedirs(save_dir, exist_ok=True)
    kiba_file = "kiba_proteins.xlsx"
    data = []
    with pd.ExcelFile(kiba_file) as f:
        for sheet in f.sheet_names:
            data.append(f.parse(sheet))
    kiba_data = pd.concat(data, axis=1)
    kiba_data = kiba_data.loc[:, ~kiba_data.columns.duplicated()]
    davis_file = "davis_proteins.xlsx"
    data = []
    with pd.ExcelFile(davis_file) as f:
        for sheet in f.sheet_names:
            data.append(f.parse(sheet))
    davis_data = pd.concat(data, axis=1)
    davis_data = davis_data.loc[:, ~davis_data.columns.duplicated()]
    ans = []
    task = []
    seqs = []
    extra = []
    dataset = [kiba_data, davis_data]
    for data in dataset:
        ### 处理ncbi seq
        for idx, row in data.iterrows():
            value = {"file": "alphafold/"+row['struct_file']}
            if not pd.isna(row['final_alphafold_domain']):
                value['domain'] = ({"loc": list(map(int, row['final_alphafold_domain_loc'].split('-'))),
                                    "seq": row['final_alphafold_domain']},)
            ans.append((row['final_alphafold_seq'], value))
            ### alphafold domain
            if 'ncbi_seq' in data.columns:
                if row['ncbi_seq']!=row['final_alphafold_seq']:
                    hash = Hasher.hash(str(row['ncbi_seq']))
                    name = f"{row['ncbi_seq'][:5]}-{hash}-esmfold-v1.pdb"
                    name2 = f"flash{row['ncbi_seq'][:5]}-{hash}-esmfold-v1.pdb"
                    name3 = f"{row['ncbi_seq'][:5]}-{hash}-{row['final_uniprot_id']}.pdb"
                    if row['ncbi_seq'] in row['final_alphafold_seq']:
                        value = {"file": f"alphafold_fragment/{name3}"}
                        tmp_save_file = os.path.join(struct_dir, value['file'])
                        #### alphafold 片段
                        if not os.path.exists(tmp_save_file):
                            alignments, start_idx, end_idx = align_domain(row['final_alphafold_seq'], row['ncbi_seq'])
                            origin_struct_file = os.path.join(struct_dir, 'alphafold', row['struct_file'])
                            structure = parse_struct(origin_struct_file)
                            chain = next(iter(structure.get_chains()))
                            res = list(chain)
                            new_structure = Structure.Structure("new_structure")
                            new_model = Model.Model(0)
                            new_chain = Chain.Chain("A")
                            for res in res[start_idx:end_idx]:
                                new_chain.add(res)  # 添加残基到新链
                            new_model.add(new_chain)
                            new_structure.add(new_model)
                            io = PDBIO()
                            io.set_structure(new_structure)
                            io.save(tmp_save_file)
                        structure = parse_struct(tmp_save_file)
                        s_seq = get_sequence(structure)
                        assert s_seq==row['ncbi_seq']
                        alignment, start_idx, end_idx = align_domain(s_seq, row['final_uniprot_domain'])
                        value['domain'] = ({"loc": (start_idx+1, end_idx),
                                            "seq": s_seq[start_idx:end_idx]},)
                        ans.append((row['ncbi_seq'], value))

                    ### esmfold 预测ncbi_seq
                    elif os.path.exists(os.path.join(save_dir, name)):
                        value = {"file": "esmfold_v1/"+name}
                        if not pd.isna(row['ncbi_domain']):
                            value['domain'] = ({"loc": list(map(int, row['ncbi_domain_loc'].split('-'))),
                                                "seq": row['ncbi_domain']},)
                        ans.append((row['ncbi_seq'], value))
                    elif os.path.exists(os.path.join(save_dir, name2)):
                        value = {"file": "esmfold_v1/"+name2}
                        if not pd.isna(row['ncbi_domain']):
                            value['domain'] = ({"loc": list(map(int, row['ncbi_domain_loc'].split('-'))),
                                                "seq": row['ncbi_domain']},)
                        ans.append((row['ncbi_seq'], value))
                    else:
                        # ans.append((row['ncbi_seq'], "esmfold_v1/" + name))
                        extra.append(row.to_dict())

    seq_set = set([item[0] for item in ans])
    for data in dataset:
        for idx, row in data.iterrows():
            s = row['DeepDTA_seq']
            if s not in seq_set:
                hash = Hasher.hash(str(s))
                name = f"{s[:5]}-{hash}-esmfold-v1.pdb"
                name2 = f"flash{s[:5]}-{hash}-esmfold-v1.pdb"
                if os.path.exists(os.path.join(save_dir, name)):
                    value = {"file": "esmfold_v1/"+name}
                    if not pd.isna(row['DeepDTA_domain']):
                        value['domain'] = ({"loc": list(map(int, row['DeepDTA_domain_loc'].split('-'))),
                                            "seq": row['DeepDTA_domain']},)
                    ans.append((s, value))
                elif os.path.exists(os.path.join(save_dir, name2)):
                    value = {"file": "esmfold_v1/"+name2}
                    if not pd.isna(row['DeepDTA_domain']):
                        value['domain'] = ({"loc": list(map(int, row['DeepDTA_domain_loc'].split('-'))),
                                            "seq": row['DeepDTA_domain']},)
                    ans.append((s, value))
                else:
                    task.append(row.to_dict())



    # ans = sorted(set(ans))
    # ans = sorted(set([(item[0], json.dumps(item[1])) for item in ans]))
    seq_set = set([item[0] for item in ans])
    # assert len(seq_set)==len(ans)
    task = pd.DataFrame(task)
    """P78527"""
    len(ans)
    row = task.iloc[0]
    assert row['final_uniprot_id']=='P78527'
    alignments, start_idx, end_idx = align_domain(row['final_alphafold_seq'], row['final_uniprot_domain'])
    hash = Hasher.hash(str(row['final_alphafold_seq'][start_idx:end_idx]))
    name3 = f"{row['final_alphafold_seq'][:5]}-{hash}-{row['final_uniprot_id']}.pdb"
    value = {"file": f"alphafold_fragment/{name3}"}
    tmp_save_file = os.path.join(struct_dir, value['file'])
    if not os.path.exists(tmp_save_file):
        origin_struct_file = os.path.join(struct_dir, 'alphafold', row['struct_file'])
        structure = parse_struct(origin_struct_file)
        chain = next(iter(structure.get_chains()))
        res = list(chain)
        new_structure = Structure.Structure("new_structure")
        new_model = Model.Model(0)
        new_chain = Chain.Chain("A")
        for res in res[start_idx:end_idx]:
            new_chain.add(res)  # 添加残基到新链
        new_model.add(new_chain)
        new_structure.add(new_model)
        io = PDBIO()
        io.set_structure(new_structure)
        io.save(tmp_save_file)
    structure = parse_struct(tmp_save_file)
    s_seq = get_sequence(structure)
    assert s_seq == row['final_uniprot_domain']
    alignment, start_idx, end_idx = align_domain(s_seq, row['final_uniprot_domain'])
    value['domain'] = ({"loc": (start_idx+1, end_idx),
                        "seq": s_seq[start_idx:end_idx]},)
    output = dict(ans)
    # output = {key: json.loads(value) for key, value in output.items()}
    output[row['final_alphafold_seq']] = value
    output[row['DeepDTA_seq']] = output[row['final_alphafold_seq']]
    output[s_seq[start_idx:end_idx]] = value
    with open(new_map_file, 'w') as f:
        json.dump(output, f, indent=1)

