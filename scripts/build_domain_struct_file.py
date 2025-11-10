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
    struct_dir = "../data/uniprot_alphafold_struct"
    map_file = "../data/uniprot_alphafold_struct/seq_struct_esmfold_map4.json"
    struct_save_file = "../data/uniprot_alphafold_struct/domain_fragment/"
    with open(map_file) as f:
        map_data = json.load(f)
    ans = {}
    for seq in tqdm(map_data):
        domain = map_data[seq].get('domain')
        if domain is not None:
            struct_file = os.path.join(struct_dir, map_data[seq]['file'])
            domain_seq = domain[0]['seq']
            if domain_seq in map_data:
                print("skip", domain_seq)
                continue
            name = os.path.basename(map_data[seq]['file'])
            name = ".".join(name.split(".")[:-1])
            hash = Hasher.hash(str(domain_seq))
            save_file = os.path.join(struct_save_file, os.path.dirname(map_data[seq]['file']), f"domain-{hash}-{name}.pdb")
            ans[domain_seq] = {"file": os.path.join("domain_fragment", os.path.dirname(map_data[seq]['file']), f"domain-{hash}-{name}.pdb")}
            start_idx, end_idx = domain[0]['loc']
            if not os.path.exists(save_file):
                structure = parse_struct(struct_file)
                chain = next(iter(structure.get_chains()))
                res = list(chain)
                new_structure = Structure.Structure("new_structure")
                new_model = Model.Model(0)
                new_chain = Chain.Chain("A")
                for res in res[start_idx-1:end_idx]:
                    new_chain.add(res)  # 添加残基到新链
                new_model.add(new_chain)
                new_structure.add(new_model)
                io = PDBIO()
                io.set_structure(new_structure)
                if not os.path.exists(os.path.dirname(save_file)):
                    os.makedirs(os.path.dirname(save_file))
                io.save(save_file)
            structure = parse_struct(save_file)
            s_seq = get_sequence(structure)
            try:
                assert s_seq==domain_seq.replace("X", ""), save_file
            except Exception as e:
                print(save_file)
                raise e

    map_data.update(ans)
    with open(map_file, 'w') as f:
        json.dump(map_data, f, indent=2)