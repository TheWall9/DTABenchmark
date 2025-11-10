import os
import json

import torch
from transformers import AutoTokenizer, EsmForProteinFolding
from tqdm import tqdm
import pandas as pd
from datasets.fingerprint import Hasher


if __name__=="__main__":
    # file = 'seq_struct_map.json'
    new_map_file = "../data/uniprot_alphafold_struct/seq_struct_esmfold_map2.json"
    save_dir = "../data/uniprot_alphafold_struct/esmfold_v1"
    os.makedirs(save_dir, exist_ok=True)

    with open(new_map_file) as f:
        data = json.load(f)
    data = sorted(data.keys(), key=lambda x: len(x))
    data = pd.read_excel("debug.xlsx")['ncbi_seq'].values
    model_name = './esmfold_v1'
    model_name = 'facebook/esmfold_v1'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EsmForProteinFolding.from_pretrained(model_name, low_cpu_mem_usage=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    model.eval()
    model.to(device)
    # if device == "cuda":
    #     model = model.half()

    ans = {}
    for sequence in tqdm(data):
        hash = Hasher.hash(sequence)
        name = f"{sequence[:5]}-{hash}-esmfold-v1.pdb"
        save_file = os.path.join(save_dir, name)
        ans[save_file] = name
        if os.path.exists(save_file):
            continue
        print(len(sequence))
        with torch.no_grad():
            print(len(sequence))
            outputs = model.infer_pdb(sequence)
        with open(save_file, "w") as f:
            f.write(outputs)
        torch.cuda.empty_cache()
    # with open(new_map_file, "w") as f:
    #     json.dump(ans, f)

