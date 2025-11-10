import os
import shutil
import pandas as pd
import json
from collections import OrderedDict

if __name__ == '__main__':
    data_dir = '../../data/davis/'
    protein_file = os.path.join(data_dir, 'proteins.txt')
    tgt_dir = '../../data/davis_domain2/'
    ref_file = "../davis_proteins.xlsx"
    os.makedirs(tgt_dir, exist_ok=True)
    ref = pd.concat([pd.read_excel(ref_file, sheet_name='DeepDTA'),pd.read_excel(ref_file, sheet_name='ncbi_info') ], axis=1)
    shutil.copytree(data_dir, tgt_dir, dirs_exist_ok=True)
    with open(protein_file, 'r') as f:
        proteins = json.load(f, object_hook=OrderedDict)
    ref_map = {row['DeepDTA_gene_name']:row['DeepDTA_domain'] if not pd.isna(row['DeepDTA_domain']) else row['DeepDTA_seq'] for idx, row in ref.iterrows()}
    ans = {}
    for key in proteins:
        ans[key] = ref_map[key]
    with open(os.path.join(tgt_dir, 'proteins.txt'), 'w') as f:
        json.dump(ans, f, indent=1)
