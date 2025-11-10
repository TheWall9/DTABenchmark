import os
import json
import pandas as pd

if __name__ == '__main__':
    davis = pd.read_excel("davis_final_alphafold_merged.xlsx")
    kiba = pd.read_excel("kiba_final_alphafold_merged.xlsx")
    save_file = '../data/uniprot_alphafold_struct/seq_struct_map1.json'

    ans = {}
    cols = ['DeepDTA_seq', 'ncbi_seq', 'uniprot_seq', 'final_alphafold_seq']
    for idx, row in davis.iterrows():
        for col in cols:
            seq = str(row[col])
            if seq not in ['NaN', "", 'nan']:
                file = row['struct_file']
                tmp = ans.get(seq, file)
                if tmp!=file:
                    print(row)
                ans[row[col]] = os.path.join("alphafold", file)

    cols = ['DeepDTA_seq', 'uniprot_seq', 'final_alphafold_seq']
    for idx, row in kiba.iterrows():
        for col in cols:
            seq = str(row[col])
            if seq not in ['NaN', "", 'nan']:
                file = row['struct_file']
                tmp = ans.get(seq, file)
                if tmp!=file:
                    print(row)
                ans[row[col]] = os.path.join("alphafold", file)
                """
                K7BYJ6: Q86V86
                """

    with open(save_file, 'w') as f:
        json.dump(ans, f, indent=1)