import os
import shutil
import zipfile
from tqdm import tqdm


def add_prefix_to_files_os(target_dir, prefix, pattern):
    target_dir = os.path.abspath(target_dir)
    if prefix==pattern:
        return

    for root, _, files in os.walk(target_dir):
        for original_filename in files:
            if original_filename.startswith(pattern):
                original_path = os.path.join(root, original_filename)
                new_filename = original_filename.replace(pattern, prefix)
                new_path = os.path.join(root, new_filename)

                if os.path.exists(new_path):
                    try:
                        os.remove(new_path)  # 删除目标文件
                        print(f"已删除现有目标文件：{new_path}")
                    except Exception as e:
                        raise RuntimeError(f"无法删除目标文件 {new_path}：{str(e)}") from e

                os.rename(original_path, new_path)



if __name__=='__main__':
    file_dir = "../data/uniprot_alphafold_struct/dogsite3/result4"
    for file in tqdm(os.listdir(file_dir)):
        # if not file.startswith("9de78470c8b87c25"):
        #     continue
        if file.endswith(".zip"):
            extract_dir = os.path.join(os.path.dirname(file_dir), 'pockets', file.replace(".zip", ""))
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)
            zip_path = os.path.join(file_dir, file)
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            except:
                print(file)
            tgt_file = [f for f in os.listdir(extract_dir) if f.endswith('_desc.txt')]
            tgt_file = sorted(tgt_file, key=lambda x:len(x))[-1]
            assert file.replace(".zip", "") in os.path.basename(tgt_file)

            name = tgt_file.replace("_desc.txt", "", 1)
            add_prefix_to_files_os(extract_dir, file.replace(".zip", ""), name)
