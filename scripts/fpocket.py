import os
import json
import shutil
from functools import partial
from toolbox.utils import MultiThreadProcessor, exec_command
from toolbox.featurizer.protein import StructureLoader


from Bio.PDB import PDBParser, PDBIO, Select


class PocketSelect(Select):
    def __init__(self, residues_set):
        self.residues_set = residues_set

    def accept_residue(self, residue):
        chain_id = residue.get_parent().id
        resseq = residue.id[1]
        icode = residue.id[2]
        return (chain_id, resseq, icode) in self.residues_set


def fpocket_fn(file, root_dir, save_dir, tmp_dir):
    pdb_file = os.path.join(root_dir, file)
    name, ext = os.path.splitext(os.path.basename(pdb_file))
    out_dir = os.path.join(save_dir, f"{name}_out")

    if os.path.exists(out_dir):
        return
    if pdb_file.endswith('.cif'):
        in_pdb_file2 = os.path.abspath(os.path.join(tmp_dir, file.replace(ext, ".pdb")))
        os.makedirs(os.path.dirname(in_pdb_file2), exist_ok=True)
        if not os.path.exists(in_pdb_file2):
            cmd = f"obabel -icif {pdb_file} -opdb -O {in_pdb_file2}"
            exec_command(cmd)
        pdb_file = in_pdb_file2
    cmd = f"fpocket -f {pdb_file}"
    exec_command(cmd)
    tmp_dir = os.path.join(os.path.dirname(pdb_file), f"{name}_out")
    if os.path.exists(tmp_dir):
        shutil.move(tmp_dir, out_dir)

    parser = PDBParser(QUIET=True)
    full_structure = parser.get_structure("full", pdb_file)
    os.makedirs(os.path.join(out_dir, "full_pockets"), exist_ok=True)
    for file in os.listdir(os.path.join(out_dir, "pockets")):
        if not file.endswith(".pdb"):
            continue
        pocket_structure = parser.get_structure("pocket", os.path.join(out_dir, 'pockets', file))
        pocket_residues = set()
        for atom in pocket_structure.get_atoms():
            residue = atom.get_parent()
            chain_id = residue.get_parent().id  # Residue 的 parent 是 Chain
            resseq = residue.id[1]
            icode = residue.id[2]  # 注意 insertion code
            pocket_residues.add((chain_id, resseq, icode))  # 加入 icode 保证唯一性
        output_file = os.path.join(out_dir, "full_pockets", file)
        io = PDBIO()
        io.set_structure(full_structure)
        selector = PocketSelect(pocket_residues)
        io.save(output_file, selector)

if __name__=="__main__":
    loader = StructureLoader(struct_type='esmfold')
    with open(loader.seq_struct_map_file) as f:
        data = json.load(f)
    root_dir = loader.struct_root_dir
    save_dir = "/mnt/data/lcc/WorkSpace/DTA_Space/Benchmark/data/uniprot_alphafold_struct/fpocket"
    tmp_dir = '/mnt/data/lcc/WorkSpace/DTA_Space/Benchmark/cached/featurizer/inputs/StructureLoader'
    seqs = sorted(set([item['file'] for item in data.values()]))
    processor = MultiThreadProcessor(max_workers=2)
    func = partial(fpocket_fn, root_dir=root_dir, save_dir=save_dir, tmp_dir=tmp_dir)
    # func(seqs[0])
    # exit()
    processor.process_batch(seqs, func)
    print(processor.get_failed_tasks())
    """AF-Q00537-F1-model_v4.pdb"""



