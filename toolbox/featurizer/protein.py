import os
import gc
import json
import pickle
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import pairwise_distances
import torch
from torch.nn import functional as F
import tokenizers as tk
from transformers import AutoTokenizer, AutoModel, pipeline
from torch_geometric.data import Batch

from Bio import PDB
from Bio.Data.IUPACData import protein_letters_3to1

from deepchem import feat
from deepchem.utils import get_atomz


from toolbox.featurizer.tools import FeaturizerBase, FeatData, GraphData
from toolbox.utils import exec_command, Hasher, read_mmcif_structure, disk_cache, serialize3d
from toolbox.config import FEATURIZER_INPUT_TEMP_DIR, STRUCT_ROOT_DIR, HHSUITE_DB_PATH

from repo.gvp.data import ProteinGraphDataset
from repo.dMaSIF.geometry_processing import atoms_to_points_normals, curvatures

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SimpleProtTokenFeaturizer(FeaturizerBase):
    def __init__(self, protein_max_lengths=1000):
        super().__init__()
        tokenizer_file = os.path.join(self.tokenizer_dir, "DeepDTA_Protein_tokenizer.json")
        tokenizer = tk.Tokenizer.from_file(tokenizer_file)
        if protein_max_lengths is None:
            tokenizer.no_padding()
            tokenizer.no_truncation()
        else:
            tokenizer.enable_truncation(max_length=protein_max_lengths)
            tokenizer.enable_padding(length=protein_max_lengths)
        self.protein_max_lengths = protein_max_lengths
        self.tokenizer = tokenizer

    def _featurize(self, datapoint, **kwargs):
        seq = str(datapoint)
        if self.protein_max_lengths is not None:
            return FeatData(input_ids=self.tokenizer.encode(seq).ids, prefix='protein')
        tokens = self.tokenizer.encode(seq)
        input_ids_batch = np.zeros(len(tokens.ids), dtype=int)
        return FeatData(graph=GraphData(input_ids=tokens.ids, input_ids_batch=input_ids_batch), prefix='protein')

    def get_feat_info(self, data=None):
        return {'num_protein_tokens': self.tokenizer.get_vocab_size()}


class LLMFeaturizer(FeaturizerBase):
    def __init__(self, pretrained_model_name_or_path="Rostlab/prot_bert", device="cuda", feat_type='cls', start_idx=1, end_idx=-1,
                 protein_max_lengths=None):
        super().__init__()
        if protein_max_lengths is None and r"facebook/esm1b" in pretrained_model_name_or_path:
            protein_max_lengths = 1024
        assert feat_type in ['cls', 'mean', 'max', 'seq', 'full']
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = device
        self.feat_type = feat_type
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.protein_max_lengths = protein_max_lengths
        self.save_dir = os.path.join(self.cached_root_output_dir, os.path.basename(pretrained_model_name_or_path))
        os.makedirs(self.save_dir, exist_ok=True)

    def load_pipeline(self, pretrained_model_name_or_path, device):
        protein_max_lengths = self.protein_max_lengths
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, do_lower_case=False)
        if protein_max_lengths is not None:
            tokenizer.model_max_length = protein_max_lengths
        model = AutoModel.from_pretrained(pretrained_model_name_or_path)
        model.eval()
        pipe = pipeline(
            "feature-extraction",
            model=model,
            tokenizer=tokenizer,
            device=device,
            truncation=protein_max_lengths is not None,
        )
        return pipe


    def _featurize(self, datapoint, **kwargs):
        hash = Hasher.hash(datapoint)
        save_file = os.path.join(self.save_dir, f'{hash}.pickle')
        if not os.path.exists(save_file):
            if not hasattr(self, "model") or self.model is None:
                self.model = self.load_pipeline(self.pretrained_model_name_or_path, device=self.device)
            seq = " ".join(list(datapoint))
            last_hidden_state = self.model(seq, return_tensors=True) # shape: (1, seq_len, 1024) # shape: (1024,)
            if self.protein_max_lengths is None:
                assert len(list(range(last_hidden_state.shape[1]))[self.start_idx:self.end_idx])==len(datapoint)
            cls_feat = last_hidden_state[0, 0].cpu()
            seq_feat = last_hidden_state[0, self.start_idx:self.end_idx]
            mean_feat = seq_feat.mean(dim=0).cpu()
            max_feat = seq_feat.max(dim=0)[0].cpu()
            feats = {"cls": cls_feat.numpy(),
                    "full": last_hidden_state.numpy()[0],
                    "mean": mean_feat.numpy(),
                    "max": max_feat.numpy(),}
            with open(save_file, "wb") as f:
                pickle.dump(feats, f)
        with open(save_file, "rb") as f:
            feats = pickle.load(f)
        if self.feat_type=='seq':
            embedding = feats.get('full')[self.start_idx:self.end_idx]
            if self.protein_max_lengths is not None:
                embedding = embedding[:self.protein_max_lengths]
            return FeatData(graph=GraphData(inputs_embeds=embedding,
                                            inputs_embeds_batch=np.zeros(len(embedding), dtype=int)), prefix='protein')
        elif self.feat_type=='full':
            embedding = feats.get('full')
            if self.protein_max_lengths is not None:
                embedding = embedding[:self.protein_max_lengths]
            return FeatData(graph=GraphData(inputs_embeds=embedding,
                                            inputs_embeds_batch=np.zeros(len(embedding), dtype=int)), prefix='protein')
        return FeatData(embedding=feats.get(self.feat_type), prefix='protein')

    def __init_subclass__(cls, **kwargs):
        raise TypeError(f"{cls.__name__} cannot inherit from FinalClass")

class LLMStructFeaturizer(FeaturizerBase):
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR, struct_type='esmfold', pocket_type=None, pocket_top=3,
                 pretrained_model_name_or_path="Rostlab/prot_bert", start_idx=1, end_idx=-1, n_res_expand=0):
        super().__init__()
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.struct_root_dir_type = struct_root_dir
        self.struct_type = struct_type
        self.pocket_type = pocket_type
        self.distance_cutoff = 8
        self.pocket_top = pocket_top
        top = 1 if self.pocket_type is None else self.pocket_top
        self.n_res_expand = n_res_expand
        self.pocket_loader = StructureLoader(struct_root_dir=struct_root_dir, struct_type=struct_type, pocket_type=pocket_type, top=top)
        self.struct_loader = StructureLoader(struct_root_dir=struct_root_dir, struct_type=struct_type)
        self.llm_featurizer = LLMFeaturizer(pretrained_model_name_or_path, feat_type='seq', start_idx=start_idx, end_idx=end_idx)
        self.save_dir = os.path.join(self.cached_root_output_dir, os.path.basename(pretrained_model_name_or_path), f'{struct_type}_{pocket_type}_{pocket_top}_{n_res_expand}')
        os.makedirs(self.save_dir, exist_ok=True)


    def _featurize(self, datapoint, **kwargs):
        hash = Hasher.hash(str(datapoint))
        save_file = os.path.join(self.save_dir, f'{hash}.pkl')
        n_res_expand = self.n_res_expand
        if not os.path.exists(save_file):
            full_struct_file = self.struct_loader.get_struct_by_seq(datapoint)[0]
            full_struct = self.struct_loader.parse_struct(full_struct_file)
            full_chain = next(iter(full_struct.get_chains()))
            full_res = self.struct_loader.get_res_coords(full_chain)
            pos_idx_map = np.ones(full_res['seq_pos_idx'].max()+1, dtype=int)*(full_res['seq_pos_idx'].max()+1)
            for i, idx in enumerate(full_res['seq_pos_idx']):
                pos_idx_map[idx] = i
            pocket_ids = []
            pocket_struct_files = self.pocket_loader.get_struct_by_seq(datapoint)
            for i, struct_file in enumerate(pocket_struct_files):
                structure = StructureLoader.parse_struct(struct_file)
                chains = list(structure.get_chains())
                assert len(chains) == 1
                chain = chains[0]
                for res in chain:
                    ids = res.id[1]
                    pocket_ids.extend(list(range(max(1, ids-n_res_expand), ids+n_res_expand+1)))
            pocket_ids = sorted(set(pocket_ids))
            new_chain = PDB.Chain.Chain('X')
            pocket_ids = [ids for ids in pocket_ids if ids in full_chain]
            for ids in pocket_ids:
                new_chain.add(full_chain[ids])
            pocket_res = self.struct_loader.get_res_coords(new_chain)

            pos = pocket_res['coords']
            coords = pos[:, 1]
            distance = pairwise_distances(coords, metric='euclidean')
            np.fill_diagonal(distance, np.inf)
            contact_map = distance < self.distance_cutoff
            row, col = np.where(contact_map)
            edge_index = np.stack([row, col])
            seq_pos_idx = pos_idx_map[pocket_res['seq_pos_idx']]
            full_seq = full_res['res_seq']
            seq = pocket_res['res_seq']
            assert "".join(np.array(list(full_seq))[seq_pos_idx])==seq
            with open(save_file, 'wb') as f:
                pickle.dump({"edge_index":edge_index,
                             "pos":pos,
                             "seq_pos_idx": seq_pos_idx,
                             "full_seq": full_seq,
                             "seq":seq}, f)
        with open(save_file, 'rb') as f:
            data = pickle.load(f)
        full_feat = self.llm_featurizer.featurize([data['full_seq']])[0].graph.inputs_embeds
        inputs_embeds = full_feat[data['seq_pos_idx']].astype(float)
        edge_index = data['edge_index'].astype(float)
        pos = data['pos'].astype(float)
        node_batch = np.zeros(len(pos), dtype=int)
        graph = GraphData(node_features=inputs_embeds, edge_index=edge_index, pos=pos, node_batch=node_batch)
        return FeatData(graph=graph, prefix='protein')


class ProtPssmFeaturizer(FeaturizerBase):
    AMINO_ACIDS_ORDER = "ACDEFGHIKLMNPQRSTVWY" # HHSuite 默认顺序
    BLOSUM62_BG = {
        "A": 0.074, "R": 0.052, "N": 0.045, "D": 0.054, "C": 0.025,
        "Q": 0.034, "E": 0.054, "G": 0.074, "H": 0.026, "I": 0.068,
        "L": 0.099, "K": 0.058, "M": 0.025, "F": 0.047, "P": 0.039,
        "S": 0.057, "T": 0.051, "W": 0.013, "Y": 0.032, "V": 0.073
    }
    def __init__(self, hhsuite_db_path=HHSUITE_DB_PATH, hhsuite_n_cpus=8, hhsuite_n_iterations=3, use_ppm=False):
        super().__init__()
        self.hhsuite_db_path = os.path.abspath(hhsuite_db_path)
        self.hhsuite_n_cpus = hhsuite_n_cpus
        self.hhsuite_n_iterations = hhsuite_n_iterations
        self.output_temp_dir = os.path.abspath(self.cached_root_output_dir)
        self.cache_dir = os.path.abspath(self.cached_root_input_dir)
        self.use_ppm = use_ppm
        assert " " not in self.hhsuite_db_path and " " not in self.cache_dir
        self.fasta_temp_dir = os.path.join(self.cache_dir, "fasta")
        self.hhsuite_temp_dir = os.path.join(self.cache_dir, "hhsuite")
        self.verbose = False
        os.makedirs(self.output_temp_dir, exist_ok=True)
        os.makedirs(self.fasta_temp_dir, exist_ok=True)
        os.makedirs(self.hhsuite_temp_dir, exist_ok=True)

    def extract_ppm_from_hmm(self, hhm_file):
        ppm = []
        with open(hhm_file) as f:
            lines = f.readlines()
        start = None
        for i, line in enumerate(lines):
            if line.startswith("HMM"):
                start = i + 3  # 矩阵开始在 "HMM" 行之后
                break
        if start is None:
            raise ValueError("找不到 HMM 部分")
        i = start
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("//"):  # 文件结束
                break
            if not line:
                i += 1
                continue
            row1 = lines[i].strip().split()
            # row2 = lines[i + 1].strip().split()
            i += 3  # 每个位置占 3 行（2 行数据 + 空行）
            scores = []
            for x in row1[2:22]:
                if x == "*":  # 缺失值
                    scores.append(0)
                else:
                    scores.append(pow(2, -int(x)/1000.0))
            ppm.append(scores)
        ppm = np.asarray(ppm)
        ppm = pd.DataFrame(ppm, columns=list(self.AMINO_ACIDS_ORDER))
        return ppm

    def ppm_to_pssm(self, ppm, scale=2.0, use_blosum_bg=True):
        if use_blosum_bg:
            bg = np.array([self.BLOSUM62_BG[aa] for aa in self.AMINO_ACIDS_ORDER])
        else:
            bg = np.ones(20) / 20.0  # 均匀分布
        ppm = np.clip(ppm, 1e-9, 1.0)
        pssm = scale * np.log2(ppm / bg)
        return pssm

    def extract_pfm_from_a3m(self, a3m_file):
        seqs = []
        with open(a3m_file) as f:
            seq = ""
            for line in f:
                if line.startswith(">"):
                    if seq:
                        seqs.append(seq)
                        seq = ""
                else:
                    seq += ''.join([c for c in line.strip() if not c.islower()])
            if seq:
                seqs.append(seq)
        L = len(seqs[0])
        pfm = np.zeros((L, len(self.AMINO_ACIDS_ORDER)), dtype=int)
        aa_to_idx = {aa: i for i, aa in enumerate(self.AMINO_ACIDS_ORDER)}
        for seq in seqs:
            for i, res in enumerate(seq):
                if res not in aa_to_idx:
                    continue
                pfm[i, aa_to_idx[res]] += 1
        pfm = pd.DataFrame(data=pfm, columns=list(self.AMINO_ACIDS_ORDER))
        return pfm

    def _featurize(self, datapoint, **kwargs):
        hash = Hasher.hash(datapoint)
        pfm_file = os.path.join(self.output_temp_dir, f"{hash}.pfm.csv")
        ppm_file = os.path.join(self.output_temp_dir, f"{hash}.ppm.csv")
        if not os.path.exists(pfm_file):
            fasta_file = os.path.join(self.fasta_temp_dir, f"{hash}.fasta")
            hhr_file = os.path.join(self.hhsuite_temp_dir, f"{hash}.hhr")
            raw_a3m_file = os.path.join(self.hhsuite_temp_dir, f"{hash}_raw.a3m")
            a3m_file = os.path.join(self.hhsuite_temp_dir, f"{hash}.a3m")
            hhm_file = os.path.join(self.hhsuite_temp_dir, f"{hash}.hhm")
            # clu_file = os.path.join(self.hhsuite_temp_dir, f"{hash}.clu")
            if not os.path.exists(fasta_file):
                with open(fasta_file, "w") as f:
                    f.write(f">{hash}\n")
                    f.write(str(datapoint))
            if not os.path.exists(raw_a3m_file):
                hhblits_cmd = (f"hhblits -i {fasta_file} -d {self.hhsuite_db_path} "
                               f"-o {hhr_file} -oa3m {raw_a3m_file} "
                               f"-n {self.hhsuite_n_iterations} -cpu {self.hhsuite_n_cpus} "
                               )
                exec_command(hhblits_cmd, verbose=self.verbose)
            if not os.path.exists(a3m_file):
                hhfilter_cmd = (f"hhfilter -id 90 -i {raw_a3m_file} -o {a3m_file}")
                exec_command(hhfilter_cmd, verbose=self.verbose)
            if not os.path.exists(hhm_file):
                hhmake_cmd = f"hhmake -i {a3m_file} -o {hhm_file}"
                exec_command(hhmake_cmd, verbose=self.verbose)
            # if not os.path.exists(clu_file):
            #     reformat_cmd = f"reformat.pl a3m clu {a3m_file} {clu_file}"
            #     exec_command(reformat_cmd, verbose=self.verbose)
            ppm = self.extract_ppm_from_hmm(hhm_file)
            pfm = self.extract_pfm_from_a3m(a3m_file)
            ppm.to_csv(ppm_file, index=False)
            pfm.to_csv(pfm_file, index=False)
            # pfm = self.extract_pfm_from_clu(clu_file)
            # pfm = self.extract_pfm_from_hmm(hhm_file)
            # pfm.to_csv(pfm_file, index=False, compression='gzip')
        # pfm = pd.read_csv(pfm_file)
        ppm = pd.read_csv(ppm_file)
        if self.use_ppm:
            return ppm
        pssm = self.ppm_to_pssm(ppm)
        data = GraphData(inputs_embeds=pssm.values, inputs_embeds_batch=np.zeros(len(pssm), dtype=int))
        return FeatData(graph=data, prefix='protein')

    def __init_subclass__(cls, **kwargs):
        raise TypeError(f"{cls.__name__} cannot inherit from FinalClass")


class ContactMapFeaturizer(FeaturizerBase):
    def __init__(self, struct_root_dir, distance_cutoff=8.0, coord_type='cb', feat_name='ContactMapFeaturizer'):
        super(ContactMapFeaturizer, self).__init__(feat_name=feat_name)
        assert coord_type in ['ca', 'cb', 'centriod']
        self.struct_root_dir = struct_root_dir
        self.struct_dir = self.struct_root_dir
        self.seq_struct_map_file = os.path.join(struct_root_dir, 'seq_struct_esmfold_map.json')
        assert os.path.exists(self.seq_struct_map_file) and os.path.exists(self.struct_dir)
        self.distance_cutoff = distance_cutoff
        self.coord_type = coord_type
        save_dir = os.path.join(self.cached_root_output_dir, f'{coord_type}{distance_cutoff}')
        self.disk_cache_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        # self.enable_cache(save_dir, overwrite=False)

    @disk_cache
    def _featurize(self, datapoint, **kwargs):
        """Q9Y2K2 -- Q14289"""
        seq = str(datapoint)
        if not hasattr(self, 'seq_struct_map'):
            with open(self.seq_struct_map_file) as f:
                self.seq_struct_map = json.load(f)

        struct_file = self.seq_struct_map[seq]['file']
        structure = StructureLoader.parse_struct(os.path.join(self.struct_dir, struct_file))
        chains = list(structure.get_chains())
        assert len(chains) == 1
        chain = chains[0]

        coords = []
        residues = []
        for residue in chain:
            residue_name = 'X'
            if PDB.is_aa(residue):
                if self.coord_type == 'ca':
                    coord = self.get_residue_CA_coord(residue)
                elif self.coord_type == 'cb':
                    coord = self.get_residue_CB_coord(residue)
                else:
                    coord = self.get_residue_centroid_coord(residue)
                residue_name = protein_letters_3to1[residue.get_resname().capitalize()]
            else:
                coord = self.get_residue_centroid_coord(residue)
            coords.append(coord)
            residues.append(residue_name)
        coords = np.asarray(coords)
        distance = pairwise_distances(coords, metric='euclidean')
        np.fill_diagonal(distance, np.inf)
        contact_map = distance<self.distance_cutoff
        row, col = np.where(contact_map)
        edge_weight = distance[row, col]
        new_seq = "".join(residues)
        edge_index = np.stack([row, col])
        return FeatData(graph=GraphData(edge_index=edge_index, edge_distance=edge_weight, seq=new_seq),
                        prefix='contactmap')


    def get_residue_centroid_coord(self, residue):
        atoms = []
        for atom in residue:
            if not atom.get_name().startswith('H'):
                atoms.append(atom.get_coord())
        if not atoms:
            return None
        return np.mean(atoms, axis=0)

    def get_residue_CA_coord(self, residue):
        try:
            ca = residue['CA']
        except KeyError:
            ca = residue['CB']
        return ca.get_coord()

    def get_residue_CB_coord(self, residue):
        try:
            cb = residue['CB']
        except KeyError:
            cb = residue['CA']
        return cb.get_coord()

    def get_feat_info(self):
        return {}

    def __init_subclass__(cls, **kwargs):
        raise TypeError(f"{cls.__name__} cannot inherit from FinalClass")



class GVPFeaturizer(FeaturizerBase):
    def __init__(self, edge_num=30, rbf_num=16):
        super(GVPFeaturizer, self).__init__()
        self.edge_num = edge_num
        self.rbf_num = rbf_num

    def _featurize(self, datapoint: Any, **kwargs):
        if isinstance(datapoint, PDB.Chain.Chain):
            datapoint = [datapoint]
        data_list = []
        for chain in datapoint:
            coords_info = StructureLoader.get_res_coords(chain)
            data_list.append({"seq": coords_info['res_seq'], "coords":np.array(coords_info['coords'][:, :4]), "name":""})
        dataset = ProteinGraphDataset(data_list, top_k=self.edge_num, num_rbf=self.rbf_num)
        graphs = [g for g in dataset]
        batch = Batch.from_data_list(data_list=graphs)
        ans = GraphData(node_features=batch.x.numpy(),
                        seq=batch.seq.numpy(),
                        edge_index=batch.edge_index.numpy(),
                        node_s=batch.node_s.numpy(),
                        node_v=batch.node_v.numpy(),
                        edge_s=batch.edge_s.numpy(),
                        edge_v=batch.edge_v.numpy(),
                        mask=batch.mask.numpy())
        return FeatData(graph=ans, prefix='protein')


class StructureLoader():
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR, struct_type='alphafold', pocket_type=None, top=1):
        assert struct_type in ('alphafold', 'esmfold')
        assert pocket_type in ('dogsite3', 'fpocket', None)
        self.struct_root_dir = struct_root_dir
        self.struct_type = struct_type
        self.pocket_type = pocket_type
        if struct_type=='alphafold':
            self.seq_struct_map_file = os.path.join(struct_root_dir, 'seq_struct_map.json')
        else:
            self.seq_struct_map_file = os.path.join(struct_root_dir, 'seq_struct_esmfold_map.json')
        self.cache_dir = os.path.join(FEATURIZER_INPUT_TEMP_DIR, 'StructureLoader', struct_type)
        self.top = top

    @classmethod
    def protonate(cls, pdb_file, cache_dir):
        # protonate (i.e., add hydrogens) a pdb using reduce and save to an output file.
        # in_pdb_file: file to protonate.
        # out_pdb_file: output file where to save the protonated pdb file.
        # Remove protons first, in case the structure is already protonated
        name, ext = os.path.splitext(os.path.basename(pdb_file))
        in_pdb_file = os.path.abspath(pdb_file)
        in_pdb_file2 = os.path.abspath(os.path.join(cache_dir, f"{name}.pdb"))
        hr_pdb_file = os.path.abspath(os.path.join(cache_dir, f"Hr_{name}.pdb"))
        ha_pdb_file = os.path.abspath(os.path.join(cache_dir, f"Ha_{name}.pdb"))

        if pdb_file.endswith('.cif'):
            if not os.path.exists(in_pdb_file2):
                cmd = f"obabel -icif {in_pdb_file} -opdb -O {in_pdb_file2}"
                exec_command(cmd)
            in_pdb_file = in_pdb_file2

        if not os.path.exists(hr_pdb_file):
            cmd = f"reduce -Trim {in_pdb_file}"
            output = exec_command(cmd, skip_error=True)
            with open(hr_pdb_file, 'w') as f:
                f.write(output)

        if not os.path.exists(ha_pdb_file):
            cmd = f"reduce -HIS {hr_pdb_file}"
            output = exec_command(cmd, skip_error=True)
            with open(ha_pdb_file, 'w') as f:
                f.write(output)
        assert os.path.exists(ha_pdb_file)
        return ha_pdb_file


    def get_struct_by_seq(self, seq, pocket_type=None, top=None, add_hs=False):
        top = top or self.top
        pocket_type = pocket_type or self.pocket_type
        struct_type = self.struct_type
        if not hasattr(self, 'seq_struct_map'):
            with open(self.seq_struct_map_file) as f:
                self.seq_struct_map = json.load(f)

        struct_file = self.seq_struct_map[seq]['file']


        if pocket_type is not None:
            struct_files = []
            for i in range(top):
                if pocket_type=='dogsite3':
                    try:
                        name = os.path.basename(struct_file).split(".")[0].split('-')[1].lower()
                    except IndexError:
                        name = os.path.basename(struct_file).split(".")[0].split('-')[0].lower()
                    file = os.path.join(self.struct_root_dir, pocket_type, 'pockets',
                                               name, 'residues', f"{name}_P_{i + 1}_res.pdb")
                elif pocket_type=='fpocket':
                    name = os.path.basename(struct_file).split(".")[0]
                    struct_dir = os.path.join(self.struct_root_dir, pocket_type, f'{name}_out', 'full_pockets')
                    file = os.path.join(struct_dir, f'pocket{i+1}_atm.pdb')
                else:
                    raise NotImplementedError
                if os.path.exists(file):
                    struct_files.append(file)
        else:
            struct_files = [os.path.join(self.struct_root_dir, struct_file)]

        if add_hs:
            cache_dir = self.cache_dir if self.pocket_type is None else os.path.join(self.cache_dir, pocket_type, name)
            os.makedirs(cache_dir, exist_ok=True)
            struct_files = [self.protonate(item, cache_dir) for item in struct_files]
        assert len(struct_files)!=0
        return struct_files

    @classmethod
    def get_atom_coords(cls, chain):
        pt = Chem.GetPeriodicTable()
        coords = []
        atomz = []
        atom_radii = []
        atom_symbol = []
        for res in chain:
            if PDB.is_aa(res):
                atoms = res.get_atoms()
                for atom in atoms:
                    coords.append(atom.get_coord())
                    atomz.append(get_atomz(atom.element))
                    atom_radii.append(pt.GetRvdw(atomz[-1]))
                    atom_symbol.append(atom.element)
        coords = np.stack(coords)
        atomz = np.array(atomz)
        norm_atom_radii = np.array(atom_radii)/pt.GetRvdw(1)
        return {"coords": coords,
                "atomz": atomz,
                "norm_atom_radii":norm_atom_radii,
                "atom_symbol": atom_symbol}

    @classmethod
    def get_res_coords(cls, chain):
        seq = ""
        n_coords = []
        ca_coords = []
        c_coords = []
        o_coords = []
        cb_coords = []
        res_id = []
        for res in chain:
            if PDB.is_aa(res):
                resname = protein_letters_3to1[res.get_resname().capitalize()]
                seq += resname
                try:
                    n_coord = res["N"].get_coord()
                except KeyError:
                    n_coord = np.array([np.nan, np.nan, np.nan])
                try:
                    ca_coord = res["CA"].get_coord()
                except KeyError:
                    ca_coord = np.array([np.nan, np.nan, np.nan])
                try:
                    c_coord = res["C"].get_coord()
                except KeyError:
                    c_coord = np.array([np.nan, np.nan, np.nan])
                try:
                    o_coord = res["O"].get_coord()
                except KeyError:
                    o_coord = np.array([np.nan, np.nan, np.nan])
                try:
                    cb_coord = res["CB"].get_coord()
                except KeyError:
                    cb_coord = np.array([np.nan, np.nan, np.nan])
                res_id.append(res.id[1])
                n_coords.append(n_coord)
                ca_coords.append(ca_coord)
                c_coords.append(c_coord)
                o_coords.append(o_coord)
                cb_coords.append(cb_coord)
        res_id = np.array(res_id).astype(int)
        coords = np.stack([n_coords, ca_coords, c_coords, o_coords, cb_coords]).transpose((1, 0, 2))
        return {"coords": coords,
                "seq_pos_idx": res_id,
                "res_seq": seq}

    @classmethod
    def parse_struct(cls, file):
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




class PocketGVPFeaturizer(GVPFeaturizer):
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR, pocket_top=3, edge_num=30, rbf_num=16, struct_type='esmfold', pocket_type='fpocket', n_res_expand=10,
                 pretrained_model_name_or_path=None):
        assert pocket_top in range(1, 4)
        super(PocketGVPFeaturizer, self).__init__(edge_num=edge_num, rbf_num=rbf_num)
        self.struct_root_dir = struct_root_dir
        self.struct_type = struct_type
        self.pocket_type = pocket_type
        self.struct_loader = StructureLoader(struct_root_dir=struct_root_dir, struct_type=struct_type,)
        self.pocket_top = pocket_top
        self.n_res_expand = n_res_expand
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.llm_featurizer = None
        if pretrained_model_name_or_path is not None:
            self.llm_featurizer = LLMFeaturizer(pretrained_model_name_or_path=pretrained_model_name_or_path, feat_type='seq')

    def _featurize(self, datapoint: Any, **kwargs):
        n_res_expand = self.n_res_expand
        full_struct_file = self.struct_loader.get_struct_by_seq(datapoint)
        full_structure = StructureLoader.parse_struct(full_struct_file[0])
        full_chain = list(full_structure.get_chains())[0]
        struct_files = self.struct_loader.get_struct_by_seq(datapoint, pocket_type=self.pocket_type, top=self.pocket_top)
        pocket_ids = []
        for i, struct_file in enumerate(struct_files):
            structure = StructureLoader.parse_struct(struct_file)
            chains = list(structure.get_chains())
            assert len(chains) == 1
            chain = chains[0]
            for res in chain:
                ids = res.id[1]
                pocket_ids.extend(list(range(max(1, ids-n_res_expand), ids+n_res_expand+1)))
        pocket_ids = sorted(set(pocket_ids))
        new_chain = PDB.Chain.Chain('X')
        pocket_ids = [ids for ids in pocket_ids if ids in full_chain]
        for ids in pocket_ids:
            new_chain.add(full_chain[ids])
        ans = super(PocketGVPFeaturizer, self)._featurize([new_chain])
        if self.llm_featurizer is not None:
            full_res = self.struct_loader.get_res_coords(full_chain)
            pos_idx_map = np.ones(full_res['seq_pos_idx'].max()+1, dtype=int)*(full_res['seq_pos_idx'].max()+1)
            for i, idx in enumerate(full_res['seq_pos_idx']):
                pos_idx_map[idx] = i

            seq_ids = pos_idx_map[pocket_ids]
            full_feat = self.llm_featurizer.featurize([full_res['res_seq']])[0].graph.inputs_embeds
            node_feature = full_feat[seq_ids]
            assert self.struct_loader.get_res_coords(new_chain)['res_seq']==("".join(np.array(list(full_res['res_seq']))[seq_ids]))
            ans.graph.node_features = np.concatenate([ans.graph.node_features, node_feature], axis=-1)
        return ans


class SurfaceNormalFeaturizer(FeaturizerBase):
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR, struct_type='esmfold', top=1, pocket_type=None):
        assert top in range(1, 4)
        super(SurfaceNormalFeaturizer, self).__init__()
        self.struct_root_dir = struct_root_dir
        self.struct_type = struct_type
        self.pocket_type = pocket_type
        self.struct_loader = StructureLoader(struct_root_dir=struct_root_dir, struct_type=struct_type, pocket_type=pocket_type,
                                             top=top)
        self.top = top
        self.distance = 1.05
        self.smoothness = 0.5
        self.resolution = 1.0
        self.nits = 4
        self.sup_sampling = 20
        self.variance = 0.1
        self.curvature_scales = [1.0, 2.0, 3.0, 5.0, 10.0]
        self.disk_cache_dir = os.path.join(self.FEATURIZER_OUTPUT_TEMP_DIR, "SurfaceNormalFeaturizer", f"{struct_type}_{pocket_type}_{top}")
        os.makedirs(self.disk_cache_dir, exist_ok=True)

    @disk_cache
    def _featurize(self, datapoint, **kwargs):
        struct_files = self.struct_loader.get_struct_by_seq(datapoint, add_hs=True)
        data = []
        for i, struct_file in enumerate(struct_files):
            structure = self.struct_loader.parse_struct(struct_file)
            chains = list(structure.get_chains())
            if len(chains) > 1:
                print(chains, struct_file)
            chain = chains[0]
            atom_info = StructureLoader.get_atom_coords(chain)
            data.append(atom_info)
        atoms = np.concatenate([item['coords'] for item in data])
        batch_atoms = np.concatenate([np.ones(len(item['coords']), dtype=int)*i for i, item in enumerate(data)])
        norm_atoms_radii = np.concatenate([item['norm_atom_radii'] for item in data])

        xyz, normals, p_curvatures, batch = self.atoms_to_points_normals(atoms, batch_atoms, norm_atoms_radii)

        data = GraphData(pos=atoms, radii=norm_atoms_radii)
        normals_graph = GraphData(node_features=p_curvatures, pos=xyz, normals=normals)
        return FeatData(atom_graph=data, normal_graph=normals_graph)

    def atoms_to_points_normals(self, atoms, batch_atoms, atom_radii, ):
        atoms = torch.from_numpy(atoms).float()
        batch_atoms = torch.from_numpy(batch_atoms)
        atom_radii = torch.from_numpy(atom_radii).float()
        xyz, normals, batch = atoms_to_points_normals(
            atoms,
            batch_atoms,
            atomtype_radii=atom_radii,
            resolution=self.resolution,
            sup_sampling=self.sup_sampling,
            variance=self.variance,
            nits=self.nits,
            smoothness=self.smoothness,
            distance=self.distance,
        )
        P_curvatures = curvatures(xyz, triangles=None, normals=normals, scales=self.curvature_scales, batch=batch)
        assert P_curvatures.isnan().sum() == 0
        return xyz.numpy(), normals.numpy(), P_curvatures.numpy(), batch.numpy()


class PocketSurfaceNormalFeaturizer(SurfaceNormalFeaturizer):
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR, struct_type='esmfold', pocket_type='fpocket', pocket_top=3,
                 pocket_point_nums=512):
        assert pocket_top in range(1, 4)
        super(PocketSurfaceNormalFeaturizer, self).__init__(struct_root_dir, struct_type)
        self.struct_type = struct_type
        self.pocket_top = pocket_top
        self.pocket_point_nums = pocket_point_nums
        self.pocket_loader = StructureLoader(struct_root_dir=struct_root_dir, struct_type=struct_type,
                                             pocket_type=pocket_type, top=pocket_top)
        self.disk_cache_dir = os.path.join(self.FEATURIZER_OUTPUT_TEMP_DIR, "PocketSurfaceNormalFeaturizer",
                                           f"{struct_type}_{pocket_type}_{pocket_top}_{pocket_point_nums}")
        os.makedirs(self.disk_cache_dir, exist_ok=True)

    def get_pocket_centers(self, datapoint):
        pocket_files = self.pocket_loader.get_struct_by_seq(datapoint)
        pocket_centers = []
        for i, struct_file in enumerate(pocket_files):
            structure = StructureLoader.parse_struct(struct_file)
            chains = list(structure.get_chains())
            if len(chains) > 1:
                print(chains, struct_file)
            chain = chains[0]
            atom_info = StructureLoader.get_atom_coords(chain)
            pocket_centers.append(atom_info['coords'].mean(axis=0))
        return np.stack(pocket_centers)

    def select_by_pocket(self, xyz, pocket_centers, point_nums=512):
        from pykeops.numpy import Vi, Vj
        pocket_centers = Vi(pocket_centers)
        xyz = Vj(xyz)
        distance = (pocket_centers-xyz).square().sum(dim=-1)
        index = distance.argKmin(K=point_nums, dim=1)
        return index

    @disk_cache
    def _featurize(self, datapoint, **kwargs):
        pocket_centers = self.get_pocket_centers(datapoint)

        struct_files = self.struct_loader.get_struct_by_seq(datapoint, add_hs=True)
        structure = StructureLoader.parse_struct(struct_files[0])
        chains = list(structure.get_chains())
        if len(chains) > 1:
            print(chains, struct_files[0])
        chain = chains[0]
        atom_info = StructureLoader.get_atom_coords(chain)


        xyz, normals, p_curvatures, batch = self.atoms_to_points_normals(atom_info['coords'],
                                                                         np.zeros(len(atom_info['coords']), dtype=int),
                                                                         atom_info['norm_atom_radii'])
        index = self.select_by_pocket(xyz, pocket_centers, point_nums=self.pocket_point_nums)
        new_xyz = xyz[index].reshape(-1, xyz.shape[-1])
        new_normals = normals[index].reshape(-1, normals.shape[-1])
        new_curvatures = p_curvatures[index].reshape(-1, p_curvatures.shape[-1])
        pocket_batch = np.arange(index.shape[0])[:,None].repeat(index.shape[-1], axis=1).flatten()
        normal_graph = GraphData(pos=new_xyz, node_features=new_curvatures,
                                 normals=new_normals, node_batch=pocket_batch)
        data = GraphData(pos=atom_info['coords'], radii=atom_info['norm_atom_radii'])
        return FeatData(atom_graph=data, normal_graph=normal_graph, prefix='pocket_surface')


class PocketUnimolFeaturizer(FeaturizerBase):
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR, unimol_model_name='unimolv1', unimol_model_size='84M',
                 struct_type='esmfold', pocket_type='fpocket', pocket_top=3):
        super().__init__()
        self.struct_root_dir = struct_root_dir
        self.unimol_model_name = unimol_model_name
        self.unimol_model_size = unimol_model_size
        self.struct_type = struct_type
        self.pocket_type = pocket_type
        self.pocket_top = pocket_top
        self.pocket_loader = StructureLoader(struct_root_dir=struct_root_dir, struct_type=struct_type,
                                             pocket_type=pocket_type, top=pocket_top)

    def load_model(self):
        from toolbox.config import UNIMOL_WEIGHT_DIR
        from unimol_tools import UniMolRepr
        from unimol_tools.utils import logger
        import logging
        logger.setLevel(logging.ERROR)
        clf = UniMolRepr(data_type='protein', remove_hs=False)
        clf.params['max_atoms'] = 5000
        clf.params['multi_process'] = False
        return clf

    def _featurize(self, datapoint, **kwargs):
        struct_files = self.pocket_loader.get_struct_by_seq(datapoint)
        if not hasattr(self, 'model') or self.model is None:
            self.model = self.load_model()
        atoms = []
        coordinates = []
        for struct_file in struct_files:
            struct = self.pocket_loader.parse_struct(struct_file)
            chain = next(iter(struct.get_chains()))
            atom_info = self.pocket_loader.get_atom_coords(chain)
            assert len(atom_info['coords'])<2000
            atoms.append(atom_info['atom_symbol'])
            coordinates.append(atom_info['coords'])
        repr = self.model.get_repr({"atoms":atoms, "coordinates": coordinates}, return_atomic_reprs=True)
        for i in range(len(struct_files)):
            assert np.allclose(repr['atomic_coords'][i], coordinates[i]-coordinates[i].mean(axis=0, keepdims=True))
        node_features = np.concatenate(repr['atomic_reprs'], axis=0)
        node_positions = np.concatenate(coordinates, axis=0)
        cls_repr = np.stack(repr['cls_repr'], axis=0)
        node_batch = np.concatenate([np.ones(len(item))*i for i, item in enumerate(atoms)])
        graph = GraphData(node_features=node_features, node_batch=node_batch, pos=node_positions,
                          graph_embedding=cls_repr, graph_batch=np.arange(len(cls_repr)))
        return FeatData(atom_graph=graph, prefix='pocket_unimol')


class PocketSurfaceNormalUnimolFeaturizer(PocketSurfaceNormalFeaturizer):
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR, struct_type='esmfold', pocket_type='fpocket', pocket_top=3,
                 pocket_point_nums=512, unimol_model_name='unimolv1', unimol_model_size='84M'):
        super(PocketSurfaceNormalUnimolFeaturizer, self).__init__(struct_root_dir, struct_type, pocket_type, pocket_top, pocket_point_nums)
        self.unimol_model_name = unimol_model_name
        self.unimol_model_size = unimol_model_size
        self.nearest_atom_num = 16
        self.disk_cache_dir = os.path.join(self.FEATURIZER_OUTPUT_TEMP_DIR, "PocketSurfaceNormalUnimolFeaturizer",
                                           f"{struct_type}_{pocket_type}_{pocket_top}_{pocket_point_nums}_{unimol_model_name}_{unimol_model_size}")
        os.makedirs(self.disk_cache_dir, exist_ok=True)

    def load_model(self):
        from toolbox.config import UNIMOL_WEIGHT_DIR
        from unimol_tools import UniMolRepr
        from unimol_tools.utils import logger
        import logging
        logger.setLevel(logging.ERROR)
        clf = UniMolRepr(data_type='protein', remove_hs=False,
                         model_name=self.unimol_model_name,
                         model_size=self.unimol_model_size)
        clf.params['max_atoms'] = 5000
        clf.params['multi_process'] = False
        return clf

    @disk_cache
    def _featurize(self, datapoint, **kwargs):
        if not hasattr(self, 'model') or self.model is None:
            self.model = self.load_model()
        pocket_centers = self.get_pocket_centers(datapoint)

        struct_files = self.struct_loader.get_struct_by_seq(datapoint, add_hs=True)
        structure = StructureLoader.parse_struct(struct_files[0])
        chains = list(structure.get_chains())
        if len(chains) > 1:
            print(chains, struct_files[0])
        chain = chains[0]
        atom_info = StructureLoader.get_atom_coords(chain)
        xyz, normals, p_curvatures, batch = self.atoms_to_points_normals(atom_info['coords'], np.zeros(len(atom_info['coords']), dtype=int),
                                                                         atom_info['norm_atom_radii'])
        assert np.isnan(p_curvatures).sum()==0
        index = self.select_by_pocket(xyz, pocket_centers, point_nums=self.pocket_point_nums)
        new_xyz = xyz[index]
        new_normals = normals[index]
        new_curvatures = p_curvatures[index]


        atom_index = self.select_by_pocket(atom_info['coords'], new_xyz.reshape(-1, xyz.shape[-1]),
                                           point_nums=self.nearest_atom_num)

        idx, rev_idx = np.unique(atom_index, return_inverse=True)
        repr_atoms = np.array(atom_info['atom_symbol'])[idx]
        repr_coords = atom_info['coords'][idx]
        repr = self.model.get_repr({"atoms": [repr_atoms], "coordinates": [repr_coords]}, return_atomic_reprs=True)

        point_feature = np.array((repr['atomic_reprs'][0]))[rev_idx.reshape(atom_index.shape)]
        new_feat = point_feature.mean(axis=1).reshape(*new_xyz.shape[:2], -1)


        assert np.isnan(point_feature).sum()==0

        # coords_to_pdb(xyz, "xyz.pdb")
        # coords_to_pdb(new_xyz[:,0], "new_xyz.pdb")
        node_features = np.concatenate([new_curvatures, new_feat], axis=-1)
        pocket_batch = np.arange(index.shape[0])[:,None].repeat(index.shape[-1], axis=1).flatten()
        normal_graph = GraphData(pos=new_xyz.reshape(-1, new_xyz.shape[-1]),
                                 node_features=node_features.reshape(-1, node_features.shape[-1]),
                                 node_curvatures_feat=new_curvatures.reshape(-1, new_curvatures.shape[-1]),
                                 node_unimol_feat=new_feat.reshape(-1, new_feat.shape[-1]),
                                 node_normals=new_normals.reshape(-1, new_normals.shape[-1]),
                                 node_batch=pocket_batch)
        data = GraphData(pos=atom_info['coords'],
                         radii=atom_info['norm_atom_radii'])
        return FeatData(atom_graph=data, normal_graph=normal_graph, prefix='pocket_surface')



def coords_to_pdb(coords, out_pdb="points.pdb"):
    with open(out_pdb, "w") as f:
        for i, (x,y,z) in enumerate(coords, 1):
            f.write(f"HETATM{i:5d}  X   UNK A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n")
        f.write("END\n")

