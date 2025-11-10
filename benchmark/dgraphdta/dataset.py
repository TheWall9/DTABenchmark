
import numpy as np


from torch_geometric.loader import DataLoader

from deepchem import feat
from deepchem.feat.graph_features import one_of_k_encoding


from toolbox import datamodule, FeaturizerBase, FeatData, GraphData
from toolbox.config import HHSUITE_DB_PATH, STRUCT_ROOT_DIR
from toolbox.featurizer.protein import ContactMapFeaturizer, ProtPssmFeaturizer
from benchmark.graphdta.dataset import GraphDTAMolFeaturizer


def dic_normalize(dic):
    # print(dic)
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    # print(max_value)
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic


pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

pro_res_aliphatic_table = ['A', 'I', 'L', 'M', 'V']
pro_res_aromatic_table = ['F', 'W', 'Y']
pro_res_polar_neutral_table = ['C', 'N', 'Q', 'S', 'T']
pro_res_acidic_charged_table = ['D', 'E']
pro_res_basic_charged_table = ['H', 'K', 'R']


res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_pl_table = {'A': 6.00, 'C': 5.07, 'D': 2.77, 'E': 3.22, 'F': 5.48, 'G': 5.97, 'H': 7.59,
                'I': 6.02, 'K': 9.74, 'L': 5.98, 'M': 5.74, 'N': 5.41, 'P': 6.30, 'Q': 5.65,
                'R': 10.76, 'S': 5.68, 'T': 5.60, 'V': 5.96, 'W': 5.89, 'Y': 5.96}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}

res_weight_table = dic_normalize(res_weight_table)
res_pka_table = dic_normalize(res_pka_table)
res_pkb_table = dic_normalize(res_pkb_table)
res_pkx_table = dic_normalize(res_pkx_table)
res_pl_table = dic_normalize(res_pl_table)
res_hydrophobic_ph2_table = dic_normalize(res_hydrophobic_ph2_table)
res_hydrophobic_ph7_table = dic_normalize(res_hydrophobic_ph7_table)


def residue_features(residue):
    res_property1 = [1 if residue in pro_res_aliphatic_table else 0, 1 if residue in pro_res_aromatic_table else 0,
                     1 if residue in pro_res_polar_neutral_table else 0,
                     1 if residue in pro_res_acidic_charged_table else 0,
                     1 if residue in pro_res_basic_charged_table else 0]
    res_property2 = [res_weight_table[residue], res_pka_table[residue], res_pkb_table[residue], res_pkx_table[residue],
                     res_pl_table[residue], res_hydrophobic_ph2_table[residue], res_hydrophobic_ph7_table[residue]]
    return np.array(res_property1 + res_property2)



class DGraphDTAProtFeaturizer(FeaturizerBase):
    def __init__(self, hhsuite_db_path=HHSUITE_DB_PATH, struct_root_dir=STRUCT_ROOT_DIR, distance_cutoff=8.0, coord_type='cb'):
        super(DGraphDTAProtFeaturizer, self).__init__()
        self.graph_featurizer = ContactMapFeaturizer(struct_root_dir, distance_cutoff, coord_type)
        self.pssm_featurizer = ProtPssmFeaturizer(hhsuite_db_path, use_ppm=True)
        self.hhsuite_db_path = hhsuite_db_path
        self.struct_root_dir = struct_root_dir
        self.distance_cutoff = distance_cutoff
        self.coord_type = coord_type

    def _featurize(self, datapoint, **kwargs):
        data = self.graph_featurizer.featurize([datapoint], **kwargs)[0]
        seq = data.graph.seq
        edge_index = data.graph.edge_index
        pssm_feat = self.pssm_featurizer.featurize([seq])[0]
        res_feat = np.stack([residue_features(res) for res in seq])
        res_coding = np.stack([one_of_k_encoding(res, pro_res_table) for res in seq])
        features = np.concatenate([pssm_feat, res_coding, res_feat, ], axis=-1)
        return FeatData(graph=GraphData(node_features=features, edge_index=edge_index), prefix='protein')


class DGraphDTADataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = GraphDTAMolFeaturizer
    protein_featurizer_cls = DGraphDTAProtFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.dataloader_cls = DataLoader
        self.to_pandas()
        self.to_tensor()

    @classmethod
    def add_parser_arguments(cls, parser):
        cls.ligand_featurizer_cls.add_argparser_args(parser)
        cls.protein_featurizer_cls.add_argparser_args(parser)
        parser.set_defaults(batch_size=512)

    # def __getitem__(self, idx):
    #     affinity = self.select_affinity(idx)
    #     affinity = self.to_pgy_data(affinity)
    #     if self.input_columns is not None:
    #         affinity = {key: affinity[key] for key in self.input_columns}
    #     return affinity


if __name__ == '__main__':
    from toolbox.datamodule import DataModule
    from tqdm import tqdm
    import argparse
    dataset_cls = DGraphDTADataset
    parser = argparse.ArgumentParser()
    DataModule.add_parser_arguments(parser)
    dataset_cls.add_parser_arguments(parser)
    args = parser.parse_args()
    args.hhsuite_db_path = "/mnt/data/lcc/Datasets/Uniclust/uniclust30_2018_08/uniclust30_2018_08"
    args.struct_root_dir = "../../data/uniprot_alphafold_struct"
    datamodule = DataModule(vars(args), dataset_cls=dataset_cls)
    datamodule.prepare_data()
    datamodule.setup()
    for batch in tqdm(datamodule.train_dataloader()):
        pass

    for dataset in datamodule.split_datasets():
        for batch in tqdm(dataset.train_dataloader()):
            pass

    """27s"""

