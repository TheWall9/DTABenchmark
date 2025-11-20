
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from deepchem.feat.graph_features import one_of_k_encoding_unk, one_of_k_encoding


# from toolbox import datamodule
from toolbox.datamodule import dataset as datamodule
from toolbox.featurizer.ligand import GraphMVPFeaturizer, MorganFeaturizer
from toolbox.featurizer.protein import LLMFeaturizer, PocketGVPFeaturizer
from toolbox.featurizer.tools import FeaturizerBase, SmilesFeaturizerBase, FeatData
from toolbox.config import STRUCT_ROOT_DIR


class PocketDTAMolFeaturizer(SmilesFeaturizerBase):
    def __init__(self, use_original_atoms_order=False, ligand_max_seq_length=40):
        super().__init__(use_original_atoms_order)
        self.seq_featurizer = MorganFeaturizer(use_original_atoms_order)
        self.struct_featurizer = GraphMVPFeaturizer(use_original_atoms_order=use_original_atoms_order,
                                                    ligand_max_seq_length=ligand_max_seq_length)
        self.ligand_max_seq_length = ligand_max_seq_length

    def _featurize(self, datapoint, **kwargs):
        morgan = self.seq_featurizer.featurize([datapoint], **kwargs)[0]
        struct = self.struct_featurizer.featurize([datapoint], **kwargs)[0]
        return FeatData(embedding=morgan.embedding, inputs_embeds=struct.inputs_embeds, prefix='ligand')


class PocketDTAProtFeaturizer(FeaturizerBase):
    def __init__(self, struct_root_dir=STRUCT_ROOT_DIR, pocket_type='dogsite3', top=3):
        super().__init__()
        self.seq_featurizer = LLMFeaturizer(feat_type='mean')
        self.struct_featurizer = PocketGVPFeaturizer(struct_root_dir, pocket_top=top, pocket_type=pocket_type)
        self.struct_root_dir = struct_root_dir
        self.pocket_type = pocket_type
        self.top = top

    def _featurize(self, datapoint, **kwargs):
        embedding = self.seq_featurizer.featurize([datapoint], **kwargs)[0]
        graph = self.struct_featurizer.featurize([datapoint], **kwargs)[0]
        return FeatData(embedding=embedding.embedding, graph=graph.graph, prefix='protein')



class PocketDTADataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = PocketDTAMolFeaturizer
    protein_featurizer_cls = PocketDTAProtFeaturizer

    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.dataloader_cls = DataLoader
        self.to_pandas()
        self.to_tensor()

    @classmethod
    def add_parser_arguments(cls, parser):
        cls.ligand_featurizer_cls.add_argparser_args(parser)
        cls.protein_featurizer_cls.add_argparser_args(parser)
        parser.set_defaults(batch_size=32)

class PocketDTADataset2(PocketDTADataset):
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        input_columns = input_columns if input_columns is None else input_columns + ['Uniprot_ID']
        super().__init__(proteins, ligands, affinities, input_columns)
        if not hasattr(self.proteins['protein_graph'], 'x'):
            data_dir = "/mnt/data/lcc/WorkSpace/DTA_Space/Benchmark/data/davis_pocketdta"
            import os
            import pickle
            with open(os.path.join(data_dir, "ligand_features.pickle"), 'rb') as f:
                ligand_features = pickle.load(f)
            with open(os.path.join(data_dir, "protein_features.pickle"), 'rb') as f:
                protein_features = pickle.load(f)
            self.proteins['protein_graph'] = protein_features['protein_graph']
            self.proteins['protein_embedding'] = protein_features['protein_embedding']
            self.ligands['ligand_embedding'] = ligand_features['ligand_embedding']
            self.ligands['ligand_inputs_embeds'] = ligand_features['ligand_inputs_embeds']

if __name__=="__main__":
    from toolbox.datamodule import DataModule
    from tqdm import tqdm
    import argparse
    dataset_cls = PocketDTADataset2
    parser = argparse.ArgumentParser()
    DataModule.add_parser_arguments(parser)
    dataset_cls.add_parser_arguments(parser)
    args = parser.parse_args()
    args.dataset_name = 'davis_pocketdta'
    args.cv_split_type = 'cv'
    datamodule = DataModule(vars(args), dataset_cls=dataset_cls)
    # datamodule.input_columns = ["ligand_embedding", "ligand_inputs_embeds", "protein_embedding", 'protein_graph', 'affinity']
    # datamodule.preprocessed_data = None
    datamodule.prepare_data()
    datamodule.setup()

    affinity = datamodule.preprocessed_data.affinities
    ligand = datamodule.preprocessed_data.ligands
    proteins = datamodule.preprocessed_data.proteins
    # affinity['LID'] = affinity['LID'].astype(int)

    # import datasets
    # d = datasets.Dataset.from_pandas(proteins)
    # d.set_format("torch")
    # s = d[[1,2,3]]
    # datamodule.preprocessed_data[[1,2,3]]


    # for item in tqdm(datamodule.preprocessed_data):
    #     f = item #180it/s

    # datamodule.preprocessed_data.to_list() # 3.19it/s
    # datamodule.preprocessed_data.to_pandas() # 6.9it/s
    # datamodule.preprocessed_data.to_pandas() # 25.40it/s

    # datamodule.preprocessed_data.save_to_disk("debug", type='torch')
    # datamodule.preprocessed_data.load_from_disk("debug", type='torch')
    # datamodule.preprocessed_data.save_pt_to_disk("debug")
    # datamodule.preprocessed_data.load_pt_from_disk("debug")

    # datamodule.preprocessed_data.to_tensor()
    # datamodule.preprocessed_data.to_pandas()

    # for item in tqdm(datamodule.preprocessed_data):
    #     f = item #180it/s

    for batch in tqdm(datamodule.train_dataloader()):
        pass



    # for idx, row in tqdm(affinity.iterrows(), total=len(affinity)):
    #     pid = int(row['PID'])
    #     lid = int(row['LID'])
    #     f = ligand.iloc[lid]
    #     s = proteins.iloc[pid]
        #  14847.09it/s
        # ligand_embedding = torch.from_numpy(f['ligand_embedding'])
        # 10856.17it/s
        # ligand_inputs_embeds = torch.from_numpy(np.stack(f['ligand_inputs_embeds']))
        # 8981.35it/s
        # protein_embedding = torch.from_numpy(s['protein_embedding'])
        # 2525.48it
        # datamodule.preprocessed_data.to_pgy_data(s[''])
        # protein_graph = s['protein_graph']
        # ans = {key:np.stack(value) for key, value in protein_graph.items()}



    # ligand_embedding = ligand['ligand_embedding'].values
    # ligand_inputs_embeds = ligand['ligand_inputs_embeds'].values
    #
    # index = np.random.randint(0, 68, size=30056)
    # for i in tqdm(index):
    #     # 1820445.08it/s
    #     torch.from_numpy(ligand_embedding[i])
    #
    # for i in tqdm(index):
    #     # 51814.18it/s
    #     torch.from_numpy(np.stack(ligand_inputs_embeds[i]))
    #
    # data = np.random.randn(442, 100, 300)
    # index = np.random.randint(0, 441, size=30056)
    #
    # for i in tqdm(index):
    #     # 1590733.02it/s
    #     f = torch.from_numpy(data[i])
    #
    # for i in tqdm(range(30056)):
    #     # 3467.67it/s
    #     d = np.random.randn(100, 300)
    #     f = torch.from_numpy(d)
    #
    # for i in tqdm(range(30056)):
    #     # 14289.93it/s
    #     f = torch.rand((100, 300))

    for batch in tqdm(datamodule.train_dataloader()):
        pass

    for dataset in datamodule.split_datasets():
        for batch in tqdm(dataset.train_dataloader()):
            pass


    # datasets.Dataset.from_dict()
