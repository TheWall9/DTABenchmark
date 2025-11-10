
import numpy as np
from rdkit.Chem import ValenceType
from torch_geometric.loader import DataLoader

from deepchem.feat.graph_features import one_of_k_encoding_unk, one_of_k_encoding


from toolbox import datamodule
from toolbox.featurizer.ligand import MolFeaturizerBase
from toolbox.featurizer.protein import SimpleProtTokenFeaturizer


class GraphDTAMolFeaturizer(MolFeaturizerBase):
    def __init__(self, use_original_atoms_order=False):
        super().__init__(use_original_atoms_order)

    def get_atom_features_fn(self, atom, **kwargs):
        feat = np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                              ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
                                               'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                               'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                               'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        one_of_k_encoding_unk(atom.GetValence(ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                        [atom.GetIsAromatic()])
        return feat / sum(feat)


class GraphDTADataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = GraphDTAMolFeaturizer
    protein_featurizer_cls = SimpleProtTokenFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.dataloader_cls = DataLoader
        self.to_pandas()
        self.to_tensor()

    # @classmethod
    # def preprocess_affinities(cls, ligands, proteins, affinities, config):
    #     ligands_key_map = {"ligand_graph": "ligand_graph"}
    #     proteins_key_map = {"protein_input_ids": "protein_input_ids"}
    #     affinities = cls.merge_datasets(ligands, proteins, affinities, ligands_key_map, proteins_key_map)
    #     affinities = affinities.to_pandas()
    #     return affinities, {}

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



# class GraphDataset():
#     def __init__(self, dataset):
#         self.dataset = dataset




if __name__=="__main__":
    from toolbox.datamodule import DataModule
    from tqdm import tqdm
    import argparse
    dataset_cls = GraphDTADataset
    parser = argparse.ArgumentParser()
    DataModule.add_parser_arguments(parser)
    dataset_cls.add_parser_arguments(parser)
    args = parser.parse_args()
    datamodule = DataModule(vars(args), dataset_cls=dataset_cls)
    datamodule.prepare_data()
    datamodule.setup()
    for batch in tqdm(datamodule.train_dataloader()):
        pass

    for dataset in datamodule.split_datasets():
        for batch in tqdm(dataset.train_dataloader()):
            pass
    """4s"""