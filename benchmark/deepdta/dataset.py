
from toolbox import datamodule
from toolbox.featurizer.ligand import SimpleSmilesTokenFeaturizer
from toolbox.featurizer.protein import SimpleProtTokenFeaturizer


class DeepDTADataset(datamodule.DTADatasetBase):
    ligand_featurizer_cls = SimpleSmilesTokenFeaturizer
    protein_featurizer_cls = SimpleProtTokenFeaturizer
    def __init__(self, proteins, ligands, affinities, input_columns=None):
        super().__init__(proteins, ligands, affinities, input_columns=input_columns)
        self.to_pandas()
        self.to_tensor()


if __name__ == '__main__':
    from toolbox.datamodule import DataModule
    from tqdm import tqdm
    import argparse

    dataset_cls = DeepDTADataset
    parser = argparse.ArgumentParser()
    DataModule.add_parser_arguments(parser)
    dataset_cls.add_parser_arguments(parser)
    args = parser.parse_args()
    args.dataset_name = 'kiba'
    args.dataset_name = 'kiba_pocketdta'
    datamodule = DataModule(vars(args), dataset_cls=dataset_cls)
    datamodule.prepare_data()
    datamodule.setup()
    # data = datamodule.train_dataloader().dataset
    # data.affinities.data.combine_chunks()
    # data.input_columns = ['PID', 'ligand_input_ids']
    # import datasets
    # datasets.Dataset.to

    for batch in tqdm(datamodule.train_dataloader()):
        pass

    for dataset in datamodule.split_datasets():
        for batch in tqdm(dataset.train_dataloader()):
            pass