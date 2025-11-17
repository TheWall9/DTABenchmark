
# Requirements
``` bash
conda create -n py10 python=3.10
conda activate py10

conda install libstdcxx-ng>=14 -c conda-forge
conda install fpocket reduce -c bioconda
conda install openbabel -c conda-forge

pip install torch torchvision
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
pip install pykeops unimol_tools 
pip install mamba-ssm[causal-conv1d]
pip install transformers datasets tokenizers torchmetrics lightning tensorboard wandb optuna-integration[pytorch_lightning]
pip install deepchem biopython rdkit xlrd openpyxl matplotlib seaborn
```
# Datasets
davis & kiba: https://drive.google.com/file/d/1U8OTSXGrKMTd-tuIF3PmFnUp5HuWcCR8/view?usp=drive_link

# Usage
```bash
python main.py --dataset_name davis --max_epochs=700 --cv_split_type=predetermined
```