import os

# os.environ["HTTP_PROXY"] = "http://100.118.242.88:4780"
# os.environ["HTTPS_PROXY"] = "socks5://100.118.242.88:4780"

ROOT_TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cached")
EVALUATION_TEMP_DIR = os.path.join(ROOT_TEMP_DIR, "evaluation")
DATASET_TEMP_DIR = os.path.join(ROOT_TEMP_DIR, "dataset")
LIGHTNING_LOGS_DIR = os.path.join(ROOT_TEMP_DIR, "lightning_logs")
FEATURIZER_INPUT_TEMP_DIR = os.path.join(ROOT_TEMP_DIR, "featurizer", "inputs")
FEATURIZER_OUTPUT_TEMP_DIR = os.path.join(ROOT_TEMP_DIR, "featurizer", "outputs")


HHSUITE_DB_PATH = "/mnt/data/datasets/Uniclust/uniclust30_2018_08/uniclust30_2018_08"
STRUCT_ROOT_DIR = os.path.join(os.path.dirname(ROOT_TEMP_DIR), 'data', 'uniprot_alphafold_struct')
THIRD_PARTY_CHECKPOINTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", 'checkpoints')

UNIMOL_WEIGHT_DIR = os.path.join(THIRD_PARTY_CHECKPOINTS_DIR, 'unimol_weights')
if not os.environ.get("UNIMOL_WEIGHT_DIR", False):
    os.environ['UNIMOL_WEIGHT_DIR'] = UNIMOL_WEIGHT_DIR