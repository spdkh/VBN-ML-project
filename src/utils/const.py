"""
    Author: SPDKH
    Date: Spring 1400
"""
from pathlib import Path

from src.utils.config import parse_args

args = parse_args()

# the path corresponding to current file directory
THIS_DIR = Path(__file__).__str__()
# the path corresponding to source path
SRC_DIR = Path(THIS_DIR).parents[1]
PROJ_DIR = Path(THIS_DIR).parents[2]
CONFIG_DIR = SRC_DIR / 'config'
OUT_DIR = PROJ_DIR / Path(args.result_dir)
DATA_DIR = Path(args.data_dir)

CHK_FOLDER = '_'.join([args.dataset,
                       args.dnn_type,
                       args.log_name])

WEIGHTS_DIR = OUT_DIR / CHK_FOLDER

SAMPLE_DIR = WEIGHTS_DIR / 'sampled_img'

LOG_DIR = OUT_DIR / 'graph' / WEIGHTS_DIR
