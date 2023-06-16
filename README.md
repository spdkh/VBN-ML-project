# VBN ML Project

## Initiate and activate the environment: 

    conda create -n "vbn_project" python=3.8.5
    conda activate vbn_project
    cd /path/to/VBN-ML-project
    pip install requirements.txt # I just like pip for this more :D

## Train with 100 epochs:

    python -m src.train --iteration 100 --data_dir /path/to/dataset --dataset <dataset name>

for more configuration parameters checkout src/utils/config.py or type:

    python -m src.train --help

## Tensorboard Logs
### see all tensorboard logs together:

    tensorboard --logdir="results"

### see specific tensorboard logs separately:

You can add as many folders as you wish

    tensorboard --logdir_spec proj1:results/proj1,proj2:results/proj2
