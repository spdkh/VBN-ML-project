# VBN ML Project

To change the deep architecture in this project,
define your architecture in src/utils/architectures.
The function should take net_input as a tensorflow Input object.
Then return net_output as a tensorflow object. 
In src/models/vbnnet.py import your function and change the architecture used in
line 34 as self.outupt to your function given the tensorflow input 
and other parameters if needed.
I will put instructions on designing a GAN later.

## Initiate and activate the environment: 

        conda create -n "vbn_project" python=3.8.5
        conda activate vbn_project
        cd /path/to/VBN-ML-project

- If you are using GPU start here:
  
  Windows:

        conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

  Linux:
  
        conda install -c conda-forge cudatoolkit=11.8.0
        python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.10.0
        mkdir -p $CONDA_PREFIX/etc/conda/activate.d
        echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
        echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
        source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

- GPU/CPU continue here:
  
        pip install -r requirements.txt # pip should be used for tensorflow


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
