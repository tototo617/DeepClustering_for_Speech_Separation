conda create -n DC python=3.8.3
conda activate DC
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch # v1.6.0
conda install -c conda-forge librosa # 0.8.0
conda install -c anaconda pyyaml # 5.3.1
conda install -c conda-forge tqdm # 4.50.2
conda install -c conda-forge tensorboard # 2.3.0
conda install -c conda-forge tensorboardx # 2.1