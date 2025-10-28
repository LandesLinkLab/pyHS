# pyHS
Python-based Hyperspectral Analysis

conda create -n py-hs python=3.11\
conda activate py-hs\
conda install -n py-hs conda-forge::numpy\
conda install -n py-hs anaconda::scipy\
conda install -n py-hs conda-forge::matplotlib\
conda install -n py-hs anaconda::scikit-image\
conda install -n py-hs conda-forge::nptdms\
conda install -n vitamin pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
