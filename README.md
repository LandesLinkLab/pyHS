# pyHS
Python-based Hyperspectral Analysis

conda create -n py-hs python=3.11\
conda activate py-hs\
conda install -n py-hs conda-forge::numpy\
conda install -n py-hs anaconda::scipy\
conda install -n py-hs conda-forge::matplotlib\
conda install -n py-hs anaconda::scikit-image\
conda install -n py-hs conda-forge::nptdms\
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip uninstall triton -y
pip install triton==3.1.0