# pyHS
Python-based Hyperspectral Analysis

conda create -n py-hs python=3.11
conda activate py-hs
conda install -n py-hs pytorch=2.1.2 torchvision=0.16.2 torchaudio=2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -n py-hs -c anaconda matplotlib=3.8.0 jupyter
conda install -n py-hs -c conda-forge wandb=0.16.5
conda install -n py-hs -c conda-forge statsmodels
conda install -n py-hs scikit-learn
conda install -n py-hs -c conda-forge xgboost shap
conda install -n py-hs -c conda-forge captum

## Jupyter notebook server setting
https://velog.io/@y2k4388/Anaconda%EC%97%90%EC%84%9C-Jupyter-Notebook-%EC%84%A4%EC%B9%98
