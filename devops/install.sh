conda create -n selective_annotation python=3.9 -y
conda activate selective_annotation

# conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl -y

pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
# pipreqs ./ --force --encoding=utf8 --ignore=transformers
pip install -r requirements.txt
pip uninstall transformers -y
pip install -e ./transformers
