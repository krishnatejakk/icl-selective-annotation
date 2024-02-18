conda create -n selective_annotation python=3.9 -y
conda activate selective_annotation

# conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl -y

pip install -r requirements-torch.txt --index-url https://download.pytorch.org/whl/cu121
# pipreqs ./ --force --encoding=utf8 --ignore=transformers
pip install -r requirements.txt
pip uninstall transformers -y
pip install -e ./transformers
sudo apt-get update && sudo apt-get install g++ -y
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib
