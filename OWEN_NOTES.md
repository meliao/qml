# Setup conda environment

```
conda create --prefix /local/meliao/conda_envs/FCHL python=3.10 numpy scipy six scikit-learn ase -c conda-forge
conda activate /local/meliao/conda_envs/FCHL/
```

# Install from source
```
pip install git+https://github.com/qmlcode/qml@develop --user -U
```


# Running

For some reason I **must** run in this directory: `/local/meliao/projects/new_qml/`

