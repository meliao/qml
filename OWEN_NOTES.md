# Setup conda environment

```
conda create --prefix /local/meliao/conda_envs/FCHL python=3.10 numpy scipy six scikit-learn -c conda-forge
conda activate /local/meliao/conda_envs/FCHL/
```

# Install from source
```
pip install git+https://github.com/qmlcode/qml@develop --user -U
```


# Running

For some reason I **must** run in this directory: `/local/meliao/projects/new_qml/`



# Re-Doing this on the TTIC core

## Set up conda environment

```
conda create --prefix /scratch/meliao/FCHL_conda_env python=3.10 numpy scipy six scikit-learn -c conda-forge
conda activate /scratch/meliao/FCHL_conda_env
```

## Install qml from source
```
pip install git+https://github.com/qmlcode/qml@develop --user -U
```

I was not able to build qml after looking up error messages for an hour. The output of the attempted install is in `TTIC_install_output.out`