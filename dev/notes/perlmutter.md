Setup in `~/.bash_profile`:
```
module load cpe/24.07
module load python/3.11
module load cudatoolkit/12.9
conda activate hpm
```

Building the conda environment:
```
module load python
conda create -n hpm python=3.11 pip
conda activate hpm
pip install --upgrade jax[cuda12]
pip install -e .
python -m ipykernel install --user --name hpm --display-name hpm
```

Setup in `~/.local/share/jupyter/kernels/hpm/kernel-helper.sh`
```
#! /bin/bash
module load cpe/24.07
module load python/3.11
module load cudatoolkit/12.9
exec "$@"
```

Setup in `~/.local/share/jupyter/kernels/hpm/kernel.json`
```
{
 "argv": [
  "{resource_dir}/kernel-helper.sh",
  "/global/common/software/des/athomsen/hpm/bin/python",
  "-Xfrozen_modules=off",
  "-m",
  "ipykernel_launcher",
  "-f",
  "{connection_file}"
 ],
 "display_name": "hpm",
 "language": "python",
 "metadata": {
  "debugger": true
 }
}
```