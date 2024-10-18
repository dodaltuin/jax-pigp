# Physics-informed Gaussian processes in JAX

Functionality for implementing physics-informed Gaussian processes (PIGPs) in Python using [JAX](https://github.com/google/jax). Source code is in 
[``jax-pigp``](jax_pigp), tests are in [``tests``](tests), while [``examples``](examples) provides numerical examples involving different partial differential equations (PDEs).


## Installation
The code can be downloaded and necessary packages installed with the following shell commands:

```
git clone https://github.com/dodaltuin/jax-pigp.git
cd jax-pigp
pip install "jax[XXX]"
pip install -r requirements.txt
```

Comments: 

* ``XXX`` will change depending on your machine and accelerator - we used ``XXX = cuda12`` . For more details, see the JAX installation guide [here](https://jax.readthedocs.io/en/latest/installation.html).
* The code was tested using Python version 3.12.3 on the [Rocky Linux](https://rockylinux.org/) distribution (version 8.8).
* To prevent conflicts, you may want to create a [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html) (using for example [Anaconda](https://www.anaconda.com/download)).

## Examples

### [``examples/BCGPs``](examples/BCGPs)

Experiments from [our JMLR paper](https://jmlr.org/papers/v25/23-1508.html)

### [``examples/PSGPs``](examples/PSGPs)

Experiments from [our ICML paper](https://openreview.net/forum?id=1V50J0emll)

## Tests

Tests can be fun as follows:

```
pytest tests -v
```

## Reference

If you use this code, please the following papers:

```
@article{dalton2024boundary,
  title={Boundary constrained Gaussian processes for robust physics-informed machine learning of linear partial differential equations},
  author={Dalton, David and Lazarus, Alan and Gao, Hao and Husmeier, Dirk},
  journal={Journal of Machine Learning Research},
  volume={25},
  number={272},
  pages={1--61},
  year={2024}
}
```

```
@inproceedings{dalton2024symmetry,
  title={Physics and Lie symmetry informed Gaussian processes},
  author={Dalton, David and Husmeier, Dirk and Gao, Hao},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```
