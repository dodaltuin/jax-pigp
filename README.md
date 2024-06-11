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

### [``examples/PSGPs``](examples/PSGPs)

Experiments from [our ICML paper](https://openreview.net/forum?id=1V50J0emll).

## Tests

Tests can be fun as follows:

```
pytest tests -v
```

## Reference

If you use this code, please the following paper:

```
@inproceedings{dalton2024symmetry,
  title={Physics and Lie symmetry informed Gaussian processes},
  author={Dalton, David and Husmeier, Dirk and Gao, Hao},
  booktitle={International Conference on Machine Learning},
  year={2024},
  organization={PMLR}
}
```
