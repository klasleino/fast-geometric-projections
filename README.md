# fast-geometric-projections
Tool for certifying local robustness in deep networks.

This tool implements the algorithm described in the work, [Fast Geometric Projections for Local Robustness Certification](https://arxiv.org/pdf/2002.04742.pdf), appearing in ICLR 2021.

If you use this tool, please use the following citation:
```bibtex
@INPROCEEDINGS{fromherz20projections,
  title={Fast Geometric Projections for Local Robustness Certification},
  author={Aymeric Fromherz and Klas Leino and Matt Fredrikson and Bryan Parno and Corina Păsăreanu},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021},
}
```

## Installation

Simply install via pip:
```
pip install fgp-cert
```

Alternatively, you can install from the source repository:

1. Clone the [repository](https://github.com/klasleino/fast-geometric-projections) and change into its root directory.

2. Install from source via
```
pip install -e .
```

## Checking Robustness

The main routine for checking robustness is the `check` function in `fgp.certification`. This takes the following arguments:
* `network` : the neural network to be certified. Must be an instance of `CheckableModel` (found in the `fgp.checkable_models` module).
* `x` : a `numpy.Array` containing the instance to check robustness for.
* `epsilon` : the value of &epsilon; to check robustness for.
* `timeout` : (optional) the number of seconds to compute for before returning a `TIMED_OUT` result. If `None`, the computation will continue until the search has completed. NOTE: the time-out functionality will not work on Windows, since it uses signals. We recommend running on a unix machine.
* `lowerbound` : (default False) boolean flag specifying that we would like to use the certified lower bound algorithm (presented in Section 2.3) rather than the standard FGP algorithm (presented in Section 2.1).
* `keepgoing` : (default False) when True, uses the heuristic given in Section 2.2 for decreasing the number of `UNKNOWN` results by continuing to search the queue when a possible false-positive is found. This heuristic can only be used when `lowerbound` is `False`.
* `batch_size` : (default 1) batch size for processing the queue. Setting to 10-100 for medium to large examples significantly speeds up the FGP algorithm; however, for very large networks, this becomes very memory-intensive. The batch size must be 1 when `lowerbound` is `True`.
* `return_num_visited` : (optional) if set to `True`, the algorithm will return a tuple containing the robustness result followed by the number of regions visited. Otherwise it will return the robustness result only.
* `recap` : (default False) if set to `True`, the algorithm will print out a few statistics as it computes.
* `debug_steps` : (default False) if set to `True`, the algorithm will print out more detailed statistics as it computes.
* `debug_print_rate` : (default 1) if `debug_steps` is True, detailed information will be printed every `debug_print_rate` iterations of the algorithm.

### The `CheckableModel` Wrapper

The functionality required to compute projections to internal and decision boundaries is contained in the wrapper class, `CheckableModel`. 
The `check` routine only functions on instances of `CheckableModel`.
The constructor for `Checkable` model takes an input shape, a list of internal layer widths, and a number of output classes.
For example, to create a dense network on the MNIST dataset with three hidden layers of 20 neurons each, the following would be used:
```python
from fgp import CheckableModel

network = CheckableModel((784,), [20, 20, 20], 10)
```

Before running the `check` function on a model, the graph for computing boundaries and projections (for the desired norm) must be compiled.
This can be done using the following:
```python
# Defaults to the L2 norm.
network.compile_backprop()

# The norm can also be specified:
network.compile_backprop('l2')
network.compile_backprop('linf')
```

### Examples

An example script demonstrating the use of `check` can be found in `examples/scripts/evaluation_script.py`.
This script can be called from the command line; an example usage can be found in `examples/scripts/example.sh`.
A number of pre-trained weights for the models used in the paper are provided in `examples/models/`.
