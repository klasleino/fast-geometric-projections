# fast-geometric-projections
Tool for certifying local robustness in deep networks.

### Installation

A Conda environment is included in `fgp_environment.yml`. You can install the environment with the following command:

```
conda env create -f fgp_environment.yml
```

Alternatively, the required packages for running the demo are `keras`, `tensorflow-gpu`, and `jupyter`.

### Running the Example Notebook

After installing and activating the environment, run a notebook server with

```
jupyter notebook
```

and navigate to the given URL. An example usage of Algorithm 1 (verifying local robustness for a fixed &epsilon;) and Algorithm 2 (finding a certified lower bound) is provided in `example.ipynb`.

### Checking Robustness

The main routine for checking robustness is the `check` function in `robustness_certification.py`. This takes the following arguments:
* `network` : the neural network to be certified. Must be an instance of `SimpleSigmoidFfn` for binary classification tasks or `SimpleSoftmaxFfn` for multi-class classification tasks.
* `x` : a `numpy.Array` containing the instance to check robustness for.
* `epsilon` : the value of &epsilon; to check robustness for.
* `timeout` : (optional) the number of seconds to compute for before returning a `TIMED_OUT` result. If `None`, the computation will continue until the search has completed. NOTE: the time-out functionality will not work on Windows, since it uses signals. We recommend running on a unix machine.
* `lowerbound` : (default False) boolean flag selecting Algorithm 1 (False) or Algorithm 2 (True).
* `keepgoing` : (default False) when True, uses the heuristic given in Section 3.3 for decreasing the number of `UNKNOWN` results by continuing to search the queue when a possible false-positive is found. This heuristic can only be used in Algorithm 1 (i.e., when `lowerbound` is False).
* `batch_size` : (default 1) batch size for processing the queue. Setting to 10-100 for medium to large examples significantly speeds up Algorithm 1. This must be 1 for Algorithm 2 (i.e., when `lowerbound` is True).
* `return_num_visited` : (default False) if set to True, Algorithm 1 will return a tuple containing the robustness result followed by the number of regions visited. Otherwise Algorithm 1 will return the robustness result only.
* `recap` : (default False) if set to True, the algorithm will print out a few statistics as it computes.
* `debug_steps` : (default False) if set to True, the algorithm will print out more detailed statistics as it computes.
* `debug_print_rate` : (default 1) if `debug_steps` is True, detailed information will be printed every `debug_print_rate` iterations of the algorithm.
