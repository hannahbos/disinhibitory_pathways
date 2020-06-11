# create simulation, analysis and figures for the manuscript 'Untangling stability and gain modulation in cortical circuits with multiple interneuron classes'

Dependencies:
 * NEST **2.16.0**
 * h5py_wrapper available on https://github.com/INM-6/h5py_wrapper

Files for figures:
  * `create_figures.py`: creates all figures
  * `create_figures_helpers.py`: helper functions for figure creation, starts NEST simulation
  * `helpers.py`, `network.py`, `network_params.py`, `stimulus_params.py`: NEST files
  * `read_sim.py`: functions that read data created by NEST
  * `meanfield/*`: scripts for meanfield analysis. They build on an earlier version of the neural network meanfield toolbox (https://github.com/INM-6/neural_network_meanfield) which only allowed one synaptic time constant.
  * `data/` empty direction, simulation data will be saved here
