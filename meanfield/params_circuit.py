"""params_circuit.py: Summarizes parameters of neurons and the network in
dictionary. Each parameter set implements two functions. One specifying
all default parameter and one calculating the parameter that can be
derived from the default parameter. The default parameter will be used
to create a hash when saving results.

Authors: Hannah Bos, Jannis Schuecker
"""

import numpy as np
import hashlib as hl
import h5py_wrapper.wrapper as h5

def get_data_3dcircuit(new_params={}):
    """ Implements dictionary specifying all parameter of the microcircuit.

    Keyword arguments:
    new_params: dictionary, overwrites default parameter

    Output:s
    params: dictionary with default and derived parameter
    param_keys: list of default parameter
    """
    params = {}

    params['populations'] = ['E', 'I', 'SOM']
    # number of neurons in populations
    params['N'] = np.array([2000, 323, 176])

    ### Neurons
    params['C'] = 250.0    # membrane capacitance in pF
    params['taum'] = 10.0  # membrane time constant in ms
    params['taur'] = 2.0   # refractory time in ms
    params['V0'] = -65.0   # reset potential in mV
    params['Vth'] = -50.0  # threshold of membrane potential in mV

    ### Synapses
    params['tauf'] = 0.5  # synaptic time constant in ms
    params['de'] = 1.5    # delay of excitatory connections in ms
    params['di'] = 0.75   # delay of inhibitory connections in ms
    # standard deviation of delay of excitatory connections in ms
    params['de_sd'] = params['de']*0.5
    # standard deviation of delay of inhibitory connections in ms
    params['di_sd'] = params['di']*0.5
    # delay distribution, options: 'none', 'gaussian' (standard deviation
    # is defined above), 'truncated gaussian' (standard deviation is
    # defined above, truncation at zero)
    params['delay_dist'] = 'truncated_gaussian'
    # PSC amplitude in pA
    params['w'] = 87.8*0.5*10

    ### Connectivity
    # connection probability
    params['p']=np.array([[0.05, 0.12, 0.1],
                          [0.1, 0.1, 0.1],
                          [0.1, 0.0, 0.11]])
    # ratio of inhibitory to excitatory weights
    params['g']=4.0

    ### External input
    params['v_ext'] = 8.0 # in Hz
    # number of external neurons
    params['Next'] = np.array([150, 100, 100])

    ### Neural response
    # Transfer function is either calculated analytically ('analytical')
    # or approximated by an exponential ('empirical'). In the latter case
    # the time constants in response to an incoming positive/negative
    # impulse ('tau_impulse_p'/'tau_impulse_n'), as well as the
    # instantaneous rate jumps ('delta_f_p'/'delta_f_n') have to be
    # specified.
    params['Wilson-Cowan'] = False
    params['tf_mode'] = 'analytical'
    if params['tf_mode'] == 'empirical':
        params['tau_impulse_p'] = np.asarray([0.0 for i in range(8)])
        params['tau_impulse_n'] = np.asarray([0.0 for i in range(8)])
        params['delta_f_p'] = np.asarray([0.0 for i in range(8)])
        params['delta_f_n'] = np.asarray([0.0 for i in range(8)])
    # number of modes used when fast response time constants are calculated
    params['num_modes'] = 1

    # create list of parameter keys that are used to create hashes
    param_keys = params.keys()
    # Remove delay parameter from key list since they don't contribute
    # when calculating the working point and they are incorporated into
    # the transfer function after it has been read from file
    for element in ['de', 'di', 'de_sd', 'di_sd', 'delay_dist']:
        param_keys.remove(element)

    # file storing results
    params['datafile'] = 'results_3dcircuit.h5'

    # update parameter dictionary with new parameters
    params.update(new_params)

    # calculate all dependent parameters
    params = get_dependend_params_3dcircuit(params)

    return params, param_keys

def get_dependend_params_3dcircuit(params):
    """Returns dictionary with parameter which can be derived from the
    default parameter.
    """
    # indegrees
    params['I'] = np.log(1-params['p'])/(params['N']*np.log(1.-1./(params['N']*params['N'])))
    # weight matrix, only calculated if not already specified in params
    if 'W' not in params:
        W = np.array([[1, -params['g'], -params['g']],
                      [1, -params['g'], -params['g']],
                      [1, 0, -params['g']]])
        params['W'] = params['w']*W
    # delay matrix
    D = np.ones((3,3))*params['di']
    D[:,0] = np.ones(3)*params['de']
    params['Delay'] = D

    # delay standard deviation matrix
    D = np.ones((3,3))*params['di_sd']
    D[:,0] = np.ones(3)*params['de_sd']
    params['Delay_sd'] = D

    return params

def create_hashes(params, param_keys):
    """Returns hash of values of parameters listed in param_keys."""
    label = ''
    for key in param_keys:
        value = params[key]
        if isinstance(value, (np.ndarray, np.generic)):
            label += value.tostring()
        else:
            label += str(value)
    return hl.md5(label).hexdigest()
