# -*- coding: utf-8 -*-
#
# network_params.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.

'''
pynest microcircuit parameters
------------------------------

Network parameters for the microcircuit.

Hendrik Rothe, Hannah Bos, Sacha van Albada; May 2016
'''

import numpy as np


def get_mean_delays(mean_delay_exc, mean_delay_inh, number_of_pop):
    """ Creates matrix containing the delay of all connections.

    Arguments
    ---------
    mean_delay_exc
        Delay of the excitatory connections.
    mean_delay_inh
        Delay of the inhibitory connections.
    number_of_pop
        Number of populations.

    Returns
    -------
    mean_delays
        Matrix specifying the mean delay of all connections.

    """

    dim = number_of_pop
    mean_delays = np.ones((dim, dim))*mean_delay_inh
    mean_delays[:, 0] = np.ones(dim)*mean_delay_exc
    return mean_delays


def get_std_delays(std_delay_exc, std_delay_inh, number_of_pop):
    """ Creates matrix containing the standard deviations of all delays.

    Arguments
    ---------
    std_delay_exc
        Standard deviation of excitatory delays.
    std_delay_inh
        Standard deviation of inhibitory delays.
    number_of_pop
        Number of populations in the microcircuit.

    Returns
    -------
    std_delays
        Matrix specifying the standard deviation of all delays.

    """

    dim = number_of_pop
    std_delays = np.ones((dim, dim))*std_delay_inh
    std_delays[:, 0] = np.ones(dim)*std_delay_exc
    return std_delays


def get_mean_PSP_matrix(PSP_e, g, number_of_pop):
    """ Creates a matrix of the mean evoked postsynaptic potential.

    The function creates a matrix of the mean evoked postsynaptic
    potentials between the recurrent connections of the microcircuit.
    The weight of the connection from L4E to L23E is doubled.

    Arguments
    ---------
    PSP_e
        Mean evoked potential.
    g
        Relative strength of the inhibitory to excitatory connection.
    number_of_pop
        Number of populations in the microcircuit.

    Returns
    -------
    weights
        Matrix of the weights for the recurrent connections.

    """
    dim = number_of_pop
    weights = np.zeros((dim, dim))
    exc = PSP_e
    inh = PSP_e * g
    weights[:, 0:dim:2] = exc
    weights[:, 1:dim:2] = inh
    weights[:, 2:dim:2] = inh
    return weights


def get_std_PSP_matrix(PSP_rel, number_of_pop):
    """ Relative standard deviation matrix of postsynaptic potential created.

    The relative standard deviation matrix of the evoked postsynaptic potential
    for the recurrent connections of the microcircuit is created.

    Arguments
    ---------
    PSP_rel
        Relative standard deviation of the evoked postsynaptic potential.
    number_of_pop
        Number of populations in the microcircuit.

    Returns
    -------
    std_mat
        Matrix of the standard deviation of postsynaptic potentials.

    """
    dim = number_of_pop
    std_mat = np.zeros((dim, dim))
    std_mat[:, :] = PSP_rel
    return std_mat

net_dict = {
    # Neuron model.
    'neuron_model': 'iaf_psc_exp',
    # The default recording device is the spike_detector. If you also
    # want to record the membrane potentials of the neurons, add
    # 'voltmeter' to the list.
    'rec_dev': ['spike_detector'],
    'rec_neurons': [1+i for i in range(100)] +
                   [4137+i for i in range(100)] +
                   [4701+i for i in range(100)],
    # Names of the simulated populations.
    'populations': ['E', 'I', 'SOM'],
    # Number of neurons in the different populations. The order of the
    # elements corresponds to the names of the variable 'populations'.
    'N_full': np.array([2000, 273, 114]),
    # Connection probabilities. The first index corresponds to the targets
    # and the second to the sources.
    'conn_probs': np.array([[0.02, 0.1, 0.1],
                            [0.1, 0.1, 0.1],
                            [0.1, 0., 0.1]]),
    # Number of external connections to the different populations.
    # The order corresponds to the order in 'populations'.
    'K_ext': np.array([150, 100, 100]),
    # Mean amplitude of excitatory postsynaptic potential (in mV).
    'PSP_e': 0.15*10.0,
    # Relative standard deviation of the postsynaptic potential.
    'PSP_sd': 0.1,
    # Relative inhibitory synaptic strength (in relative units).
    'g': -4,
    # Rate of the Poissonian spike generator (in Hz).
    'bg_rate': 8.,
    # Turn Poisson input on or off (True or False).
    'poisson_input': True,
    # Delay of the Poisson generator (in ms).
    'poisson_delay': 0.1,
    # Mean delay of excitatory connections (in ms).
    'mean_delay_exc': 1.5,
    # Mean delay of inhibitory connections (in ms).
    'mean_delay_inh': 0.75,
    # Relative standard deviation of the delay of excitatory and
    # inhibitory connections (in relative units).
    'rel_std_delay': 1.0,
    # Parameters of the neurons.
    'neuron_params': {
        # Membrane potential average for the neurons (in mV).
        'V0_mean': -58.0,
        # Standard deviation of the average membrane potential (in mV).
        'V0_sd': 10.0,
        # Reset membrane potential of the neurons (in mV).
        'E_L': -65.0,
        # Threshold potential of the neurons (in mV).
        'V_th': -50.0,
        # Membrane potential after a spike (in mV).
        'V_reset': -65.0,
        # Membrane capacitance (in pF).
        'C_m': 250.0,
        # Membrane time constant (in ms).
        'tau_m': 10.0,
        # Time constant of postsynaptic excitatory currents (in ms).
        'tau_syn_ex': 0.5,
        # Time constant of postsynaptic inhibitory currents (in ms).
        'tau_syn_in': 0.5,
        # Time constant of external postsynaptic excitatory current (in ms).
        'tau_syn_E': 0.5,
        # Refractory period of the neurons after a spike (in ms).
        't_ref': 2.0}
    }

updated_dict = {
    # PSP mean matrix.
    'PSP_mean_matrix': get_mean_PSP_matrix(
        net_dict['PSP_e'], net_dict['g'], len(net_dict['populations'])
        ),
    # PSP std matrix.
    'PSP_std_matrix': get_std_PSP_matrix(
        net_dict['PSP_sd'], len(net_dict['populations'])
        ),
    # mean delay matrix.
    'mean_delay_matrix': get_mean_delays(
        net_dict['mean_delay_exc'], net_dict['mean_delay_inh'],
        len(net_dict['populations'])
        ),
    # std delay matrix.
    'std_delay_matrix': get_std_delays(
        net_dict['mean_delay_exc'] * net_dict['rel_std_delay'],
        net_dict['mean_delay_inh'] * net_dict['rel_std_delay'],
        len(net_dict['populations'])
        ),
    }


net_dict.update(updated_dict)
