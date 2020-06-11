# -*- coding: utf-8 -*-
#
# stimulus_params.py
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
microcircuit stimulus parameters
--------------------------------

Stimulus parameters for the microcircuit.

Hendrik Rothe, Hannah Bos, Sacha van Albada; May 2016
'''

import numpy as np
from network_params import net_dict

stim_dict = {
    # Turn thalamic input on or off (True or False).
    'thalamic_input': False,
    # Turn DC input on or off (True or False).
    'dc_input': False,

    # turns stimulus vector on or off
    'stimulus_vector': False,
    # each neuron receives Gaussian weighted input from an independent
    # Poisson source
    # 'fixed': all Poisson sources have the same rate
    # 'exponential': the rate of the Poisson sources is drawn from an
    #                exponential distribution
    'stimulus_type': 'exponential',

    # turns tuned input on or off
    'tuned_input': False,
    # each neuron (in pop0 and pop1) receives input from a poisson source
    # with rate is modeled
    # according to r=r_theta*exp(-(theta-theta_p)^2/(2*sigma^2))
    # preferred angle for neuron n: theta_p=180*n/N
    # parameter required: r0, sigma_theta, theta
    'r_theta': '20.0',
    'sigma_theta': '20.0',
    'theta': '0.0',

    # inhibits some SOM cells with probability p_inhibit_SOM
    'inhibit_SOM': False,
    'p_inhibit_SOM': '0.0',

    ## Thalamus
    # Number of thalamic neurons.
    'n_thal': 1,
    # Mean amplitude of the thalamic postsynaptic potential (in mV).
    'PSP_th': 0.15*3.,
    # Standard deviation of the postsynaptic potential (in relative units).
    'PSP_sd': 0.1,
    # Start of the thalamic input (in ms).
    'th_start': 0.0,
    # Duration of the thalamic input (in ms).
    'th_duration': 1000.0,
    # Rate of the thalamic input (in Hz).
    'th_rate': 10.0,
    # Connection probabilities of the thalamus to the different populations.
    # Order as in 'populations' in 'network_params.py'
    'conn_probs_th':
        np.array([0.5, 0.5]),
    # Mean delay of the thalamic input (in ms).
    'delay_th':
        np.asarray([0.1 for i in list(range(len(net_dict['populations'])))]),
    # Standard deviation of the thalamic delay (in ms).
    'delay_th_sd':
        np.asarray([0.1 for i in list(range(len(net_dict['populations'])))]),

    ## DC generator
    # Start of the DC generator (in ms).
    'dc_start': 0.0,
    # Duration of the DC generator (in ms).
    'dc_dur': 1000.0,
    # Amplitude of the DC generator (in pA).
    'dc_amp': np.ones(len(net_dict['populations'])) * 0.3,
    }
