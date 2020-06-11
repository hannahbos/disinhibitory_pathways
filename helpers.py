# -*- coding: utf-8 -*-
#
# helpers.py
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
pynest microcircuit helpers
---------------------------

Helper functions for the simulation and evaluation of the microcircuit.

Hendrik Rothe, Hannah Bos, Sacha van Albada; May 2016
'''

import numpy as np
import os
import sys
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def compute_DC(net_dict, w_ext):
    """ Computes DC input if no Poisson input is provided to the microcircuit.

    Parameters
    ----------
    net_dict
        Parameters of the microcircuit.
    w_ext
        Weight of external connections.

    Returns
    -------
    DC
        DC input, which compensates lacking Poisson input.
    """
    DC = (
        net_dict['bg_rate'] * net_dict['K_ext'] *
        w_ext * net_dict['neuron_params']['tau_syn_E'] * 0.001
        )
    return DC


def get_weight(PSP_val, net_dict):
    """ Computes weight to elicit a change in the membrane potential.

    This function computes the weight which elicits a change in the membrane
    potential of size PSP_val. To implement this, the weight is calculated to
    elicit a current that is high enough to implement the desired change in the
    membrane potential.

    Parameters
    ----------
    PSP_val
        Evoked postsynaptic potential.
    net_dict
        Dictionary containing parameters of the microcircuit.

    Returns
    -------
    PSC_e
        Weight value(s).

    """
    C_m = net_dict['neuron_params']['C_m']
    tau_m = net_dict['neuron_params']['tau_m']
    tau_syn_ex = net_dict['neuron_params']['tau_syn_ex']

    PSC_e_over_PSP_e = (((C_m) ** (-1) * tau_m * tau_syn_ex / (
        tau_syn_ex - tau_m) * ((tau_m / tau_syn_ex) ** (
            - tau_m / (tau_m - tau_syn_ex)) - (tau_m / tau_syn_ex) ** (
                - tau_syn_ex / (tau_m - tau_syn_ex)))) ** (-1))
    PSC_e = (PSC_e_over_PSP_e * PSP_val)
    return PSC_e


def get_total_number_of_synapses(net_dict):
    """ Returns the total number of synapses between all populations.

    The first index (rows) of the output matrix is the target population
    and the second (columns) the source population.

    Parameters
    ----------
    net_dict
        Dictionary containing parameters of the microcircuit.
    N_full
        Number of neurons in all populations.
    number_N
        Total number of populations.
    conn_probs
        Connection probabilities of the eight populations.
    Returns
    -------
    K
        Total number of synapses with
        dimensions [len(populations), len(populations)].

    """
    N_full = net_dict['N_full']
    number_N = len(N_full)
    conn_probs = net_dict['conn_probs']
    prod = np.outer(N_full, N_full)
    n_syn_temp = np.log(1. - conn_probs)/np.log((prod - 1.) / prod)
    N_full_matrix = np.column_stack(
        (N_full for i in list(range(number_N)))
        )
    K = (((n_syn_temp * (
        N_full_matrix ).astype(int)) / N_full_matrix).astype(int))
    return K


def synapses_th_matrix(net_dict, stim_dict):
    """ Computes number of synapses between thalamus and microcircuit.

    Parameters
    ----------
    net_dict
        Dictionary containing parameters of the microcircuit.
    stim_dict
        Dictionary containing parameters of stimulation settings.
    N_full
        Number of neurons in the eight populations.
    number_N
        Total number of populations.
    conn_probs
        Connection probabilities of the thalamus to the eight populations.
    T_full
        Number of thalamic neurons.

    Returns
    -------
    K
        Total number of synapses.

    """
    N_full = net_dict['N_full']
    number_N = len(N_full)
    conn_probs = stim_dict['conn_probs_th']
    T_full = stim_dict['n_thal']
    prod = (T_full * N_full).astype(float)
    n_syn_temp = np.log(1. - conn_probs)/np.log((prod - 1.)/prod)
    K = (((n_syn_temp * (N_full).astype(int))/N_full).astype(int))
    return K

def read_name(path, name):
    """ Reads names and ids of spike detector.

    The names of the spike detectors are gathered and the lowest and
    highest id of each spike detector is computed. If the simulation was
    run on several threads or mpi-processes, one name per spike detector
    per mpi-process/thread is extracted.

    Arguments
    ---------
    path
        Path where the spike detector files are stored.
    name
        Name of the spike detector.

    Returns
    -------
    files
        Name of all spike detectors, which are located in the path.
    gids
        Lowest and highest ids of the spike detectors.

    """
    # Import filenames$
    files = []
    for file in os.listdir(path):
        if file.endswith('.gdf') and file.startswith(name):
            temp = file.split('-')[0] + '-' + file.split('-')[1]
            if temp not in files:
                files.append(temp)

    # Import GIDs
    gidfile = open(path + 'population_GIDs.dat', 'r')
    gids = []
    for l in gidfile:
        a = l.split()
        gids.append([int(a[0]), int(a[1])])
    files = sorted(files)
    return files, gids


def load_spike_times(path, name, begin, end):
    """ Loads spike times of each spike detector.

    Arguments
    ---------
    path
        Path where the files with the spike times are stored.
    name
        Name of the spike detector.
    begin
        Lower boundary value to load spike times.
    end
        Upper boundary value to load spike times.

    Returns
    -------
    data
        Dictionary containing spike times in the interval from 'begin'
        to 'end'.

    """
    files, gids = read_name(path, name)
    data = {}
    for i in list(range(len(files))):
        all_names = os.listdir(path)
        temp3 = [
            all_names[x] for x in list(range(len(all_names)))
            if all_names[x].endswith('gdf') and
            all_names[x].startswith('spike') and
            (all_names[x].split('-')[0] + '-' + all_names[x].split('-')[1]) in
            files[i]
            ]
        data_temp = [np.loadtxt(os.path.join(path, f)) for f in temp3]
        data_concatenated = np.concatenate(data_temp)
        data_raw = data_concatenated[np.argsort(data_concatenated[:, 1])]
        idx = ((data_raw[:, 1] > begin) * (data_raw[:, 1] < end))
        data[i] = data_raw[idx]
    return data

