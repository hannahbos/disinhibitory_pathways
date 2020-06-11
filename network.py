# -*- coding: utf-8 -*-
#
# network.py
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
pynest microcircuit network
---------------------------

Main file for the microcircuit.

Hendrik Rothe, Hannah Bos, Sacha van Albada; May 2016

This example uses the function GetNodes, which is deprecated. A deprecation
warning is therefore issued. For details about deprecated functions, see
documentation.
'''

import nest
import numpy as np
import os
from helpers import synapses_th_matrix
from helpers import get_total_number_of_synapses
from helpers import get_weight
from helpers import compute_DC
import h5py_wrapper.wrapper as h5


class Network:
    """ Handles the setup of the network parameters and
    provides functions to connect the network and devices.

    Arguments
    ---------
    sim_dict
        dictionary containing all parameters specific to the simulation
        such as the directory the data is stored in and the seeds
        (see: sim_params.py)
    net_dict
         dictionary containing all parameters specific to the neurons
         and the network (see: network_params.py)

    Keyword Arguments
    -----------------
    stim_dict
        dictionary containing all parameter specific to the stimulus
        (see: stimulus_params.py)

    """
    def __init__(self, sim_dict, net_dict, stim_dict=None):
        self.sim_dict = sim_dict
        self.net_dict = net_dict
        if stim_dict is not None:
            self.stim_dict = stim_dict
        else:
            self.stim_dict = None
        self.data_path = sim_dict['data_path']
        if nest.Rank() == 0:
            if os.path.isdir(self.sim_dict['data_path']):
                print('data directory already exists')
            else:
                os.mkdir(self.sim_dict['data_path'])
                print('data directory created')
            print('Data will be written to %s' % self.data_path)

    def setup_nest(self):
        """ Hands parameters to the NEST-kernel.

        Resets the NEST-kernel and passes parameters to it.
        The number of seeds for the NEST-kernel is computed, based on the
        total number of MPI processes and threads of each.

        """
        nest.ResetKernel()
        master_seed = self.sim_dict['master_seed']
        if nest.Rank() == 0:
            print('Master seed: %i ' % master_seed)
        nest.SetKernelStatus(
            {'local_num_threads': self.sim_dict['local_num_threads']}
            )
        N_tp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
        if nest.Rank() == 0:
            print('Number of total processes: %i' % N_tp)
        rng_seeds = list(
            range(
                master_seed + 1 + N_tp,
                master_seed + 1 + (2 * N_tp)
                )
            )
        grng_seed = master_seed + N_tp
        if nest.Rank() == 0:
            print(
                'Seeds for random number generators of virtual processes: %r'
                % rng_seeds
                )
            print('Global random number generator seed: %i' % grng_seed)
        self.pyrngs = [np.random.RandomState(s) for s in list(range(
            master_seed, master_seed + N_tp))]
        self.sim_resolution = self.sim_dict['sim_resolution']
        kernel_dict = {
            'resolution': self.sim_resolution,
            'grng_seed': grng_seed,
            'rng_seeds': rng_seeds,
            'overwrite_files': self.sim_dict['overwrite_files'],
            'print_time': self.sim_dict['print_time'],
            }
        nest.SetKernelStatus(kernel_dict)

    def create_populations(self):
        """ Creates the neuronal populations.

        The neuronal populations are created and the parameters are assigned
        to them. The initial membrane potential of the neurons is drawn from a
        normal distribution.

        """
        self.N_full = self.net_dict['N_full']
        self.synapses = get_total_number_of_synapses(self.net_dict)
        self.nr_neurons = self.N_full
        self.K_ext = self.net_dict['K_ext']
        if 'K_ext_inh' in self.net_dict:
            self.K_ext_inh = self.net_dict['K_ext_inh']
        if 'K_ext_inh_excweight' in self.net_dict:
            self.K_ext_inh_excweight = self.net_dict['K_ext_inh_excweight']
        self.w_from_PSP = get_weight(self.net_dict['PSP_e'], self.net_dict)
        self.weight_mat = get_weight(
            self.net_dict['PSP_mean_matrix'], self.net_dict
            )
        self.weight_mat_std = self.net_dict['PSP_std_matrix']
        self.w_ext = self.w_from_PSP
        if self.net_dict['poisson_input']:
            self.DC_amp_e = np.zeros(len(self.net_dict['populations']))
        else:
            if nest.Rank() == 0:
                print(
                    '''
                    no poisson input provided
                    calculating dc input to compensate
                    '''
                    )
            self.DC_amp_e = compute_DC(self.net_dict, self.w_ext)

        # Create cortical populations.
        self.pops = []
        pop_file = open(
            os.path.join(self.data_path, 'population_GIDs.dat'), 'w+'
            )
        for i, pop in enumerate(self.net_dict['populations']):
            population = nest.Create(
                self.net_dict['neuron_model'], int(self.nr_neurons[i])
                )
            nest.SetStatus(
                population, {
                    'tau_syn_ex': self.net_dict['neuron_params']['tau_syn_ex'],
                    'tau_syn_in': self.net_dict['neuron_params']['tau_syn_in'],
                    'E_L': self.net_dict['neuron_params']['E_L'],
                    'V_th': self.net_dict['neuron_params']['V_th'],
                    'V_reset':  self.net_dict['neuron_params']['V_reset'],
                    't_ref': self.net_dict['neuron_params']['t_ref'],
                    'I_e': self.DC_amp_e[i]
                    }
                )
            self.pops.append(population)
            pop_file.write('%d  %d \n' % (population[0], population[-1]))
        pop_file.close()
        for thread in np.arange(nest.GetKernelStatus('local_num_threads')):
            # Using GetNodes is a work-around until NEST 3.0 is released. It
            # will issue a deprecation warning.
            local_nodes = nest.GetNodes(
                [0], {
                    'model': self.net_dict['neuron_model'],
                    'thread': thread
                    }, local_only=True
                )[0]
            vp = nest.GetStatus(local_nodes)[0]['vp']
            # vp is the same for all local nodes on the same thread
            nest.SetStatus(
                local_nodes, 'V_m', self.pyrngs[vp].normal(
                    self.net_dict['neuron_params']['V0_mean'],
                    self.net_dict['neuron_params']['V0_sd'],
                    len(local_nodes))
                    )

    def create_devices(self):
        """ Creates the recording devices.

        Only devices which are given in net_dict['rec_dev'] are created.

        """
        self.spike_detector = []
        self.voltmeter = []
        for i, pop in enumerate(self.pops):
            if 'spike_detector' in self.net_dict['rec_dev']:
                recdict = {
                    'withgid': True,
                    'withtime': True,
                    'to_memory': False,
                    'to_file': True,
                    'label': os.path.join(self.data_path, 'spikes-pop' + str(i))
                    }
                dummy = nest.Create('spike_detector', params=recdict)
                self.spike_detector.append(dummy)
            if 'voltmeter' in self.net_dict['rec_dev']:
                recdictmem = {
                    'interval': self.sim_dict['rec_V_int'],
                    'withgid': True,
                    'withtime': True,
                    'to_memory': False,
                    'to_file': True,
                    'label': os.path.join(self.data_path, 'voltmeter'),
                    'record_from': ['V_m'],
                    }
                volt = nest.Create('voltmeter', params=recdictmem)
                self.voltmeter.append(volt)
        if 'multimeter' in self.net_dict['rec_dev']:
                recdictmulti = {
                    'withgid': True,
                    'withtime': True,
                    'to_memory': False,
                    'to_file': True,
                    "record_from": ["V_m", "I_syn_ex",
                                    "I_syn_in"],
                    'label': os.path.join(self.data_path,
                                          'multimeter' + str(i))
                    }
                self.multimeter = nest.Create('multimeter',
                                              params=recdictmulti)
        if 'spike_detector' in self.net_dict['rec_dev']:
            if nest.Rank() == 0:
                print('Spike detectors created')
        if 'voltmeter' in self.net_dict['rec_dev']:
            if nest.Rank() == 0:
                print('Voltmeters created')
        if 'multimeter' in self.net_dict['rec_dev']:
            if nest.Rank() == 0:
                print('Multimeters created')

    def create_thalamic_input(self):
        """ This function creates the thalamic neuronal population if this
        is specified in stimulus_params.py.

        """
        if self.stim_dict['thalamic_input']:
            if nest.Rank() == 0:
                print('Thalamic input provided')
            self.thalamic_population = nest.Create(
                'parrot_neuron', self.stim_dict['n_thal']
                )
            self.thalamic_weight = get_weight(
                self.stim_dict['PSP_th'], self.net_dict
                )
            self.stop_th = (
                self.stim_dict['th_start'] + self.stim_dict['th_duration']
                )
            self.poisson_th = nest.Create('poisson_generator')
            nest.SetStatus(
                self.poisson_th, {
                    'rate': self.stim_dict['th_rate'],
                    'start': self.stim_dict['th_start'],
                    'stop': self.stop_th
                    }
                )
            nest.Connect(self.poisson_th, self.thalamic_population)
            self.nr_synapses_th = synapses_th_matrix(
                self.net_dict, self.stim_dict
                )
        else:
            if nest.Rank() == 0:
                print('Thalamic input not provided')

    def create_poisson(self):
        """ Creates the Poisson generators.

        If Poissonian input is provided, the Poissonian generators are created
        and the parameters needed are passed to the Poissonian generator.

        """
        if self.net_dict['poisson_input']:
            if nest.Rank() == 0:
                print('Poisson background input created')
            rate_ext = self.net_dict['bg_rate'] * self.K_ext
            self.poisson = []
            for i, target_pop in enumerate(self.pops):
                poisson = nest.Create('poisson_generator')
                nest.SetStatus(poisson, {'rate': rate_ext[i]})
                self.poisson.append(poisson)

    def create_poisson_inh(self):
        """ Creates the Poisson generators.

        If Poissonian input is provided, the Poissonian generators are created
        and the parameters needed are passed to the Poissonian generator.

        """
        if self.net_dict['poisson_input']:
            if nest.Rank() == 0:
                print('Poisson background input created')
            rate_ext = self.net_dict['bg_rate'] * self.K_ext_inh
            self.poisson_inh = []
            for i, target_pop in enumerate(self.pops):
                poisson = nest.Create('poisson_generator')
                nest.SetStatus(poisson, {'rate': rate_ext[i]})
                self.poisson_inh.append(poisson)

    def create_poisson_inh_excweight(self):
        """ Creates the Poisson generators.

        If Poissonian input is provided, the Poissonian generators are created
        and the parameters needed are passed to the Poissonian generator.

        """
        if self.net_dict['poisson_input']:
            if nest.Rank() == 0:
                print('Poisson background input created')
            rate_ext = self.net_dict['bg_rate'] * self.K_ext_inh_excweight
            self.poisson_inh_excweight = []
            for i, target_pop in enumerate(self.pops):
                poisson = nest.Create('poisson_generator')
                nest.SetStatus(poisson, {'rate': rate_ext[i]})
                self.poisson_inh_excweight.append(poisson)

    def create_poisson_inh_fake_space(self):
        """ Creates the Poisson generators.

        If Poissonian input is provided, the Poissonian generators are created
        and the parameters needed are passed to the Poissonian generator.

        """
        if self.net_dict['poisson_input']:
            if nest.Rank() == 0:
                print('Poisson background input created')
            self.poisson_inh = []
            rho_start = self.net_dict['rho_start']
            rho_stop = self.net_dict['rho_stop']
            for i, target_pop in enumerate([self.pops[0], self.pops[1]]):
                self.poisson_inh.append([])
                N = self.net_dict['N_full'][i]
                for j in range(N):
                    poisson = nest.Create('poisson_generator')
                    rate_ext = self.net_dict['bg_rate'] * self.K_ext_inh[i]
                    rate_ext *= (rho_start+j/float(N)*(rho_stop-rho_start))
                    nest.SetStatus(poisson, {'rate': rate_ext})
                    self.poisson_inh[i] += poisson

    def create_poisson_tuned_input(self):
        """ Creates the Poisson generators.

        If Poissonian input is provided, the Poissonian generators are created
        and the parameters needed are passed to the Poissonian generator.

        """
        r_th = self.stim_dict['r_theta']
        s_th = self.stim_dict['sigma_theta']
        theta = self.stim_dict['theta']
        # tuned input to excitatory neurons
        N = int(self.net_dict['N_full'][0])
        modulated_rate = [r_th*np.exp(-np.power(theta-180*i/float(N),2)/(2.*s_th*s_th)) for i in range(N)]
        self.poisson_tuned_input_exc = []
        for i in range(N):
            poisson = nest.Create('poisson_generator')
            nest.SetStatus(poisson, {'rate': modulated_rate[i]})
            self.poisson_tuned_input_exc.append(poisson[0])
        # tuned input to inhibitory neurons
        N = int(self.net_dict['N_full'][1])
        tPV = self.stim_dict['tuned_PV']
        modulated_rate = [r_th*np.exp(-np.power(theta-180*i/float(N),2)/(2.*s_th*s_th)) for i in range(N)]
        self.poisson_tuned_input_inh = []
        for i in range(N):
            poisson = nest.Create('poisson_generator')
            nest.SetStatus(poisson, {'rate': tPV*modulated_rate[i]})
            self.poisson_tuned_input_inh.append(poisson[0])

    def create_poisson_inhibit_SOM(self):
        """ Creates the Poisson generators.

        If Poissonian input is provided, the Poissonian generators are created
        and the parameters needed are passed to the Poissonian generator.

        """
        rate_ext = self.net_dict['bg_rate'] * 200
        poisson = nest.Create('poisson_generator')
        nest.SetStatus(poisson, {'rate': rate_ext})
        self.poisson_inhibit_SOM = poisson

    def create_dc_generator(self):
        """ Creates a DC input generator.

        If DC input is provided, the DC generators are created and the
        necessary parameters are passed to them.

        """
        if self.stim_dict['dc_input']:
            if nest.Rank() == 0:
                print('DC generator created')
            dc_amp_stim = self.net_dict['K_ext'] * self.stim_dict['dc_amp']
            self.dc = []
            if nest.Rank() == 0:
                print('DC_amp_stim', dc_amp_stim)
            for i, target_pop in enumerate(self.pops):
                dc = nest.Create(
                    'dc_generator', params={
                        'amplitude': dc_amp_stim[i],
                        'start': self.stim_dict['dc_start'],
                        'stop': (
                            self.stim_dict['dc_start'] +
                            self.stim_dict['dc_dur']
                            )
                        }
                    )
                self.dc.append(dc)

    def create_connections(self):
        """ Creates the recurrent connections.

        The recurrent connections between the neuronal populations are created.

        """
        if nest.Rank() == 0:
            print('Recurrent connections are established')
        mean_delays = self.net_dict['mean_delay_matrix']
        std_delays = self.net_dict['std_delay_matrix']
        for i, target_pop in enumerate(self.pops):
            for j, source_pop in enumerate(self.pops):
                synapse_nr = int(self.synapses[i][j])
                if synapse_nr >= 0.:
                    weight = self.weight_mat[i][j]
                    w_sd = abs(weight * self.weight_mat_std[i][j])
                    conn_dict_rec = {
                        'rule': 'fixed_total_number', 'N': synapse_nr
                        }
                    syn_dict = {
                        'model': 'static_synapse',
                        'weight': {
                            'distribution': 'normal_clipped', 'mu': weight,
                            'sigma': w_sd
                            },
                        'delay': {
                            'distribution': 'normal_clipped',
                            'mu': mean_delays[i][j], 'sigma': std_delays[i][j],
                            'low': self.sim_resolution
                            }
                        }
                    if weight < 0:
                        syn_dict['weight']['high'] = 0.0
                    else:
                        syn_dict['weight']['low'] = 0.0
                    nest.Connect(
                        source_pop, target_pop,
                        conn_spec=conn_dict_rec,
                        syn_spec=syn_dict
                        )

    def connect_poisson(self):
        """ Connects the Poisson generators to the microcircuit."""
        if nest.Rank() == 0:
            print('Poisson background input is connected')
        for i, target_pop in enumerate(self.pops):
            conn_dict_poisson = {'rule': 'all_to_all'}
            syn_dict_poisson = {
                'model': 'static_synapse',
                'weight': self.w_ext,
                'delay': self.net_dict['poisson_delay']
                }
            nest.Connect(
                self.poisson[i], target_pop,
                conn_spec=conn_dict_poisson,
                syn_spec=syn_dict_poisson
                )

    def connect_poisson_inh(self):
        """ Connects the Poisson generators to the microcircuit."""
        if nest.Rank() == 0:
            print('Poisson background input is connected')
        for i, target_pop in enumerate(self.pops):
            conn_dict_poisson = {'rule': 'all_to_all'}
            syn_dict_poisson = {
                'model': 'static_synapse',
                'weight': self.net_dict['g']*self.w_ext,
                'delay': self.net_dict['poisson_delay']
                }
            nest.Connect(
                self.poisson_inh[i], target_pop,
                conn_spec=conn_dict_poisson,
                syn_spec=syn_dict_poisson
                )

    def connect_poisson_inh_excweight(self):
        """ Connects the Poisson generators to the microcircuit."""
        if nest.Rank() == 0:
            print('Poisson background input is connected')
        for i, target_pop in enumerate(self.pops):
            conn_dict_poisson = {'rule': 'all_to_all'}
            syn_dict_poisson = {
                'model': 'static_synapse',
                'weight': -1*self.w_ext,
                'delay': self.net_dict['poisson_delay']
                }
            nest.Connect(
                self.poisson_inh_excweight[i], target_pop,
                conn_spec=conn_dict_poisson,
                syn_spec=syn_dict_poisson
                )

    def connect_poisson_inh_fake_space(self):
        """ Connects the Poisson generators to the microcircuit."""
        if nest.Rank() == 0:
            print('Poisson background input is connected')
        for i, target_pop in enumerate([self.pops[0],self.pops[1]]):
            conn_dict_poisson = {'rule': 'one_to_one'}
            syn_dict_poisson = {
                'model': 'static_synapse',
                'weight': self.net_dict['g']*self.w_ext,
                'delay': self.net_dict['poisson_delay']
                }
            nest.Connect(
                self.poisson_inh[i], target_pop,
                conn_spec=conn_dict_poisson,
                syn_spec=syn_dict_poisson
                )

    def connect_tuned_input(self):
        """ Connects the Poisson generators to the microcircuit."""
        if nest.Rank() == 0:
            print('Poisson background input is connected')
        conn_dict_poisson = {'rule': 'one_to_one'}
        syn_dict_poisson = {
            'model': 'static_synapse',
            'weight': self.w_ext,
            'delay': self.net_dict['poisson_delay']
            }
        nest.Connect(
            self.poisson_tuned_input_exc, self.pops[0],
            conn_spec=conn_dict_poisson,
            syn_spec=syn_dict_poisson
            )
        nest.Connect(
            self.poisson_tuned_input_inh, self.pops[1],
            conn_spec=conn_dict_poisson,
            syn_spec=syn_dict_poisson
            )

    def connect_inhibit_SOM(self):
        """ Connects the Poisson generators to the microcircuit."""
        if nest.Rank() == 0:
            print('Poisson background input is connected')
        conn_dict_poisson = {'rule': 'pairwise_bernoulli',
            'p': self.stim_dict['p_inhibit_SOM']}
        syn_dict_poisson = {
            'model': 'static_synapse',
            'weight': self.net_dict['g']*self.w_ext,
            'delay': self.net_dict['poisson_delay']
            }
        nest.Connect(
            self.poisson_inhibit_SOM, self.pops[2],
            conn_spec=conn_dict_poisson,
            syn_spec=syn_dict_poisson
            )

    def connect_thalamus(self):
        """ Connects the thalamic population to the microcircuit."""
        if nest.Rank() == 0:
            print('Thalamus connection established')
        for i, target_pop in enumerate(self.pops):
            conn_dict_th = {
                'rule': 'fixed_total_number',
                'N': int(self.nr_synapses_th[i])
                }
            syn_dict_th = {
                'weight': {
                    'distribution': 'normal_clipped',
                    'mu': self.thalamic_weight,
                    'sigma': (
                        self.thalamic_weight * self.net_dict['PSP_sd']
                        ),
                    'low': 0.0
                    },
                'delay': {
                    'distribution': 'normal_clipped',
                    'mu': self.stim_dict['delay_th'][i],
                    'sigma': self.stim_dict['delay_th_sd'][i],
                    'low': self.sim_resolution
                    }
                }
            nest.Connect(
                self.thalamic_population, target_pop,
                conn_spec=conn_dict_th, syn_spec=syn_dict_th
                )

    def connect_dc_generator(self):
        """ Connects the DC generator to the microcircuit."""
        if nest.Rank() == 0:
            print('DC Generator connection established')
        for i, target_pop in enumerate(self.pops):
            if self.stim_dict['dc_input']:
                nest.Connect(self.dc[i], target_pop)

    def connect_devices(self):
        """ Connects the recording devices to the microcircuit."""
        if nest.Rank() == 0:
            if ('spike_detector' in self.net_dict['rec_dev'] and
                    'voltmeter' not in self.net_dict['rec_dev']):
                print('Spike detector connected')
            elif ('spike_detector' not in self.net_dict['rec_dev'] and
                    'voltmeter' in self.net_dict['rec_dev']):
                print('Voltmeter connected')
            elif ('spike_detector' in self.net_dict['rec_dev'] and
                    'voltmeter' in self.net_dict['rec_dev']):
                print('Spike detector and voltmeter connected')
            else:
                print('no recording devices connected')
        for i, target_pop in enumerate(self.pops):
            if 'voltmeter' in self.net_dict['rec_dev']:
                nest.Connect(self.voltmeter[i], target_pop)
            if 'spike_detector' in self.net_dict['rec_dev']:
                nest.Connect(target_pop, self.spike_detector[i])
        if 'multimeter' in self.net_dict['rec_dev']:
            nest.Connect(self.multimeter, self.net_dict['rec_neurons'])


    def connect_stimulus_vector(self):
        """ This function connects the stimulus vector
        as specified in stimulus_params.py.

        """
        # indices where stimulus vector is not zero
        nonzero_stim = np.nonzero(self.stim_dict['PSP_stim'])
        PSPs = self.stim_dict['PSP_stim'][nonzero_stim]
        stimulus_weight = get_weight(PSPs, self.net_dict)
        all_neurons = self.pops[0]+self.pops[1]+self.pops[2]
        neurons = [all_neurons[nzs] for nzs in nonzero_stim[0]]
        conn_dict = {'rule': 'one_to_one'}
        syn_dict = {'weight': stimulus_weight}

        rate_vec = np.zeros(np.sum(self.net_dict['N_full']))
        if self.stim_dict['stimulus_type'] != 'fixed':
            rate_mean = self.stim_dict['stim_rate']
            stim_vec = []
            for i,stim in enumerate(self.stim_dict['PSP_stim']):
                if stim != 0:
                    rate = np.random.exponential(rate_mean)
                    rate_vec[i] = rate
                    poisson_stim = nest.Create('poisson_generator',
                                               params={'rate': rate})
                    stim_vec.append(poisson_stim[0])
            # save stimulus vector
            h5.add_to_h5(self.sim_dict['data_path'] + '/results.h5',
                         {'stim_rate_vec': rate_vec}, 'a', overwrite_dataset=True)
        else:
            poisson_stim = nest.Create('poisson_generator',
                            params={'rate': self.stim_dict['stim_rate']})
            stim_vec = [poisson_stim[0] for i in range(len(neurons))]
        nest.Connect(stim_vec, neurons, conn_dict, syn_dict)

    def setup(self):
        """ Execute subfunctions of the network.

        This function executes several subfunctions to create neuronal
        populations, devices and inputs, connects the populations with
        each other and with devices and input nodes.

        """
        self.setup_nest()
        self.create_populations()
        self.create_devices()
        self.create_thalamic_input()
        self.create_poisson()
        self.create_dc_generator()
        self.create_connections()
        if self.net_dict['poisson_input']:
            self.connect_poisson()
        if self.stim_dict['thalamic_input']:
            self.connect_thalamus()
        if self.stim_dict['dc_input']:
            self.connect_dc_generator()
        if self.stim_dict['stimulus_vector']:
            self.connect_stimulus_vector()
        if self.stim_dict['tuned_input']:
            self.create_poisson_tuned_input()
            self.connect_tuned_input()
        if self.stim_dict['inhibit_SOM']:
            self.create_poisson_inhibit_SOM()
            self.connect_inhibit_SOM()
        if 'K_ext_inh' in self.net_dict:
            self.create_poisson_inh()
            self.connect_poisson_inh()
        if 'K_ext_inh_excweight' in self.net_dict:
            self.create_poisson_inh_excweight()
            self.connect_poisson_inh_excweight()
        self.connect_devices()

    def simulate(self):
        """ Simulates the microcircuit."""
        nest.Simulate(self.sim_dict['t_sim'])

    def get_weighted_connectivity_matrix(self, pop1, pop2):
        '''
        Returns a weighted connectivity matrix describing all connections from pop1 to pop2
        such that M_ij describes the connection between the jth neuron in pop1 to the ith
        neuron in pop2.
        Only works without multapses.
        '''

        source_min = pop1[0]
        target_min = pop2[0]
        M = np.zeros((len(pop2),len(pop1)))
        connections = nest.GetConnections(pop1,pop2)
        connections_detailed = nest.GetStatus(connections)
        for i,conn in enumerate(connections_detailed):
            M[conn['target']-target_min][conn['source']-source_min] += conn['weight']
        label = 'source_from' + str(source_min) + 'to' + str(pop1[-1]) + '_'
        label += 'target_from' + str(target_min) + 'to' + str(pop2[-1])
        h5.add_to_h5(self.sim_dict['data_path'] + '/results.h5',
                     {label: {'M': M}}, 'a', overwrite_dataset=True)
