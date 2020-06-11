import numpy as np
from shutil import copyfile
import os
import time

import h5py_wrapper.wrapper as h5
import read_sim as rs
import meanfield.circuit as circuit

# import scripts for NEST simulation
from network_params import net_dict
import network_params as netp
from stimulus_params import stim_dict
import network

## default parameter ##
eps = 1.0
w0 = 87.8*0.5*100.*eps

def set_default_params():
    ## default parameter ##
    params = {}
    params['N'] = 2*2068 # number of neurons
    params['g'] = 4. # inhibitory weight scaling
    params['pEE0'] = 0.05

    # connection probabilities
    params['pEE'] = 0.03
    params['pEI'] = 0.1
    params['pIE'] = 0.05
    params['pII'] = 0.1
    params['pES'] = 0.1
    params['pSE'] = 0.05
    params['pIS'] = 0.07
    params['pSI'] = 0.0
    params['pSS'] = 0.0

    # external input
    params['KEext'] = 110
    params['KIext'] = 100
    params['KSext'] = 100
    params['KEext_inh'] = 0
    params['KIext_inh'] = 0
    params['KSext_inh'] = 45
    return params
    
def get_folder(params):
    folder = '/'
    for key in sorted(params):
        folder += key + str(params[key])
    folder += '/'
    return folder

def get_N(params):
    # Pfeffer: 36% PV, 30% SST
    N_inh = 1/4.*params['N']
    rho0 = 30/36.
    N_PV = N_inh/(1+rho0)
    # same as N_S=rho0*N_PV
    N_S = N_inh-N_PV
    return np.array([params['N'], N_PV, N_S])

def get_connection_probabilities(params):
    p = np.array([[params['pEE'], params['pEI'], params['pES']],
                  [params['pIE'], params['pII'], params['pIS']],
                  [params['pSE'], params['pSI'], params['pSS']]])
    return p

def get_Next(params):
    Next = np.array([params['KEext']/2000., params['KIext']/2000.,
        params['KSext']/2000.])*params['N']
    return Next

def get_Next_inh(params):
    Next_inh = np.array([params['KEext_inh']/2000., params['KIext_inh']/2000.,
        params['KSext_inh']/2000.])*params['N']
    return Next_inh

def set_circuit_params(params):
    circuit_params = {'p': get_connection_probabilities(params),
        'Next': get_Next(params),
        'g': params['g'],
        'w': w0/np.sqrt(params['pEE0']*params['N'])}
    circuit_params['Next_inh'] = get_Next_inh(params)
    circuit_params['N'] = get_N(params)
    return circuit_params

def get_rates_sim(params, T=10000.0, tmin=300.0, calc=False,
    read_old_results=False):
    folder = get_folder(params)
    label = 'plot_rates' + folder

    if read_old_results:
        path = '/Users/bos/Documents/pittsburgh/disinhibitory_circuits/py/E-I-VIP/'
        rates_sim = h5.load_h5(path + 'results.h5', label + '/rates_sim')
        return rates_sim

    if calc:
        rates_sim = []
        for pop in range(3):
            rate = rs.get_all_rates(folder, pop, T, tmin)
            rates_sim.append(np.mean(rate))
        np.asarray(rates_sim)
        h5.add_to_h5('results.h5',{label:{
            'rates_sim': rates_sim}}, 'a', overwrite_dataset=True)
        rates_sim = h5.load_h5('results.h5', label + '/rates_sim')
    else:
        rates_sim = h5.load_h5('results.h5', label + '/rates_sim')
    return rates_sim

def get_rate_one_neuron_sim(params, pop, gid, T=10000.0, tmin=300.0, calc=False):
    folder = get_folder(params)
    label = 'plot_rate_one_neuron' + folder + 'gid' + str(gid) + 'pop' + str(pop)
    if calc:
        rate = rs.get_rate_one_neuron(folder, pop, gid, T, tmin)
        h5.add_to_h5('results.h5',{label:{'rate' : rate}}, 'a',
            overwrite_dataset=True)
    else:
        rate = h5.load_h5('results.h5', label + '/rate')
    return rate

def get_gain_sim(rates0, params, Kstim, T=10000.0, tmin=300.0, calc=False, read_old_results=False):
    stim_params = params.copy()
    stim_params['KEext'] = params['KEext'] + Kstim
    stim_params['KIext'] = params['KIext'] + Kstim
    folder = get_folder(stim_params)
    label = 'plot_rates_gain_sim' + str(Kstim) + folder

    if read_old_results:
        path = '/Users/bos/Documents/pittsburgh/disinhibitory_circuits/py/E-I-VIP/'
        g = h5.load_h5(path + 'results.h5', label + '/g')
        return g

    if calc:
        rates_sim = []
        rate = rs.get_all_rates(folder, 0, T, tmin)
        rate = np.mean(rate)
        # divide by 8 since this is the external firing rate in simulation
        g = (rate-rates0[0])/(Kstim*8.)
        h5.add_to_h5('results.h5',{label:{
            'g': g}}, 'a', overwrite_dataset=True)
    else:
        g = h5.load_h5('results.h5', label + '/g')
    return g

def get_dotplot(params, dt, calc=False, read_old_results=False):
    folder = get_folder(params)
    label = 'plot_rates_all_cvs_dt' + str(dt[0]) + str(dt[1]) + folder

    if read_old_results:
        path = '/Users/bos/Documents/pittsburgh/disinhibitory_circuits/py/E-I-VIP/'
        times  = h5.load_h5(path + 'results.h5', label + '/times')
        gids  = h5.load_h5(path + 'results.h5', label + '/gids')
        return times, gids

    times = []
    gids = []
    if calc:
        for pop in range(3):
            spikes, this_gids = rs.get_data_dotplot(
                 folder, dt[0],  dt[1], pop)
            times.append(np.asarray(spikes)-dt[0])
            gids.append(abs(np.asarray(this_gids)-5170))
        h5.add_to_h5('results.h5',{label:{
             'times': times, 'gids': gids}},'a', overwrite_dataset=True)
    else:
        times  = h5.load_h5('results.h5', label + '/times')
        gids  = h5.load_h5('results.h5', label + '/gids')
    return times, gids

def get_diffapprox_rates(params, calc=False, read_old_results=False):
    folder = get_folder(params)
    label = 'plot_rates_' + folder

    if read_old_results:
        path = '/Users/bos/Documents/pittsburgh/disinhibitory_circuits/py/E-I-VIP/'
        rates = h5.load_h5(path + 'results.h5', label + '/rates_calc')
        return rates

    circuit_params = set_circuit_params(params)

    if calc:
        circ = circuit.Circuit('3dcircuit',
                               circuit_params,
                               analysis_type='stationary',
                               from_file='False')
        rates = circ.th_rates
        h5.add_to_h5('results.h5',{label:{'rates_calc': rates}},
            'a', overwrite_dataset=True)
    else:
        rates = h5.load_h5('results.h5', label + '/rates_calc')
    return rates

def get_diffapprox_rates_betas(params, calc=False):
    folder = get_folder(params)
    label = 'plot_rates_betas' + folder
    circuit_params = set_circuit_params(params)

    if calc:
        circ = circuit.Circuit('3dcircuit',
                               circuit_params,
                               analysis_type='stationary',
                               from_file='False')
        rates = circ.th_rates
        betas = circ.ana.create_H0_mu()[:,0]
        h5.add_to_h5('results.h5',{label:{'rates_calc': rates, 'betas': betas}},
            'a', overwrite_dataset=True)
    else:
        rates = h5.load_h5('results.h5', label + '/rates_calc')
        betas = h5.load_h5('results.h5', label + '/betas')
    return rates, betas

def get_diffapprox_rates_Next_array(params, Next_array, calc=False):
    folder = get_folder(params)
    label = 'plot_rates_Next_array' + folder
    label += str(Next_array[0]) + str(Next_array[1]-Next_array[0])
    label += str(Next_array[-1])
    circuit_params = set_circuit_params(params)

    if calc:
        circ = circuit.Circuit('3dcircuit',
                               circuit_params,
                               analysis_type='stationary',
                               from_file='False')
        rates = np.zeros_like(Next_array)
        for i in range(Next_array.shape[0]):
            circuit_params['Next']= Next_array[i]
            circ.alter_default_params(circuit_params)
            rates[i] = circ.th_rates
        h5.add_to_h5('results.h5',{label:{'rates_calc': rates}},
            'a', overwrite_dataset=True)
    else:
        rates = h5.load_h5('results.h5', label + '/rates_calc')
    return rates

def get_diffapprox_gain(rates0, params, Kstim, calc=False, read_old_results=False):
    folder = get_folder(params)
    label = 'plot_rates_gain' + str(Kstim) + folder

    if read_old_results:
        path = '/Users/bos/Documents/pittsburgh/disinhibitory_circuits/py/E-I-VIP/'
        g = h5.load_h5(path + 'results.h5', label + '/g')
        return g

    circuit_params = set_circuit_params(params)
    circuit_params['Next'] += np.array([Kstim,Kstim,0])

    if calc:
        circ = circuit.Circuit('3dcircuit',
                               circuit_params,
                               analysis_type='stationary',
                               from_file='False')
        rates = circ.th_rates
        # divide by 8 since this is the external firing rate in simulation
        g = (rates[0]-rates0[0])/(Kstim*8.)
        M = circ.ana.I*circ.ana.W
        dI = [M[0][i]*(rates[i]-rates0[i]) for i in range(3)]
        h5.add_to_h5('results.h5',{label:{'g': g}},
            'a', overwrite_dataset=True)
    else:
        g = h5.load_h5('results.h5', label + '/g')
    return g

def get_diffapprox_gain_coefficient(params, calc=False):
    folder = get_folder(params)
    label = 'plot_gain_coefficient' + folder
    circuit_params = set_circuit_params(params)

    if calc:
        circ = circuit.Circuit('3dcircuit',
                               circuit_params,
                               analysis_type='stationary',
                               from_file='False')
        rates_calc = circ.th_rates
        MH = circ.ana.create_MH0()
        P = np.linalg.inv(np.eye(3)-MH)
        betas = circ.ana.create_H0_mu()[:,0]
        factor = np.dot(P[:,:2], betas[:2])
        h5.add_to_h5('results.h5',{label:{
            'rates_calc': rates_calc, 'factor': factor}},
            'a', overwrite_dataset=True)
    else:
        rates_calc = h5.load_h5('results.h5', label + '/rates_calc')
        factor = h5.load_h5('results.h5', label + '/factor')
    return rates_calc, factor

def get_diffapprox_Next(params, rates, calc=False):
    folder = get_folder(params)
    label = 'plot_rates_Next_' + folder + str(rates)
    if rates[2]>0:
        label += 'rS_' + str(rates[2])
    circuit_params = set_circuit_params(params)

    if calc:
        print 'calculate Next'
        circ = circuit.Circuit('3dcircuit',
                               circuit_params,
                               analysis_type='stationary',
                               from_file='False')
        rates_calc = circ.th_rates
        M = circ.I*circ.W
        Next, betas = circ.get_external_input(rates)
        h5.add_to_h5('results.h5', {label:{'Next': Next, 'betas': betas, 'M': M}},
        'a', overwrite_dataset=True)
    else:
        Next = h5.load_h5('results.h5', label + '/Next')
        betas = h5.load_h5('results.h5', label + '/betas')
        M = h5.load_h5('results.h5', label + '/M')
    return Next, betas, M

def get_diffapprox_Next_for_rE_rI_grid(params, rE_array, rI_array, rS,
    calc_Next=False):
    folder = get_folder(params)
    label_Next = 'Next_betas_matrix_' + folder
    circuit_params = set_circuit_params(params)

    if calc_Next:
        print 'calculate Next'
        circ = circuit.Circuit('3dcircuit',
                               circuit_params,
                               analysis_type='stationary',
                               from_file='False')
        rates_calc = circ.th_rates
        M = circ.I*circ.W
        Next = np.zeros((len(rE_array), len(rI_array), 3))
        betas = np.zeros((len(rE_array), len(rI_array), 3))
        initial_Next = get_Next(params)
        for i,re in enumerate(rE_array):
            for j,ri in enumerate(rI_array):
                target_rates = np.array([re, ri, rS])
                circ.Next = initial_Next
                Next[i][j], betas[i][j] = circ.get_external_input(target_rates)
        h5.add_to_h5('results.h5', {label_Next:{'Next': Next, 'betas': betas,
            'M': M}}, 'a', overwrite_dataset=True)
    else:
        Next = h5.load_h5('results.h5', label_Next + '/Next')
        betas = h5.load_h5('results.h5', label_Next + '/betas')
        M = h5.load_h5('results.h5', label_Next + '/M')
    return Next, betas, M

def get_diffapprox_Next_for_rS_array(params, rE, rI, rS_array, calc_Next=False):
    folder = get_folder(params)
    label_Next = 'Next_betas_array_' + folder + str(rE) + str(rI)
    circuit_params = set_circuit_params(params)

    if calc_Next:
        print 'calculate Next'
        circ = circuit.Circuit('3dcircuit',
                               circuit_params,
                               analysis_type='stationary',
                               from_file='False')
        rates_calc = circ.th_rates
        M = circ.I*circ.W
        Next = np.zeros((len(rS_array), 3))
        betas = np.zeros((len(rS_array), 3))
        initial_Next = get_Next(params)
        for i,rS in enumerate(rS_array):
            target_rates = np.array([rE, rI, rS])
            circ.Next = initial_Next
            Next[i], betas[i] = circ.get_external_input(target_rates)
        h5.add_to_h5('results.h5', {label_Next:{'Next': Next, 'betas': betas,
            'M': M}}, 'a', overwrite_dataset=True)
    else:
        Next = h5.load_h5('results.h5', label_Next + '/Next')
        betas = h5.load_h5('results.h5', label_Next + '/betas')
        M = h5.load_h5('results.h5', label_Next + '/M')
    return Next, betas, M

def get_diffapprox_Next_for_pSE_array(params, rE, rI, rS, pSE_array,
    calc_Next=False):
    folder = get_folder(params)
    label_Next = 'Next_betas_pSE_array_' + folder + str(rE) + str(rI)
    circuit_params = set_circuit_params(params)

    if calc_Next:
        print 'calculate Next'
        circ = circuit.Circuit('3dcircuit',
                               circuit_params,
                               analysis_type='stationary',
                               from_file='False')
        rates_calc = circ.th_rates
        M = circ.I*circ.W
        Next = np.zeros((len(pSE_array), 3))
        betas = np.zeros((len(pSE_array), 3))
        initial_Next = get_Next(params)
        for i,pSE in enumerate(pSE_array):
            target_rates = np.array([rE, rI, rS])
            circ.Next = initial_Next
            params['pSE'] = pSE
            circ.alter_default_params({'p': get_connection_probabilities(params)})
            Next[i], betas[i] = circ.get_external_input(target_rates)
        h5.add_to_h5('results.h5', {label_Next:{'Next': Next, 'betas': betas,
            'M': M}}, 'a', overwrite_dataset=True)
    else:
        Next = h5.load_h5('results.h5', label_Next + '/Next')
        betas = h5.load_h5('results.h5', label_Next + '/betas')
        M = h5.load_h5('results.h5', label_Next + '/M')
    return Next, betas, M

def get_dmin(params, betas, M, calc=False):
    if not np.any(betas):
        return 0
    MH = np.dot(np.diag(betas),M)
    e, U = np.linalg.eig(MH)

    if e[0].real>=1 or e[1].real>=1 or e[2].real>=1:
        return 0
    ds = np.zeros(3)
    for i,eig in enumerate(e):
        if e[i].imag == 0:
            ds[i] = min(1, abs(1.-e[i].real))
        else:
            eigR = e[i].real
            eigI = abs(e[i].imag)
            p = (1-eigI*eigI-np.power(1-eigR,2))/eigI
            q = -1.
            x = -p/2+np.sqrt(p*p/4.-q)
            ds[i] = np.sqrt((np.power((1-eigR),2)+np.power((x-eigI),2))/(1+x*x))
    dist_min = np.min(ds)
    return dist_min

def get_stability_matrix(params, rE_array, rI_array, rS, calc_Next=False,
    calc_dmin=False):
    folder = get_folder(params)
    label_dmin = 'stability_matrix_' + folder

    Next, betas, M = get_diffapprox_Next_for_rE_rI_grid(params, rE_array,
        rI_array, rS, calc_Next=calc_Next)

    if calc_dmin:
        dmin = np.zeros((len(rE_array), len(rI_array)))
        for i,re in enumerate(rE_array):
            for j,ri in enumerate(rI_array):
                dmin[i][j] = get_dmin(params, betas[i][j], M, calc=calc_dmin)
        h5.add_to_h5('results.h5', {label_dmin:{'dmin': dmin}}, 'a',
            overwrite_dataset=True)
    else:
        dmin = h5.load_h5('results.h5', label_dmin + '/dmin')
    return dmin

def get_gain_matrix(params, rE_array, rI_array, rS, calc_Next=False,
    calc_gain=False):
    folder = get_folder(params)
    label = 'gain_matrix_' + folder

    Next, betas, M = get_diffapprox_Next_for_rE_rI_grid(params, rE_array,
        rI_array, rS, calc_Next=calc_Next)

    if calc_gain:
        gain = np.zeros((len(rE_array), len(rI_array)))
        for i,re in enumerate(rE_array):
            for j,ri in enumerate(rI_array):
                P = np.eye(3)-np.dot(np.diag(betas[i][j]),M)
                stim = betas[i][j]
                stim[2] = 0.
                gain[i][j] = np.dot(np.linalg.inv(P), np.array(stim))[0]
        h5.add_to_h5('results.h5',{label:{'gain': gain}}, 'a',
            overwrite_dataset=True)
    else:
        gain  = h5.load_h5('results.h5', label + '/gain')
    return gain

def get_eigs2d_matrix(params, rE_array, rI_array, rS, calc_Next=False,
    calc_eigs=False):
    folder = get_folder(params)
    label = 'eigs2d_matrix_' + folder

    Next, betas, M = get_diffapprox_Next_for_rE_rI_grid(params, rE_array,
        rI_array, rS, calc_Next=calc_Next)

    if calc_eigs:
        eigs = np.zeros((len(rE_array), len(rI_array),2), dtype=complex)
        for i,re in enumerate(rE_array):
            for j,ri in enumerate(rI_array):
                MH = np.dot(np.diag(betas[i][j][:2]),M[:2,:2])
                eigs[i][j], _ = np.linalg.eig(MH)
        h5.add_to_h5('results.h5',{label:{'eigs': eigs}}, 'a',
            overwrite_dataset=True)
    else:
        eigs  = h5.load_h5('results.h5', label + '/eigs')
    return eigs

def get_dr_dIm_matrix(params, rE_array, rI_array, rS, pES, pIS, sign,
    calc_Next=False, calc_dr_dIm=False):
    folder = get_folder(params)
    label = 'dr_dIm_matrix_' + folder + str(pES) + str(pIS) + str(sign)

    Next, betas, M = get_diffapprox_Next_for_rE_rI_grid(params, rE_array,
        rI_array, rS, calc_Next=calc_Next)

    if calc_dr_dIm:
        dr_dIm_matrix = np.zeros((len(rE_array), len(rI_array), 2))
        for i,re in enumerate(rE_array):
            for j,ri in enumerate(rI_array):
                betas[i][j][2] = 1.
                P = M.copy()
                P[0][2] = P[0][1]/0.1*pES
                P[1][2] = P[0][1]/0.1*pIS
                P = np.eye(3)-np.dot(np.diag(betas[i][j]),P)
                dr = np.dot(np.linalg.inv(P), np.array([0, 0, sign]))[0:2]
                dr_dIm_matrix[i][j] = dr/np.max(abs(dr))
        h5.add_to_h5('results.h5',{label:{'dr_dIm_matrix': dr_dIm_matrix}}, 'a',
            overwrite_dataset=True)
    else:
        dr_dIm_matrix  = h5.load_h5('results.h5', label + '/dr_dIm_matrix')
    return dr_dIm_matrix

def run_simulation(params, T=10000.0):
    sim_dict = {'t_sim': T, 'sim_resolution': 0.1,
        'data_path': os.path.join(os.getcwd(), 'data/'),
        'master_seed': 55, 'local_num_threads': 1,
        'overwrite_files': True, 'print_time': True}
    for key in sorted(params):
        sim_dict['data_path'] += key + str(params[key])

    net_dict['N_full'] = get_N(params)
    net_dict['conn_probs'] = np.array([[params['pEE'], params['pEI'], params['pES']],
                  [params['pIE'], params['pII'], params['pIS']],
                  [params['pSE'], params['pSI'], params['pSS']]])
    net_dict['K_ext'] = np.array([params['KEext']/2000., params['KIext']/2000.,
            params['KSext']/2000.])*params['N']
    net_dict['K_ext_inh'] = np.array([params['KEext_inh']/2000., params['KIext_inh']/2000.,
            params['KSext_inh']/2000.])*params['N']
    net_dict['PSP_e'] = 0.15*100./np.sqrt(params['pEE0']*params['N'])
    net_dict['PSP_mean_matrix'] =  netp.get_mean_PSP_matrix(
        net_dict['PSP_e'], net_dict['g'], len(net_dict['populations']))
    net_dict['PSP_std_matrix'] =  netp.get_std_PSP_matrix(
        net_dict['PSP_sd'], len(net_dict['populations']))

    if params['tuned_input']:
        stim_dict['tuned_input'] = True
        stim_dict['r_theta'] = params['r_theta']
        stim_dict['sigma_theta'] = params['sigma_theta']
        stim_dict['tuned_PV'] = params['tuned_PV']
        stim_dict['theta'] = params['theta']

    # copy all scripts to data_path
    if not os.path.exists(sim_dict['data_path']):
        os.makedirs(sim_dict['data_path'])
    files = ['helpers.py', 'network.py', 'network_params.py',
        'stimulus_params.py', 'create_figures.py', 'create_figures_helpers.py']
    for file in files:
        copyfile(file, sim_dict['data_path']+'/'+file)

    # Initialize the network and pass parameters to it.
    tic = time.time()
    net = network.Network(sim_dict, net_dict, stim_dict)
    toc = time.time() - tic
    print("Time to initialize the network: %.2f s" % toc)
    # Connect all nodes.
    tic = time.time()
    net.setup()
    toc = time.time() - tic
    print("Time to create the connections: %.2f s" % toc)
    # Simulate.
    tic = time.time()
    net.simulate()
    toc = time.time() - tic
    print("Time to simulate: %.2f s" % toc)
