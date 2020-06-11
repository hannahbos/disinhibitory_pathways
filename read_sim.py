import numpy as np
import glob

path_base = './data'

def get_gids(folder):
    path = path_base + folder
    data = np.loadtxt(path + '/population_GIDs.dat')
    gid_mins = np.transpose(data)[0].astype(int)
    pop_sizes = np.transpose(data)[1].astype(int)-gid_mins+1
    return gid_mins, pop_sizes

def read_one_pop(folder, pop, T, tmin, gid_lims=None):
    path = path_base + folder
    gid_mins, pop_sizes = get_gids(folder)
    nr_gids = pop_sizes[pop]
    gid_min = gid_mins[pop]
    gid_max = gid_mins[pop] + nr_gids - 1
    if gid_lims is not None:
        gid_min = gid_min + gid_lims[0]
        gid_max = gid_min + gid_lims[1]
    spikes = [[] for i in range(nr_gids)]
    filestart = path + 'spikes-pop' + str(pop) + '*'
    filelist = glob.glob(filestart)
    for filename in filelist:
        input_file = open(filename,'r')
        for index,line in enumerate(input_file):
            data = map(float, line.split())
            if int(data[0])>=gid_min and int(data[0])<gid_max:
                if data[1]>tmin and data[1]<T:
                    spikes[int(data[0])-gid_min].append(data[1])
        input_file.close()
    return spikes

def read_one_neuron(folder, pop, T, tmin, gid):
    path = path_base + folder
    gid_mins, pop_sizes = get_gids(folder)
    gid += gid_mins[pop]
    spikes = []
    filestart = path + 'spikes-pop' + str(pop) + '*'
    filelist = glob.glob(filestart)
    for filename in filelist:
        input_file = open(filename,'r')
        for index,line in enumerate(input_file):
            data = map(float, line.split())
            if int(data[0])==gid:
                if data[1]>tmin and data[1]<T:
                    spikes.append(data[1])
        input_file.close()
    return spikes

# gids limit is specified as an intervall, [a,b], returning data for the
# gids a+gid_min up to b+gid_min
def get_data_dotplot(folder, tmin, tmax, pop):
    path = path_base + folder
    gid_mins, pop_sizes = get_gids(folder)
    gid_min = gid_mins[pop]
    spikes = []
    gids = []
    filestart = path + 'spikes-pop' + str(pop) + '*'
    filelist = glob.glob(filestart)
    for filename in filelist:
        input_file = open(filename,'r')
        for index,line in enumerate(input_file):
            data = map(float, line.split())
            if data[1]>=tmin and data[1]<=tmax:
                spikes.append(data[1])
                gids.append(data[0])
        input_file.close()
    return spikes, gids

def get_all_rates(folder, pop, T, tmin):
    all_spike_times = read_one_pop(folder, pop, T, tmin)
    all_rates = [1000*len(spike_times)/(T-tmin) for spike_times in all_spike_times]
    return all_rates

def get_rate_one_neuron(folder, pop, gid, T, tmin):
    spike_times = np.asarray(read_one_neuron(folder, pop, T, tmin, gid))
    return 1000*len(spike_times)/(T-tmin)
