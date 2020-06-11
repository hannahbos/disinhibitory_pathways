import numpy as np
import matplotlib.pylab as plt
import matplotlib.ticker as ticker

import create_figures_helpers as cfh

plt.rcParams['figure.dpi'] = 300
plt.rcParams['lines.markersize'] = 3.0
plt.rcParams['ps.useafm'] = False
plt.rcParams['ps.fonttype'] = 3
plt.rcParams['mathtext.fontset'] = "stix"
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['font.size'] = 9
plt.rcParams['legend.fontsize'] = 9

## default parameter ##
simtime = 10000.0 # Simulation time in ms
tmin = 300.0

colors = [(189/255.,44/255.,50/255.),
          (77/255.,144/255.,201/255.),
          (71/255.,158/255.,71/255.)]

def create_panels_figure_2(mode, run_simulation=False, calculate_meanfield=False,
    read_raw_sim_data=False):
    plt.rcParams['figure.figsize'] = (6.9, 1.7)

    nx = 6
    ny = 4
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.6, hspace=1.5, top=0.88,
                        bottom=0.24, left=0.02, right=0.91)
    if mode == 'panels_A':
        ax = [plt.subplot2grid((nx,ny), (0,1), rowspan=6),
              plt.subplot2grid((nx,ny), (1,2), rowspan=5),
              plt.subplot2grid((nx,ny), (0,3), rowspan=2),
              plt.subplot2grid((nx,ny), (2,3), rowspan=2),
              plt.subplot2grid((nx,ny), (4,3), rowspan=2),
              plt.subplot2grid((nx,ny), (0,2))]
    elif mode == 'panels_B':
        ax = [plt.subplot2grid((nx,ny), (0,1), rowspan=6),
              plt.subplot2grid((nx,ny), (0,2), rowspan=6),
              plt.subplot2grid((nx,ny), (0,3), rowspan=2),
              plt.subplot2grid((nx,ny), (2,3), rowspan=2),
              plt.subplot2grid((nx,ny), (4,3), rowspan=2)]

    ## parameter ##
    params = cfh.set_default_params()
    if mode == 'panels_A':
        params['pES'] = 0.0
        params['pIS'] = 0.1
        params['KEext_inh'] = 20
        time_interval_raster_plot = [2120,2170]
    elif mode == 'panels_B':
        params['pES'] = 0.1
        params['pIS'] = 0.0
        params['KEext_inh'] = 0
        time_interval_raster_plot = [2000,2050]
    params['pSE'] = 0.0

    # this is different for SIE and VSE such that the maximum value is one
    gain_max_calc = 0.0022
    # this is the same for both SIE and VSE to make them comparable
    gain_max_sim = 0.0058

    ## simulation ##
    # get data for rates and gain
    Kstim_low = 10 # change in external indegree for small perturbation
    if run_simulation:
        params_stim = params.copy()
        params_stim['KEext'] = params['KEext'] + Kstim_low
        params_stim['KIext'] = params['KIext'] + Kstim_low
    da_ext = 0.05
    if mode == 'panels_A':
        a_ext_sim = np.arange(0,0.71,da_ext)
    elif mode == 'panels_B':
        a_ext_sim = np.arange(0.5,1.01,da_ext)
    rates_sim = np.zeros((len(a_ext_sim),3))
    gain_sim = np.zeros_like(a_ext_sim)
    modulation_sim = np.zeros_like(a_ext_sim)
    for i,a in enumerate(a_ext_sim):
        if mode == 'panels_A':
            params['KSext_inh'] = 45*(1-a)
        elif mode == 'panels_B':
            params['KSext_inh'] = 45*a
        if run_simulation:
            cfh.run_simulation(params)
            if mode == 'panels_A':
                params_stim['KSext_inh'] = 45*(1-a)
            elif mode == 'panels_B':
                params_stim['KSext_inh'] = 45*a
            cfh.run_simulation(params_stim)
        modulation_sim[i] = 45*(a-a_ext_sim[0])
        rates_sim[i] = cfh.get_rates_sim(params, calc=read_raw_sim_data)
        gain_sim[i]= cfh.get_gain_sim(rates_sim[i], params, Kstim_low,
            calc=read_raw_sim_data)
    rates_sim = np.transpose(rates_sim)

    # get data for raster plot
    if mode == 'panels_A':
        a_ext_dotplot = [0.2,0.55,0.7]
    elif mode == 'panels_B':
        a_ext_dotplot = [0.5,0.6,0.9]
    modulation_dotplot = np.zeros_like(a_ext_dotplot)
    times = []
    gids = []
    for i,a in enumerate(a_ext_dotplot):
        if mode == 'panels_A':
            params['KSext_inh'] = 45*(1-a)
        elif mode == 'panels_B':
            params['KSext_inh'] = 45*a
        modulation_dotplot[i] = 45*(a-a_ext_sim[0])
        t, g = cfh.get_dotplot(params, time_interval_raster_plot,
            calc=read_raw_sim_data)
        times.append(t)
        gids.append(g)

    ## mean-field theory ##
    da_ext_calc = 0.01
    if mode == 'panels_A':
        a_ext_calc = np.arange(0,0.71,da_ext_calc)
    elif mode == 'panels_B':
        a_ext_calc = np.arange(0.5,1.01,da_ext_calc)
    rates_calc = np.zeros((len(a_ext_calc),3))
    gain_calc = np.zeros_like(a_ext_calc)
    modulation_calc = np.zeros_like(a_ext_calc)
    for i,a in enumerate(a_ext_calc):
        if mode == 'panels_A':
            params['KSext_inh'] = 45*(1-a)
        elif mode == 'panels_B':
            params['KSext_inh'] = 45*a
        modulation_calc[i] = 45*(a-a_ext_calc[0])
        rates_calc[i] = cfh.get_diffapprox_rates(params, calc=calculate_meanfield)
        gain_calc[i] = cfh.get_diffapprox_gain(rates_calc[i], params, Kstim_low,
            calc=calculate_meanfield)
    rates_calc = np.transpose(rates_calc)

    ## plot results ##
    # plot rates
    for i in range(3):
        ax[0].plot(modulation_calc, rates_calc[i], color=colors[i])
        ax[0].plot(modulation_sim[0], rates_sim[i][0], '.', color=colors[i])
        ax[0].plot(modulation_sim[1:], rates_sim[i][1:], '.', color=colors[i],
            zorder=10, clip_on=False)
    if mode == 'panels_A':
        ax[0].set_yticks([10,20])
    elif mode == 'panels_B':
        ax[0].set_yticks([5,10])
    ax[0].set_ylabel('rates' + r'$(\mathrm{Hz})$')
    # Hide the right and top spines
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    # plot gain
    ax[1].plot(modulation_calc, gain_calc/gain_max_calc, color='black')
    gain_sim = gain_sim/gain_max_sim
    ax[1].plot(modulation_sim, gain_sim, '.', color='black')
    if mode == 'panels_A':
        ax[5].plot(modulation_sim[-1], gain_sim[-1], '.', color='black',
            zorder=10, clip_on=False)
        ax[1].set_yticks([0,1])
        ax[5].set_ylim(4.0, 4.5)
        ax[5].set_yticks([4.3])
        ax[1].set_ylim([0.0,1.5])
    elif mode == 'panels_B':
        ax[1].plot(modulation_sim[-1], gain_sim[-1], '.', color='black',
            zorder=10, clip_on=False)
        ax[1].set_yticks([0,0.5,1.0])
    ax[1].set_ylabel('gain')

    # add extra panel for broken axis
    if mode == 'panels_A':
        box = ax[5].get_position()
        y0 = box.y0
        y0_new = 0.95*y0
        ax[5].set_position([box.x0, y0_new, box.width, box.height+y0-y0_new])

        # hide the spines between ax and ax2
        ax[5].spines['bottom'].set_visible(False)
        ax[1].spines['top'].set_visible(False)
        ax[5].xaxis.tick_top()
        ax[5].tick_params(labeltop='off')  # don't put tick labels at the top
        ax[1].xaxis.tick_bottom()
        ax[5].set_xticks([])
        d = .8  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        # kwargs = dict(transform=ax[1].transAxes, color='k', clip_on=False)
        eps = 0.01
        a = 0.7
        xmax = 45*a_ext_sim[-1]
        kwargs = dict(color='k', clip_on=False)
        ax[1].plot((-d, +d), (1.5+a*eps, 1.5-a*eps), **kwargs)
        ax[5].plot((-d, +d), (4.0+eps, 4.0-eps), **kwargs)
        ax[1].patch.set_visible(False)

    # add markers
    symbols = ['o', 'D', 's']
    for i in range(2):
        ax[1].plot(modulation_dotplot[i],
            gain_sim[np.argmin(abs(modulation_sim-modulation_dotplot[i]))],
            symbols[i], markersize=5, markeredgewidth=1, markeredgecolor='k',
            markerfacecolor='None')

    if mode == 'panels_A':
        ax[5].plot(modulation_dotplot[2],
            gain_sim[np.argmin(abs(modulation_sim-modulation_dotplot[2]))],
            symbols[2], markersize=5,  markeredgewidth=1, markeredgecolor='k',
            markerfacecolor='None', clip_on=False)
        # Hide the right and top spines
        ax[1].spines['right'].set_visible(False)
        ax[5].spines['right'].set_visible(False)
        ax[5].spines['top'].set_visible(False)
    elif mode == 'panels_B':
        ax[1].plot(modulation_dotplot[2],
            gain_sim[np.argmin(abs(modulation_sim-modulation_dotplot[2]))],
            symbols[2], markersize=5, markeredgewidth=1, markeredgecolor='k',
            markerfacecolor='None')
        # Hide the right and top spines
        ax[1].spines['right'].set_visible(False)
        ax[1].spines['top'].set_visible(False)

    for i in [0,1]:
        if mode == 'panels_A':
            ax[i].set_xlabel(r'$I_{\mathrm{mod}} (\mathrm{Hz})$')
            ax[i].set_xticks([10,30])
        elif mode == 'panels_B':
            ax[i].set_xlabel(r'$|I_{\mathrm{mod}}| (\mathrm{Hz})$')
            ax[i].set_xticks([10,20])
    for i in [0,1]:
        ax[i].set_xlim([modulation_sim[0],modulation_sim[-1]])
    if mode == 'panels_A':
        ax[5].set_xlim([modulation_sim[0],modulation_sim[-1]])

    # plot raster plot
    clrs=['0','0.5','0']
    markers = ['o', 'D', 's']
    for n in range(3):
        box = ax[2+n].get_position()
        ax[2+n].set_position([(1+0.04)*box.x0, box.y0,
            box.width+0.05*box.x0, box.height])
        for j in range(1):
            ax[2+n].plot(times[n][j], gids[n][j],
                'o', ms=1, color=colors[j])
            ax[2+n].plot(-1,0,markers[n], markersize=5, markeredgewidth=1,
                markeredgecolor='k', markerfacecolor='None',label=' ')
            ax[2+n].legend(loc=9, frameon=False, bbox_to_anchor=(-.06, 1.05))
        ax[2+n].set_ylim([5170-4136,5170])
        ax[2+n].set_xlim([0,50])
        ax[2+n].set_xticks([0, 25, 50])
        ax[2+n].set_yticks([])
        ax[2].set_yticklabels([r'$o$'])
        ax[4].set_xlabel('time' r'($\mathrm{ms}$)')
        ax[2].set_xticklabels([])
        ax[3].set_xticklabels([])

    plt.savefig('figure_2_' + mode + '.eps')

def create_heatmaps(figure, panel=None, calc=False):
    plt.rcParams['figure.figsize'] = (5.5, 1.7)

    nx = 1
    ny = 2
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.8, hspace=0.7, top=0.95,
                        bottom=0.24, left=0.08, right=0.85)
    ax= [plt.subplot2grid((nx,ny), (0,0)), plt.subplot2grid((nx,ny), (0,1))]

    ## parameter ##
    params = cfh.set_default_params()
    if figure == 'figure3':
        vectorfield = False
    elif figure == 'figure4':
        vectorfield = True
        if panel == 'panels_B':
            pES = 0.0
            pIS = 0.1
            sign = 1
        elif panel == 'panels_C':
            pES = 0.07
            pIS = 0.1
            sign = 1
        elif panel == 'panels_D':
            pES = 0.1
            pIS = 0.0
            sign = -1
        elif panel == 'panels_E':
            pES = 0.1
            pIS = 0.07
            sign = -1

    # remove SST from the network
    params['pES'] = 0.0
    params['pSE'] = 0.0
    params['pSI'] = 0.0
    params['pIS'] = 0.0
    params['pSS'] = 0.0

    dp = 0.1
    rE_array = np.arange(0.5,11.6,dp)
    rI_array = np.arange(0.5,11.6,dp)
    rS = 0.0

    # calculate stability
    d = cfh.get_stability_matrix(params, rE_array, rI_array, rS,
        calc_Next=calc, calc_dmin=calc)
    d = d/(np.max(d))

    # calculate gain
    g = cfh.get_gain_matrix(params, rE_array, rI_array, rS, calc_Next=False,
        calc_gain=calc)
    g = g/(np.max(g)*0.0017)
    g[g==0] = np.ones_like(g[g==0])
    g[g<0] = np.ones_like(g[g<0])

    matrices = [d,g]
    orig_cmaps = [plt.cm.Reds_r, plt.cm.Reds_r]
    labels = ['stability', 'gain']
    clim = [True, True]
    c0 = [0,0]
    c1 = [1,1]
    zticks = [[0,1], [0,1]]

    for k, d in enumerate(matrices):
        per = 0.9
        im1 = ax[k].pcolor(d, cmap=orig_cmaps[k])
        if clim[k]:
            im1.set_clim(c0[k],c1[k])
        box = ax[k].get_position()
        ax[k].set_position([box.x0, box.y0, box.width * per, box.height])
        cbar_ax = fig.add_axes([box.x0 + box.width, box.y0, 0.02, box.height])
        if zticks[k]:
            cb = fig.colorbar(im1, cax=cbar_ax, ticks=zticks[k])
        else:
            cb = fig.colorbar(im1, cax=cbar_ax)
        if figure == 'figure3':
            if k == 0:
                cb.ax.set_yticklabels([' low \n stability', '\n high \n stability'])
            elif k == 1:
                cb.ax.set_yticklabels([' low \n gain', '\n high \n gain'])
        elif figure == 'figure4':
            cb.set_label(labels[k])
        ax[k].set_xlabel(r'$r_{P}$')
        ax[k].set_ylabel(r'$r_{E}$')
        ax[k].set_xticks([5/dp, 10/dp])
        ax[k].set_yticks([5/dp, 10/dp])
        ax[k].set_xticklabels(['5', '10'])
        ax[k].set_yticklabels(['5', '10'])
        ax[k].set_xlim([(0.5-rE_array[0])/dp, (11.5-rE_array[0])/dp])
        ax[k].set_ylim([(0.5-rI_array[0])/dp, (11.5-rI_array[0])/dp])

        ## add isolines ##
        len_old = len(rE_array)
        rE_array = rE_array[rE_array<11.6]
        len_new = len(rE_array)
        # for some isolines we need to specify a minimum or maximum value of rE
        # or rp this can best be tested by adjusting the min and max values for
        # the zrange to a narrow range and checking the colors
        if k == 0:
            d0s = [[0.3], [0.5], [0.7], [0.9]]
            a = 1/float(len(d0s)+1)
            d0color = [str(i*a) for i in range(1,len(d0s)+1)]
        elif k == 1:
            d0s = [[0.1], [0.2], [0.3], [0.4,0,1],
                   [0.5,3.5], [0.6,3.5], [0.7,4]]
            a = 1/float(len(d0s)+2)
            d0color = [str(1-i*a) for i in range(2,len(d0s)+2)][::-1]

        for n,d0 in enumerate(d0s):
            rEs = rE_array.copy()
            indices = np.argmin(abs(d[:len_new,:len_new]-d0[0]), axis=1)
            rPs = rE_array[indices]
            rPmi, rEma, rPma = rI_array[0], rE_array[-1]*1.1, rI_array[-1]*1.1
            if len(d0) == 1:
                rEmi = rE_array[0]
            elif len(d0) == 2:
                rEmi = d0[1]
            rPs = rPs[(rEs<rEma)&(rEs>rEmi)]
            rEs = rEs[(rEs<rEma)&(rEs>rEmi)]
            rEs = rEs[(rPs<rPma)&(rPs>rPmi)]
            rPs = rPs[(rPs<rPma)&(rPs>rPmi)]
            ax[k].plot((rPs-rI_array[0])/dp, (rEs-rE_array[0])/dp, linewidth=1,
                color=d0color[n], zorder=1)

        eigs = cfh.get_eigs2d_matrix(params, rE_array, rI_array, rS,
            calc_Next=False, calc_eigs=calc)
        e_real = np.max(eigs.real, axis=2)
        indices = np.argmin(abs(e_real[:len_new,:len_new]-1), axis=1)[:len_new]
        rPs = rE_array[indices]
        ax[k].plot((rPs[rE_array>4]-rI_array[0])/dp,
            (rE_array[rE_array>4]-rE_array[0])/dp, linewidth=1, color='black',
            zorder=1)

        ## add arrows for SOM modulation ##
        if vectorfield:
            E = np.vstack([rE_array[1::15]/dp for i in range(len(rE_array[1::15]))])
            I = np.transpose(np.vstack([rI_array[1::15]/dp for i in range(len(rI_array[1::15]))]))
            dr_dIm_matrix = cfh.get_dr_dIm_matrix(params, rE_array, rI_array,
                rS, pES, pIS, sign, calc_Next=False, calc_dr_dIm=calc)
            drE = np.transpose(dr_dIm_matrix[1::15,1::15,0])
            drI = np.transpose(dr_dIm_matrix[1::15,1::15,1])

            eps = 0.006
            drE = drE*eps
            drI = drI*eps
            I -= rI_array[0]/dp
            E -= rE_array[0]/dp
            # indexing removes arrows that start in unstable region
            for i in range(2):
                ax[i].quiver(I[:,:3], E[:,:3], drI[:,:3], drE[:,:3], color='black', scale=0.1, zorder=2,
                    headwidth=4)
                ax[i].quiver(I[1:,3:6], E[1:,3:6], drI[1:,3:6], drE[1:,3:6], color='black', scale=0.1, zorder=2,
                    headwidth=4)
                ax[i].quiver(I[2:,6:], E[2:,6:], drI[2:,6:], drE[2:,6:], color='black', scale=0.1, zorder=2,
                    headwidth=4)

            scale_x = 1/dp
            scale_y = 1/dp
            ticks_x = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_x))
            ax[k].xaxis.set_major_formatter(ticks_x)
            ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_y))
            ax[k].yaxis.set_major_formatter(ticks_y)

    filename = figure
    if figure == 'figure3':
        filename += '_panels_A_C'
    if panel:
        filename += '_' + panel
    filename += '.eps'
    plt.savefig(filename)

def create_gain_along_stability_isolines(calc=False):
    plt.rcParams['figure.figsize'] = (5.5, 1.7)

    nx = 1
    ny = 2
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.8, hspace=0.7, top=0.95,
                        bottom=0.24, left=0.08, right=0.85)
    ax= [plt.subplot2grid((nx,ny), (0,1))]

    ## parameter ##
    params = cfh.set_default_params()
    # remove SST from the network
    params['pES'] = 0.0
    params['pSE'] = 0.0
    params['pSI'] = 0.0
    params['pIS'] = 0.0
    params['pSS'] = 0.0

    dp = 0.1
    rE_array = np.arange(0.5,11.6,dp)
    rI_array = np.arange(0.5,11.6,dp)
    rS = 0.0

    # calculate stability
    d = cfh.get_stability_matrix(params, rE_array, rI_array, rS,
        calc_Next=calc, calc_dmin=calc)
    d = d/(np.max(d))

    # calculate gain
    g = cfh.get_gain_matrix(params, rE_array, rI_array, rS, calc_Next=False,
        calc_gain=calc)
    g = g/(np.max(g)*0.0017)
    g[g==0] = np.ones_like(g[g==0])
    g[g<0] = np.ones_like(g[g<0])

    # rates at fixed stability
    rI_matrix = np.vstack([rE_array for i in range(len(rI_array))])
    d0ls = [0.895, 0.795, 0.695, 0.595]
    d0rs = [0.905, 0.805, 0.705, 0.605]
    # same colors as in heatmap (high stability isolines)
    d0colors = ['0.8', '0.7', '0.6', '0.5']
    for i in range(len(d0colors)):
        d0l = d0ls[i]
        d0r = d0rs[i]
        ax[0].plot(rI_matrix[(d>d0l) & (d<d0r)], g[(d>d0l) & (d<d0r)], '.',
            color=d0colors[i], zorder=10, clip_on=False)
    ax[0].set_xlabel(r'$r_{P}$')
    ax[0].set_ylabel('gain')
    ax[0].set_xlim([rI_array[0],rI_array[-1]])
    # Hide the right and top spines
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['top'].set_visible(False)

    plt.savefig('figure3_panel_D.eps')

def create_figure5(calc=False):
    plt.rcParams['figure.figsize'] = (6.5, 5.0)
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 12

    nx = 7
    ny = 4
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.8, hspace=2.8, top=0.98,
                        bottom=0.1, left=0.06, right=0.97)

    ax = [ plt.subplot2grid((nx,ny), (0,1), rowspan=2),
           plt.subplot2grid((nx,ny), (0,2), rowspan=2),
           plt.subplot2grid((nx,ny), (0,3), rowspan=2),
           plt.subplot2grid((nx,ny), (2,1), rowspan=2),
           plt.subplot2grid((nx,ny), (2,2), rowspan=2),
           plt.subplot2grid((nx,ny), (2,3), rowspan=2),
           plt.subplot2grid((nx,ny), (4,0), colspan=2, rowspan=3),
           plt.subplot2grid((nx,ny), (4,2), colspan=2, rowspan=3),
           plt.subplot2grid((nx,ny), (2,0), rowspan=2),
           plt.subplot2grid((nx,ny), (0,0), rowspan=2)]

    ax[8].axis('off')
    ax[9].axis('off')
    # Hide the right and top spines
    for i in range(8):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

    ## parameter ##
    params = cfh.set_default_params()
    params['KSext'] = 150

    # width of bars
    width = 0.8
    x = [1,2,3]

    # initial population rates
    rE = 1.0
    rI = 2.0
    rS = 7.0
    # inhibitory external input of SOM neurons for default and modulated state
    KSext_inh_default = 25
    KSext_inh_mod = 100
    # additional external input from stiumlus, same for E and PV
    Kext_stim = 120

    ## network without/with feedback from E to SOM, panels A/panels B ##
    for j,pSE in enumerate([0.0,0.05]):
        params['pSE'] = pSE
        params['KSext_inh'] = KSext_inh_default
        # get external input required for initial population rates
        Next, _, _ = cfh.get_diffapprox_Next(params, np.array([rE,rI,rS]),
            calc=calc)
        Kext = np.round(Next*2000./float(params['N'])).astype(int)
        params['KEext'], params['KIext'], params['KSext'] = Kext

        # no stimulus
        r_NSt = cfh.get_diffapprox_rates(params, calc=calc)
        # stimulus
        params['KEext'] += Kext_stim
        params['KIext'] += Kext_stim
        r_St = cfh.get_diffapprox_rates(params, calc=calc)
        # stimulus plos modulation
        params['KSext_inh'] = KSext_inh_mod
        r_M_St = cfh.get_diffapprox_rates(params, calc=calc)

        for i in range(3):
            ax[3*j+i].bar(x, [r_NSt[i], r_St[i], r_M_St[i]], width, color=colors[i])

    for i in [0,3]:
        ax[i].set_ylabel(r'$r_{\mathrm{E}}$')
        ax[i].set_ylim([0,9.3])
    for i in [1,4]:
        ax[i].set_ylabel(r'$r_{\mathrm{P}}$')
        ax[i].set_ylim([0,22.5])
    for i in [2,5]:
        ax[i].set_ylabel(r'$r_{\mathrm{S}}$')
        ax[i].set_ylim([0,14])
    for i in range(6):
        ax[i].set_xticks([1,2,3])
        ax[i].set_xlim([0.4,3.6])
        ax[i].set_xticklabels(['NSt','St','  M+St'])

    ## panels C ##
    rstarts = [[2.0,3.0],[2.0,2.0],[2.0,1.5]]
    color_starts = ['k', '0.4', '0.8']

    ## gain amplification depending on SOM rate
    rS_array = np.arange(1,10.05,0.1)
    alpha_ff = np.zeros((len(rstarts),len(rS_array)))
    alpha_rec = np.zeros((len(rstarts),len(rS_array)))
    for j,pSE in enumerate([0.0,0.05]):
        params['pSE'] = pSE
        for l,rstart in enumerate(rstarts):
            params['KSext_inh'] = KSext_inh_default
            Next_array_no_stim, _, _ = cfh.get_diffapprox_Next_for_rS_array(params,
                rstart[0], rstart[1], rS_array, calc_Next=calc)
            ## gain without modulation
            params['KSext_inh'] = KSext_inh_default
            # no stimulus
            r_NSt = np.transpose(np.vstack([rstart[0]*np.ones_like(rS_array),
                rstart[1]*np.ones_like(rS_array), rS_array]))

            # stimulus
            Next_array = Next_array_no_stim.copy()
            Next_array[:,0] += Kext_stim/2000.*params['N']
            Next_array[:,1] += Kext_stim/2000.*params['N']
            r_St = cfh.get_diffapprox_rates_Next_array(params, Next_array,
                calc=calc)
            # gain unmodulated
            gain_u = (r_St[:,0]-r_NSt[:,0])/float(Kext_stim)

            ## gain with modulation
            params['KSext_inh'] = KSext_inh_mod
            # no stimulus
            r_NSt = cfh.get_diffapprox_rates_Next_array(params,
                Next_array_no_stim, calc=calc)
            # stimulus
            r_St = cfh.get_diffapprox_rates_Next_array(params, Next_array,
                calc=calc)
            # gain modulated
            gain_mod = (r_St[:,0]-r_NSt[:,0])/float(Kext_stim)

            if j == 0:
                alpha_ff[l] = gain_mod/gain_u
            elif j == 1:
                alpha_rec[l] = gain_mod/gain_u

    for l,rstart in enumerate(rstarts):
        ax[6].plot(rS_array, alpha_rec[l]/alpha_ff[l], color=color_starts[l])
    ax[6].plot(rS_array, np.ones_like(rS_array), color='k', linestyle='dashed')
    ax[6].set_xlabel(r'$r_{\mathrm{S}}$')
    ax[6].set_ylabel(r'$\alpha^{\mathrm{rec}}/\alpha^{\mathrm{ff}}$')
    ax[6].set_xlim([2,10])
    ax[6].set_xticks([2,4,6,8,10])
    # ax[6].set_ylim([2.7,4.5])
    ax[6].set_ylim([0.9,4.5])
    ax[6].set_yticks([1,2,3,4])
    box = ax[6].get_position()
    ax[6].set_position([box.x0+0.02, box.y0,
        box.width, box.height])

    ## gain amplification depending on connection probability from E to SOM
    rS = 7.0
    pSE_array = np.arange(0.04,0.101,0.001)
    alpha_ff = np.zeros((len(rstarts)))
    alpha_rec = np.zeros((len(rstarts),len(pSE_array)))
    for l,rstart in enumerate(rstarts):
        for k,pSE in enumerate(pSE_array):
            params['pSE'] = pSE
            params['KSext_inh'] = KSext_inh_default
            params['KEext'], params['KIext'], params['KSext'] = 110, 100, 150
            Next, _, _ = cfh.get_diffapprox_Next(params,
                np.array([rstart[0], rstart[1], rS]), calc=calc)
            Kext_no_stim = np.round(Next*2000./float(params['N'])).astype(int)
            params['KEext'], params['KIext'], params['KSext'] = Kext_no_stim

            ## gain without modulation
            # no stimulus
            r_NSt = cfh.get_diffapprox_rates(params, calc=calc)
            # stimulus
            params['KEext'] += Kext_stim
            params['KIext'] += Kext_stim
            r_St = cfh.get_diffapprox_rates(params, calc=calc)
            # gain unmodulated
            gain_u = (r_St[0]-r_NSt[0])/float(Kext_stim)

            ## gain with modulation
            params['KSext_inh'] = KSext_inh_mod
            # no stimulus
            params['KEext'], params['KIext'], params['KSext'] = Kext_no_stim
            r_NSt = cfh.get_diffapprox_rates(params, calc=calc)
            # stimulus
            params['KEext'] += Kext_stim
            params['KIext'] += Kext_stim
            r_St = cfh.get_diffapprox_rates(params, calc=calc)
            # gain modulated
            gain_mod = (r_St[0]-r_NSt[0])/float(Kext_stim)

            # gain amplification
            if k == 0:
                alpha_ff[l] = gain_mod/gain_u
            alpha_rec[l][k] = gain_mod/gain_u

    for l,rstart in enumerate(rstarts):
        ax[7].plot(pSE_array/params['pIE'], np.ones_like(pSE_array), color='k',
            linestyle='dashed')
        ax[7].plot(pSE_array/params['pIE'], alpha_rec[l]/alpha_ff[l],
            color=color_starts[l])
        ax[7].set_xlabel(r'$J_{\mathrm{SE}}/J_{\mathrm{PE}}$')
        ax[7].set_xticks([1,1.5,2])
        ax[7].set_xlim([pSE_array[0]/params['pIE'],2])
        ax[7].set_yticks([1,1.5])

    plt.savefig('figure5.eps')

def create_figure6(run_simulation=False, read_raw_sim_data=False,
    calculate_meanfield=False):
    plt.rcParams['figure.figsize'] = (5.0, 5.0)
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['font.size'] = 11
    plt.rcParams['legend.fontsize'] = 11

    nx = 5
    ny = 3
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.7, hspace=0.5, top=0.99,
                        bottom=0.09, left=0.1, right=0.96)
    ax = [ plt.subplot2grid((nx,ny), (0,0)), plt.subplot2grid((nx,ny), (0,1)),
           plt.subplot2grid((nx,ny), (0,2)),
           plt.subplot2grid((nx,ny), (1,0)), plt.subplot2grid((nx,ny), (1,1)),
           plt.subplot2grid((nx,ny), (1,2)),
           plt.subplot2grid((nx,ny), (2,0)), plt.subplot2grid((nx,ny), (2,1)),
           plt.subplot2grid((nx,ny), (2,2)),
           plt.subplot2grid((nx,ny), (3,0)), plt.subplot2grid((nx,ny), (3,1)),
           plt.subplot2grid((nx,ny), (3,2)),
           plt.subplot2grid((nx,ny), (4,0)), plt.subplot2grid((nx,ny), (4,1)),
           plt.subplot2grid((nx,ny), (4,2))]

    # Hide the right and top spines
    for i in range(len(ax)):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

    colors = [[(255/255.,104/255.,97/255.),(189/255.,44/255.,50/255.)],
              [(136/255.,189/255.,255/255.),(77/255.,144/255.,201/255.)],
              [(156/255.,223/255.,131/255.),(71/255.,158/255.,71/255.)]]

    ## parameter ##
    params = cfh.set_default_params()
    params['KEext'] = 37
    params['KIext'] = 35
    params['KSext'] = 89
    KSext_inh_default = 25
    KSext_inh_modulated = 35
    KEext_offset = 37
    KIext_offset = 35
    r_theta = 400.0
    sigma_theta = 20.0
    tuned_PV = 1.0

    ## meanfield theory ##
    thetas = np.arange(48,133.0,6)
    factor_mod = np.zeros((len(thetas),3))
    # modulated state
    params['KSext_inh'] = KSext_inh_modulated
    for i,theta in enumerate(thetas):
        dtheta = theta-90.
        tuned_input = r_theta/8.*np.exp(-np.power(dtheta,2)/(2.*np.power(sigma_theta,2)))
        params['KEext'] = KEext_offset +tuned_input
        params['KIext'] = KIext_offset +tuned_PV*tuned_input
        _, factor_mod[i]  = cfh.get_diffapprox_gain_coefficient(params,
            calc=calculate_meanfield)
    # unmodulated state
    params['KSext_inh'] = KSext_inh_default
    factor = np.zeros((len(thetas),3))
    for i,theta in enumerate(thetas):
        dtheta = theta-90.
        tuned_input = r_theta/8.*np.exp(-np.power(dtheta,2)/(2.*np.power(sigma_theta,2)))
        params['KEext'] = KEext_offset+tuned_input
        params['KIext'] = KIext_offset +tuned_PV*tuned_input
        _, factor[i]  = cfh.get_diffapprox_gain_coefficient(params,
            calc=calculate_meanfield)

    for i in range(3):
        ax[3+i].plot(thetas, factor_mod[:,i]-factor[:,i], color=colors[i][0])
    ax[3].set_yticklabels(['2', '4'])
    ax[3].set_yticks([2e-6, 4e-6])
    ax[4].set_yticklabels(['5', '10'])
    ax[4].set_yticks([5e-6, 10e-6])
    ax[5].set_yticklabels(['-4', '-2'])
    ax[5].set_yticks([-4e-6, -2e-6])
    ax[3].annotate(r'$x10^{-6}$', xy=(.01, 1.02), xycoords='axes fraction')
    ax[4].annotate(r'$x10^{-6}$', xy=(.01, 1.02), xycoords='axes fraction')
    ax[5].annotate(r'$x10^{-6}$', xy=(.01, 1.02), xycoords='axes fraction')


    ## simulation: all neurons are tuned to the same theta ##
    thetas = np.arange(0.0,180,6)

    # modulated state
    rates = np.zeros((len(thetas),3))
    params['KSext_inh'] = KSext_inh_modulated
    for i,theta in enumerate(thetas):
        dtheta = theta-90.
        tuned_input = r_theta/8.*np.exp(-np.power(dtheta,2)/(2.*np.power(sigma_theta,2)))
        params['KEext'] = KEext_offset + tuned_input
        params['KIext'] = KIext_offset + tuned_PV*tuned_input
        if run_simulation:
            cfh.run_simulation(params)
        rates[i] = cfh.get_rates_sim(params, calc=read_raw_sim_data)
    for i in range(3):
        ax[i].plot(thetas, rates[:,i], color=colors[i][1])

    # unmodulated state
    params['KSext_inh'] = KSext_inh_default
    rates = np.zeros((len(thetas),3))
    for i,theta in enumerate(thetas):
        dtheta = theta-90.
        tuned_input = r_theta/8.*np.exp(-np.power(dtheta,2)/(2.*np.power(sigma_theta,2)))
        params['KEext'] = KEext_offset + tuned_input
        params['KIext'] = KIext_offset + tuned_PV*tuned_input
        if run_simulation:
            cfh.run_simulation(params)
        rates[i] = cfh.get_rates_sim(params, calc=read_raw_sim_data)
    for i in range(3):
        ax[i].plot(thetas, rates[:,i], color=colors[i][0])

    ## simulation: neurons are tuned to different thetas ##
    params['KEext'] = KEext_offset
    params['KIext'] = KIext_offset
    params['r_theta'] = r_theta
    params['sigma_theta'] = sigma_theta
    params['tuned_PV'] = tuned_PV
    params['tuned_input'] = True

    gids = [[800, 1600, 2000],
            [150, 300, 400],
            [0, 160, 440]]

    # unmodulated state
    params['KSext_inh'] = KSext_inh_default
    for pop in [0,1,2]:
        for i, gid in enumerate(gids[pop]):
            rates = []
            for theta in thetas:
                params['theta'] = theta
                if pop==0 and i==0 and run_simulation:
                    cfh.run_simulation(params)
                rates.append(cfh.get_rate_one_neuron_sim(params, pop, gid,
                    calc=read_raw_sim_data))
            ax[6+pop*3+i].plot(thetas, rates, color=colors[pop][0])

    # modulated state
    params['KSext_inh'] = KSext_inh_modulated
    for pop in [0,1,2]:
        for i, gid in enumerate(gids[pop]):
            rates = []
            for theta in thetas:
                params['theta'] = theta
                if pop==0 and i==0 and run_simulation:
                    cfh.run_simulation(params)
                rates.append(cfh.get_rate_one_neuron_sim(params, pop, gid,
                    calc=read_raw_sim_data))
            ax[6+pop*3+i].plot(thetas, rates, color=colors[pop][1])

    for i in range(15):
        ax[i].set_xlim([0,thetas[-1]])
        ax[i].set_xticks([0,90,180])
    for i in range(3,6):
        ax[i].set_xlabel(r'$\theta$')
    for i in range(12,15):
        ax[i].set_xlabel(r'$\theta$')
    for i in range(3):
        ax[i].set_xticklabels([])
    for i in range(6,12):
        ax[i].set_xticklabels([])
    ax[0].set_ylabel(r'$r_{\mathrm{E}}$')
    ax[1].set_ylabel(r'$r_{\mathrm{P}}$')
    ax[2].set_ylabel(r'$r_{\mathrm{S}}$')
    ax[3].set_ylabel(r'$\Delta g_{\mathrm{E}}$')
    ax[4].set_ylabel(r'$\Delta g_{\mathrm{P}}$')
    ax[5].set_ylabel(r'$\Delta g_{\mathrm{S}}$')
    ax[6].set_ylabel(r'$r_{\mathrm{E}_1}$')
    ax[7].set_ylabel(r'$r_{\mathrm{E}_2}$')
    ax[8].set_ylabel(r'$r_{\mathrm{E}_3}$')
    ax[9].set_ylabel(r'$r_{\mathrm{P}_1}$')
    ax[10].set_ylabel(r'$r_{\mathrm{P}_2}$')
    ax[11].set_ylabel(r'$r_{\mathrm{P}_3}$')
    ax[12].set_ylabel(r'$r_{\mathrm{S}_1}$')
    ax[13].set_ylabel(r'$r_{\mathrm{S}_2}$')
    ax[14].set_ylabel(r'$r_{\mathrm{S}_3}$')
    for i in range(3):
        ylims = ax[i].get_ylim()
        ax[i].set_ylim([0,ylims[1]])
    for i in range(6,15):
        ylims = ax[i].get_ylim()
        ax[i].set_ylim([0,ylims[1]])
    ax[0].set_yticks([0,2])
    ax[8].set_yticks([0,3])
    ax[12].set_yticks([0,2])
    ax[13].set_yticks([0,3])
    ax[14].set_yticks([0,3])

    for i in range(3,6):
        box = ax[i].get_position()
        ax[i].set_position([box.x0, box.y0+0.02, box.width, box.height])
    for i in range(6,9):
        box = ax[i].get_position()
        ax[i].set_position([box.x0, box.y0-2*0.03, box.width, box.height])
    for i in range(9,12):
        box = ax[i].get_position()
        ax[i].set_position([box.x0, box.y0-0.03, box.width, box.height])

    plt.savefig('figure6.eps')

def create_figure7(calc=False):
    plt.rcParams['figure.figsize'] = (6.9, 1.7)
    plt.rcParams['legend.fontsize'] = 6.5

    nx = 1
    ny = 3
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.6, hspace=1.5, top=0.9,
                        bottom=0.24, left=0.07, right=0.99)
    ax= [plt.subplot2grid((nx,ny), (0,0)), plt.subplot2grid((nx,ny), (0,1)),
         plt.subplot2grid((nx,ny), (0,2))]
    # Hide the right and top spines
    for i in range(len(ax)):
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)

    ## parameter ##
    params = cfh.set_default_params()
    params['KEext_inh'] = 10
    params['pIS'] = 0.1
    rates_init = [3.0, 7.7, 0.7]
    a_ext = np.arange(0,0.61,0.01)

    ## panel A ##
    colors_panel_I = [(152/255., 255/255., 132/255.),
                      (94/255., 156/255., 81/255.),
                      (54/255.,90/255.,47/255.)]
    labels_panel_I = [r'$p_{\mathrm{SS}}=0$', r'$p_{\mathrm{SS}}=0.05$',
        r'$p_{\mathrm{SS}}=0.1$']

    for j, pSS in enumerate([0.0, 0.05, 0.1]):
        params['pSS'] = pSS
        if pSS == 0.0:
            rE, rI, rS = rates_init
        if j == 0:
            params['KSext_inh'] = 45
            Next, _, _ = cfh.get_diffapprox_Next(params,
                np.array([rE,rI,rS]), calc=calc)
            Kext = Next*2000./float(params['N'])
            params['KEext'], params['KIext'], params['KSext'] = Kext
        rates = np.zeros((len(a_ext),3))
        modulation = []
        for i,a in enumerate(a_ext):
            params['KSext_inh'] = 45*(1-a)
            modulation.append(45*(a-a_ext[0]))
            rates[i]  = cfh.get_diffapprox_rates(params, calc=calc)
        rates = np.transpose(rates)
        ax[0].plot(modulation, rates[0], color=colors_panel_I[j],
            label=labels_panel_I[j])
    ax[0].set_xlim([modulation[0],modulation[-1]])
    ax[0].set_ylabel('rates' + r'$(\mathrm{Hz})$')
    ax[0].set_xlabel(r'$I_{\mathrm{mod}} (\mathrm{Hz})$')
    ax[0].set_xticks([10,20])
    ax[0].set_xlim([0,45*a_ext[-1]*1.0])
    ax[0].legend(handlelength=0.9)

    ## panel B ##
    params['pSS'] = 0.01
    rE, rI, rS = rates_init
    colors_panel_II = [(118/255., 185/255., 255/255.),
                      (93/255., 145/255., 198/255.),
                      (26/255.,42/255.,57/255.)]
    labels_panel_II = [r'$p_{\mathrm{II}}=0.07$', r'$p_{\mathrm{II}}=0.1$',
        r'$p_{\mathrm{II}}=0.13$']

    for j,pII in enumerate([0.07, 0.1, 0.13]):
        params['pII'] = pII
        params['KEext'] = 110
        params['KIext'] = 100
        params['KSext'] = 100
        params['KSext_inh'] = 45
        Next, _, M = cfh.get_diffapprox_Next(params,
            np.array([rE,rI,rS]), calc=calc)
        Kext = Next*2000./float(params['N'])
        params['KEext'], params['KIext'], params['KSext'] = Kext
        dmins = np.zeros_like(a_ext)
        modulation = []
        for i,a in enumerate(a_ext):
            params['KSext_inh'] = 45*(1-a)
            modulation.append(45*(a-a_ext[0]))
            r, betas = cfh.get_diffapprox_rates_betas(params, calc=calc)
            dmins[i] = cfh.get_dmin(params, betas, M, calc=calc)
        ax[1].plot(modulation, dmins, color=colors_panel_II[j],
            label=labels_panel_II[j])
    ax[1].set_xlim([modulation[0],modulation[-1]])
    ax[1].set_ylabel('stability')
    ax[1].set_xlabel(r'$I_{\mathrm{mod}} (\mathrm{Hz})$')
    ax[1].set_xticks([10,20])
    ax[1].set_xlim([0,45*a_ext[-1]*1.0])
    ax[1].legend(ncol=2, columnspacing=0.9, handlelength=0.9)
    ax[1].set_ylim([0.07,0.9])

    ## panel C ##
    params['pSS'] = 0.01
    params['pII'] = 0.1
    rE, rI, rS = rates_init
    labels_panel_III = [r'$p_{\mathrm{IS}}=0$', r'$p_{\mathrm{IS}}=0.07$']

    for j,pIS in enumerate([0.0, 0.07]):
        params['pIS'] = pIS
        params['KEext'] = 110
        params['KIext'] = 100
        params['KSext'] = 100
        params['KSext_inh'] = 45
        Next, _, M = cfh.get_diffapprox_Next(params,
            np.array([rE,rI,rS]), calc=calc)
        Kext = Next*2000./float(params['N'])
        params['KEext'], params['KIext'], params['KSext'] = Kext
        dmins = np.zeros_like(a_ext)
        modulation = []
        for i,a in enumerate(a_ext):
            params['KSext_inh'] = 45*(1-a)
            modulation.append(45*(a-a_ext[0]))
            r, betas = cfh.get_diffapprox_rates_betas(params, calc=calc)
            dmins[i] = cfh.get_dmin(params, betas, M, calc=calc)
        ax[2].plot(modulation, dmins, color=colors_panel_II[j],
            label=labels_panel_III[j])
    ax[2].set_xlim([modulation[0],modulation[-1]])
    ax[2].set_ylabel('stability')
    ax[2].set_xlabel(r'$I_{\mathrm{mod}} (\mathrm{Hz})$')
    ax[2].set_xticks([10,20])
    ax[2].set_xlim([0,45*a_ext[-1]*1.0])
    ax[2].legend(handlelength=0.9)
    ax[2].set_ylim([0.53,1])

    plt.savefig('figure7.eps')

if __name__ == "__main__":
    ### figure 2 ###
    ## create panels for the SOM-PV-E and the VIP-SOM-E pathway
    for panels in ['panels_A', 'panels_B']:
        create_panels_figure_2(panels, run_simulation=False,
            read_raw_sim_data=False, calculate_meanfield=False)

    ## figure 3 ###
    # panels A and C ##
    create_heatmaps('figure3', calc=False)
    # panels D ##
    create_gain_along_stability_isolines(calc=False)
    
    ### figure 4 ###
    for panels in ['panels_B', 'panels_C', 'panels_D', 'panels_E']:
        create_heatmaps('figure4', panels, calc=False)

    ### figure 5 ###
    create_figure5(calc=False)

    ### figure 6 ###
    create_figure6(run_simulation=False, read_raw_sim_data=False,
        calculate_meanfield=False)

    ### figure 7 ###
    create_figure7(calc=False)
