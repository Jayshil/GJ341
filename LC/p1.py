import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import pickle
import juliet
import os
import utils as utl
import matplotlib.gridspec as gd
from poetss import poetss
import multiprocessing
multiprocessing.set_start_method('fork')

# This file is to analyse all white light lightcurve together -- all parameters but the eclipse depth
# would be the same for all visits: for `stark` and `transitspectroscopy`

# Output folder
pout = os.getcwd() + '/NRCLW/Analysis/Joint'

visits = ['V1', 'V2', 'V3']
## List of segments
segs = []
for i in range(13):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

# Data files
tim, fl, fle = {}, {}, {}
lin_pars = {}

for vis in range(len(visits)):
    # Location of the spectrum files
    pin = os.getcwd() + '/NRCLW/Outputs/' + visits[vis]# + '_v2'

    # Loading the spectra...
    lc_all, lc_err_all = [], []
    times = np.array([])
    for i in range(len(segs)):
        dataset1 = pickle.load(open(pin + '/Spectrum_cube_seg' + segs[i] + '_' + visits[vis] + '.pkl', 'rb'))
        lc_all.append(dataset1['spectra'])
        lc_err_all.append(dataset1['variance'])
        times = np.hstack((times, dataset1['times']))
    ## A giant cube with all lightcurves and their errors
    lc1, lc_err1 = np.vstack(lc_all), np.vstack(lc_err_all)
    lc_err1 = np.sqrt(lc_err1)

    ## White-light lightcurve
    wht_light_lc, wht_light_err = poetss.white_light(lc1, lc_err1)
    #wht_light_lc, wht_light_err = utl.white_light_by_sum(lc1, lc_err1)

    # And the final lightcurve
    tim9, fl9, fle9 = times, wht_light_lc, wht_light_err
    tim9 = tim9 + 2400000.5

    # Removing Nan values
    tim7, fl7, fle7 = tim9[~np.isnan(fl9)], fl9[~np.isnan(fl9)], fle9[~np.isnan(fl9)]

    # Outlier removal
    msk2 = utl.outlier_removal(tim7, fl7, fle7, clip=5, msk1=False)
    tim7, fl7, fle7 = tim7[msk2], fl7[msk2], fle7[msk2]

    # Saving them!
    tim[visits[vis]], fl[visits[vis]], fle[visits[vis]] = tim7, fl7/np.median(fl7), fle7/np.median(fl7)

    # Linear regressors
    lins = np.vstack([tim7-np.mean(tim7), (tim7-np.mean(tim7))**2])
    lin_pars[visits[vis]] = np.transpose(lins)

# Some planetary parameters
per, per_err = 7.576863, 0.01
bjd0, bjd0_err = 2459301.771, 0.002
ar, ar_err = 24.50, np.sqrt((2.07**2) + (4.01**2))
cycle = round((tim[visits[-1]][0]-bjd0)/per)
tc1 = np.random.normal(bjd0, bjd0_err, 100000) + (cycle*np.ones(100000)*per)

## Priors
### Planetary parameters
par_P = ['P_p1', 't0_p1', 'p_p1_' + '_'.join(visits), 'b_p1', 'q1_' + '_'.join(visits), 'q2_' + '_'.join(visits), 'ecc_p1', 'omega_p1', 'a_p1']
dist_P = ['fixed', 'normal', 'uniform', 'uniform', 'uniform', 'uniform', 'fixed', 'fixed', 'loguniform']
hyper_P = [per, [np.median(tc1), np.std(tc1)], [0., 1.], [0., 1.], [0., 1.], [0., 1.], 0., 90., [1., 100.]]


### Instrumental and linear parameters
par_ins, dist_ins, hyper_ins = [], [], []
par_lin, dist_lin, hyper_lin = [], [], []
for i in range(len(visits)):
    ## Instrumental parameters
    par_ins = par_ins + ['mdilution_' + visits[i], 'mflux_' + visits[i], 'sigma_w_' + visits[i]]
    dist_ins = dist_ins + ['fixed', 'normal', 'loguniform']
    hyper_ins = hyper_ins + [1.0, [0., 0.1], [0.1, 10000.]]
    ## Linear parameters
    for j in range(lin_pars[visits[i]].shape[1]):
        par_lin.append('theta' + str(j) + '_' + visits[i])
        dist_lin.append('uniform')
        hyper_lin.append([-1., 1.])

# Total priors
par_tot = par_P + par_ins + par_lin
dist_tot = dist_P + dist_ins + dist_lin
hyper_tot = hyper_P + hyper_ins + hyper_lin

priors = juliet.utils.generate_priors(par_tot, dist_tot, hyper_tot)

## And fitting
dataset = juliet.load(priors=priors, t_lc=tim, y_lc=fl, yerr_lc=fle, linear_regressors_lc=lin_pars, #GP_regressors_lc=gp_pars
                      out_folder=pout)
res = dataset.fit(sampler = 'dynamic_dynesty', nthreads=8)

for i in range(len(visits)):
    instrument = visits[i]
    # Some plots
    model = res.lc.evaluate(instrument)

    # Binned datapoints
    tbin, flbin, flebin, _ = utl.lcbin(time=tim[instrument], flux=fl[instrument], binwidth=0.003)

    # Let's make sure that it works:
    fig = plt.figure(figsize=(16,9))
    gs = gd.GridSpec(2,1, height_ratios=[2,1])

    # Top panel
    ax1 = plt.subplot(gs[0])
    ax1.errorbar(tim[instrument], fl[instrument], yerr=fle[instrument], fmt='.', alpha=0.3)
    ax1.errorbar(tbin, flbin, yerr=flebin, fmt='o', color='red', zorder=10)
    ax1.plot(tim[instrument], model, c='k', zorder=100)
    ax1.set_ylabel('Relative Flux')
    ax1.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))
    ax1.xaxis.set_major_formatter(plt.NullFormatter())

    # Bottom panel
    ax2 = plt.subplot(gs[1])
    ax2.errorbar(tim[instrument], (fl[instrument]-model)*1e6, yerr=fle[instrument]*1e6, fmt='.', alpha=0.3)
    ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
    ax2.set_ylabel('Residuals (ppm)')
    ax2.set_xlabel('Time (BJD)')
    ax2.set_xlim(np.min(tim[instrument]), np.max(tim[instrument]))

    plt.savefig(pout + '/full_model_' + instrument + '.png')

    residuals = fl[instrument]-model
    rms, stderr, binsz = utl.computeRMS(residuals, binstep=1)
    normfactor = 1e-6

    print(rms[0] / normfactor)

    plt.figure(figsize=(8,6))
    plt.plot(binsz, rms / normfactor, color='black', lw=1.5,
                    label='Fit RMS', zorder=3)
    plt.plot(binsz, stderr / normfactor, color='red', ls='-', lw=2,
                    label=r'Std. Err. ($1/\sqrt{N}$)', zorder=1)
    plt.xlim(0.95, binsz[-1] * 2)
    plt.ylim(stderr[-1] / normfactor / 2., stderr[0] / normfactor * 2.)
    plt.xlabel("Bin Size (N frames)", fontsize=14)
    plt.ylabel("RMS (ppm)", fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='best')
    plt.savefig(pout + '/alan_deviation_' + instrument + '.png')

utl.corner_plot(pout, False)