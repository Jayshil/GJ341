import numpy as np
import os
import juliet

# This file is to generate detrended lightcurves for all visits with residuals

extract = 'NRCLW'

# Input folder
pin = os.getcwd() + '/' + extract + '/Analysis/Joint'
pout = os.getcwd() + '/' + extract + '/Analysis/Figures'

# Visit Number:
instruments = np.array(['V1', 'V2', 'V3'])#, 'V4', 'V5'])

dataset = juliet.load(input_folder=pin)
res = dataset.fit(sampler='dynamic_dynesty')

for i in range(len(instruments)):
    print('Working on Visit:', instruments[i])
    
    tim7, fl7, fle7 = dataset.times_lc[instruments[i]], dataset.data_lc[instruments[i]], dataset.errors_lc[instruments[i]]
    model, model_uerr, model_derr, comps = res.lc.evaluate(instruments[i], return_err=True, return_components=True)#, all_samples=True)

    mflx = np.median(res.posteriors['posterior_samples']['mflux_' + instruments[i]])

    # Detrended flux
    fl9 = (fl7 - comps['lm']) * (1 + mflx)

    # Detrended error
    errs1 = np.median(res.posteriors['posterior_samples']['sigma_w_' + instruments[i]] * 1e-6)#(model_uerr-model_derr)/2
    fle9 = np.sqrt((errs1**2) + (fle7**2))

    # Residuals
    resid9 = (fl7-model)*1e6

    # Transit model
    tmodel = (model - comps['lm']) * (1 + mflx)

    # Saving results
    f11 = open(pout + '/Data_' + instruments[i] + '.dat', 'w')
    f11.write('# Time \t Flux \t Err \t Detrened flux \t Detrended err \t Resids (in ppm) \t Full model \t Transit model\n')
    for j in range(len(tim7)):
        f11.write(str(tim7[j]) + '\t' + str(fl7[j]) + '\t' + str(fle7[j]) + '\t' + \
                  str(fl9[j]) + '\t' + str(fle9[j]) + '\t' + str(resid9[j]) + '\t' +\
                  str(model[j]) + '\t' + str(tmodel[j]) + '\n')
    f11.close()