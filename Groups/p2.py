import numpy as np
import matplotlib.pyplot as plt
import os
import utils
from stark import SingleOrderPSF, optimal_extract
from jwst import datamodels
from glob import glob
from stark import reduce
import pickle
from tqdm import tqdm
from path import Path
from poetss import poetss
import time

# Necessary inputs
aprad1, aprad2 = 9,2        # Aperture radius in the first and second interation, respectively
xstart, xend = 800, 1900    # Start and end column number for spectral extraction
visit = 'V3'                # Visit numbers, can either be V1, V2, V3 or V4
group = 1
# Value of gain and readnoise
gain = 1.82      # e/DN (Units of data are in DN)
read = 7.267459  # e (Readnoise is different for different pixels, so, this is median readnoise)

sub_med_resid = True
nint, ncol = np.random.randint(0,92), np.random.randint(xstart,xend)      # Arbitrary Integration Number and Column number for figures

# Input and Output paths
pin = os.getcwd() + '/Groups/Stage1'
pout = os.getcwd() + '/Groups/Outputs/'
if not Path(pout + '/Figures').exists():
    os.mkdir(pout + '/Figures')

## Segment!!!
segs = []
for i in range(13):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

for seg in range(len(segs)):
    t1 = time.time()
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> --- Working on Segment ' + str(segs[seg]))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')

    f12 = Path(pout + '/Spectrum_cube_seg' + segs[seg] + '_' + visit + '.pkl')
    f13 = Path(pout + '/Traces_seg' + segs[seg] + '_' + visit + '.pkl')
    f14 = Path(pout + '/PSF_data_seg' + segs[seg] + '_' + visit + '.pkl')
    if f12.exists() and f13.exists() and f14.exists():
        print('>>>> --- The file already exists...')
        print('         Continuing to the next segment...')
        continue
    else:
        pass

    print('>>>> --- Loading the dataset...')
    fname_lin = glob(pin + '/*seg' + segs[seg] + '_nrcalong_linearitystep.fits')[0]
    linearity_step_data = datamodels.open(fname_lin)
    lin_res_data = linearity_step_data.data[:,group,:,:]
    ### Computing the errors (from Poisson noise and readnoise)
    poisson_noise_e = np.sqrt(np.abs(lin_res_data) * gain)   # Data is in DN, gain is in e/DN => data * gain = e
    read_noise_e = read
    total_nose_e = np.sqrt((poisson_noise_e**2) + (read_noise_e**2))
    lin_res_err = total_nose_e / gain                # Total noise is in e, gain is in e/DN => total_noise / gain = e / e/DN = DN
    bmap_lin = np.load(pin + '/bmap_seg' + segs[seg] + '.npy')
    mask = np.ones(bmap_lin.shape)
    mask[bmap_lin > 0] = 0.
    print('>>>> --- Done!!')

    print('>>>> --- Correcting errorbars (for zeros and NaNs)...')
    ## Correct errorbars
    med_err = np.nanmedian(lin_res_err.flatten())
    ## Changing Nan's and zeros in error array with median error
    corr_err1 = np.copy(lin_res_err)
    corr_err2 = np.where(lin_res_err != 0., corr_err1, med_err)                     # Replacing error == 0 with median error
    corrected_errs = np.where(np.isnan(lin_res_err) != True, corr_err2, med_err)    # Replacing error == Nan with median error
    print('>>>> --- Done!!')

    print('>>>> --- Creating a bad-pixel map...')
    ## Making a bad-pixel map
    mask_bp1 = np.ones(lin_res_data.shape)
    mask_bp2 = np.where(lin_res_err != 0., mask_bp1, 0.)                 # This will place 0 in mask where errorbar == 0
    mask_bp3 = np.where(np.isnan(lin_res_err) != True, mask_bp2, 0.)     # This will place 0 in mask where errorbar is Nan
    #mask_badpix = np.where(dq == 0., mask_bp3, 0.)                               # This will place 0 in mask where darkdq != 0
    mask_badpix = mask * mask_bp3                                                 # Adding those pixels which are identified as bad by the pipeline (and hence 0)

    ## Mask with cosmic rays
    ### Essentially this mask will add 0s in the places of bad pixels...
    mask_bcr = utils.identify_crays(lin_res_data, mask_badpix)
    non_msk_pt_fr = np.sum(mask_bcr) / (mask_bcr.shape[0] * mask_bcr.shape[1] * mask_bcr.shape[2])
    print('---- Total per cent of masked points: {:.4f} %'.format(100 * (1 - non_msk_pt_fr)))
    print('>>>> --- Done!!')

    print('>>>> --- Correcting data...')
    corrected_data = np.copy(lin_res_data)
    corrected_data[mask_bcr == 0] = np.nan
    for i in tqdm(range(corrected_data.shape[0])):
        corrected_data[i,:,:] = utils.replace_nan(corrected_data[i,:,:])
    print('>>>> --- Done!!')

    print('>>>> --- Column-by-column background subtraction...')
    bkg_corr_data1 = np.ones(corrected_data.shape)
    for integrations in tqdm(range(bkg_corr_data1.shape[0])):
        # Let's first create a mask!!
        mask = np.ones(corrected_data[0, :, :].shape)
        for i in range(mask.shape[1]):
            mask[int(10):int(60), int(i)] = 0.
        mask = mask * mask_bcr[integrations,:,:]
        bkg_corr_data1[integrations,:,:], _ = reduce.polynomial_bkg_cols(corrected_data[integrations,:,:], mask, 1, 5)
    print('>>>> --- Done!!')

    print('>>>> --- Row-by-row background subtraction...')
    bkg_corr_data = np.ones(corrected_data.shape)
    for integrations in tqdm(range(bkg_corr_data.shape[0])):
        ## Let's first create a mask
        mask = np.ones(corrected_data[0, :, :].shape)
        for i in range(mask.shape[1]):
            if i>650:
                mask[:,i] = np.zeros(len(mask[:,i]))
        mask = mask * mask_bcr[integrations,:,:]
        bkg_corr_data[integrations, :, :], _ = reduce.row_by_row_bkg_sub(bkg_corr_data1[integrations,:,:], mask)
    print('>>>> --- Done!!')

    
    print('>>>> --- Loading the dataset...')
    corrected_data = np.copy(bkg_corr_data)
    times_bjd = np.load(pin + '/Times_bjd_seg' + segs[seg] + '.npy')
    print('>>>> --- Done!!')
    print('')
    
    print('>>> --- --- --- Starting the extraction...')
    print('')
    
    print('>>>> --- Finding trace positions...')
    cent_cub1 = poetss.find_trace_cof(clean_cube=corrected_data[:,4:,xstart:xend], margin=5)
    trace1, dx1 = poetss.fit_multi_trace(cent_mat=cent_cub1, deg=2, clip=3)
    xpos = np.arange(xstart, xend, 1)

    ## Saving the results
    traces_pos = {}
    traces_pos['xpos'] = xpos
    traces_pos['median_trace'] = trace1
    traces_pos['jitter'] = dx1
    pickle.dump(traces_pos, open(pout + '/Traces_seg' + segs[seg] + '_' + visit + '.pkl','wb'))

    ypos2d = np.zeros((corrected_data.shape[0], len(xpos)))
    for i in range(ypos2d.shape[0]):
        ypos2d[i,:] = trace1 + dx1[i]
    print('>>>> --- Done!!')

    print('>>>> --- Initial 1D spline fitting and spectral extraction...')
    data1d = SingleOrderPSF(frame=corrected_data[:,4:,xpos[0]:xpos[-1]+1],\
                            variance=corrected_errs[:,4:,xpos[0]:xpos[-1]+1]**2,\
                            ord_pos=ypos2d, ap_rad=aprad1, mask=mask_bcr[:,4:,xpos[0]:xpos[-1]+1])
    psf_frame1d, psf_spline1d, _ = data1d.univariate_psf_frame(niters=3, oversample=2, clip=10000)

    ts1 = np.linspace(np.min(data1d.norm_array[:,0]), np.max(data1d.norm_array[:,0]), 1000)
    msk1 = np.asarray(data1d.norm_array[:,4], dtype=bool)
    plt.figure(figsize=(16/1.5, 9/1.5))
    plt.errorbar(data1d.norm_array[msk1,0], data1d.norm_array[msk1,1], fmt='.')
    plt.plot(ts1, psf_spline1d(ts1), c='k', lw=2., zorder=10)
    plt.xlabel('Distance from the trace')
    plt.ylabel('Normalised flux')
    plt.savefig(pout + '/Figures/1dSpline_Seg' + segs[seg] + '.png')

    spec1d, var1d = np.zeros((psf_frame1d.shape[0], psf_frame1d.shape[2])), np.zeros((psf_frame1d.shape[0], psf_frame1d.shape[2]))
    syth1d = np.zeros(psf_frame1d.shape)
    for inte in tqdm(range(spec1d.shape[0])):
        spec1d[inte,:], var1d[inte,:], syth1d[inte,:,:] = optimal_extract(psf_frame=psf_frame1d[inte,:,:],\
                                                                          data=corrected_data[inte,4:,xpos[0]:xpos[-1]+1],\
                                                                          variance=corrected_errs[inte,4:,xpos[0]:xpos[-1]+1]**2,\
                                                                          mask=mask_bcr[inte,4:,xpos[0]:xpos[-1]+1],\
                                                                          ord_pos=ypos2d[inte,:], ap_rad=aprad1)
    print('>>>> --- Done!!')

    print('>>>> --- Robust 2D spline fitting and spectral extraction...')
    for _ in range(2):
        data2 = SingleOrderPSF(frame=corrected_data[:,4:,xpos[0]:xpos[-1]+1],\
                               variance=corrected_errs[:,4:,xpos[0]:xpos[-1]+1]**2,\
                               ord_pos=ypos2d, ap_rad=aprad1, mask=mask_bcr[:,4:,xpos[0]:xpos[-1]+1],\
                               spec=spec1d)
        psf_frame2d, psf_spline2d, m_ = data2.bivariate_psf_frame(niters=3, oversample=2, knot_col=10, clip=10000)
        spec1d, var1d = np.zeros((psf_frame2d.shape[0], psf_frame2d.shape[2])), np.zeros((psf_frame2d.shape[0], psf_frame2d.shape[2]))
        syth1d = np.zeros(psf_frame2d.shape)
        aprad3 = aprad2 if _ == 1 else aprad1
        for inte in tqdm(range(spec1d.shape[0])):
            spec1d[inte,:], var1d[inte,:], syth1d[inte,:,:] = optimal_extract(psf_frame=psf_frame2d[inte,:,:],\
                                                                              data=corrected_data[inte,4:,xpos[0]:xpos[-1]+1],\
                                                                              variance=corrected_errs[inte,4:,xpos[0]:xpos[-1]+1]**2,\
                                                                              mask=mask_bcr[inte,4:,xpos[0]:xpos[-1]+1],\
                                                                              ord_pos=ypos2d[inte,:], ap_rad=aprad3)
        
        # Creating residual image
        # For creating residual image
        resid1 = np.zeros(syth1d.shape)
        resid1 = corrected_data[:,4:,xpos[0]:xpos[-1]+1] - syth1d
        
        # Whether to subtract median static residuals from the data or not!
        if sub_med_resid:
            med_resid_img = np.median(resid1, axis=0)
            corrected_data[:,4:,xstart:xend] = corrected_data[:,4:,xstart:xend] - med_resid_img[None,:,:]
    
    print('>>>> --- Done!!')

    print('>>>> --- Making some figures...')
    ## Saving figures to make sure that 2D spline fitting was working!!
    des_pts, cont_pts = utils.spln2d_func(ncol1=ncol-xpos[0], datacube=data2)
    fits_2d = psf_spline2d(cont_pts[0], cont_pts[1], grid=False)

    plt.figure(figsize=(16/1.5,9/1.5))
    plt.errorbar(des_pts[0], des_pts[2], fmt='.')
    plt.plot(cont_pts[0], fits_2d, 'k-')
    plt.plot(des_pts[0], psf_spline2d(des_pts[0], des_pts[1], grid=False), 'k.')
    plt.axvline(0., color='k', ls='--')
    plt.title('All frames, for Column ' + str(ncol))
    plt.savefig(pout + '/Figures/2D_Splines_Seg' + segs[seg] + '.png')

    ## Saving all extracted spectrum in a single file!
    plt.figure(figsize=(15,5))
    for i in range(spec1d.shape[0]):
        plt.plot(xpos, spec1d[i,:], 'k', alpha=0.5)
    plt.savefig(pout + '/Figures/All_spectra_Seg' + segs[seg] + '_v1.png')

    print('>>>> --- Done!!')
    
    print('>>>> --- Saving results...')
    dataset = {}
    dataset['spectra'] = spec1d
    dataset['variance'] = var1d
    dataset['resid'] = resid1
    dataset['times'] = times_bjd
    pickle.dump(dataset, open(pout + '/Spectrum_cube_seg' + segs[seg] + '_' + visit + '.pkl','wb'))
    print('>>>> --- Done!!')

    t2 = time.time()
    print('>>>> --- Total time taken: {:.4f} minutes'.format((t2-t1)/60))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> ---  Segment ' + str(segs[seg]) + ' analysis completed!!')
    print('>>>> --- --- --- --- --- --- --- --- --- ---')