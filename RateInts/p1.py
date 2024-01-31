import numpy as np
from jwst import datamodels
import multiprocessing
multiprocessing.set_start_method('fork')
from jwst.pipeline import calwebb_detector1
from glob import glob
from path import Path
from tqdm import tqdm
from stark import reduce
import os
import utils
import time

# ------------------------------------------
# For the final analysis: NIRCam LW data
# ------------------------------------------

# This file is to generate corrected data from uncal files:
# Steps: Stage 1 of the JWST pipeline (refpix step and group level background subtraction)
# Further steps: correcting errorbars for zeros and Nan and creating a bad-pixel map

segs = []
for i in range(13):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))
visit = 'V1'
p1 = os.getcwd() + '/Data/' + visit + '_LW'
p2 = os.getcwd() + '/RateInts/Corr_NRCLW' + visit    # To store corrected files

for i in range(len(segs)):
    t1 = time.time()
    # Segment no:
    seg = segs[i]
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> --- Working on Segment ' + str(seg))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    # See if outputs are there or not
    f1 = Path(p2 + '/Corrected_data_seg' + seg + '.npy')
    f2 = Path(p2 + '/Corrected_errors_seg' + seg + '.npy')
    f3 = Path(p2 + '/Mask_bcr_seg' + seg + '.npy')
    f4 = Path(p2 + '/Times_bjd_seg' + seg + '.npy')

    if f1.exists() and f2.exists() and f3.exists() and f4.exists():
        all_completed = True
    else:
        all_completed = False
    
    if not all_completed:
        print('>>>> --- Stage 1 processing Starts...')
        fname = glob(p1 + '/*' + seg + '_nrcalong_uncal.fits')[0]
        uncal = datamodels.RampModel(fname)
        times_bjd = uncal.int_times['int_mid_BJD_TDB']

        # Stage 1 pipeline
        det1 = calwebb_detector1.Detector1Pipeline.call(uncal,\
                                                        steps={'jump' : {'skip' : True},\
                                                               'dark_current' : {'skip' : True},
                                                               'ramp_fit' : {'maximum_cores' : 'half'}},\
                                                        output_dir=os.getcwd() + '/RateInts/Ramp_NRCLW' + visit, save_results=True)

        # Data quality mask
        dq = det1.dq
        mask = np.ones(dq.shape)
        mask[dq > 0] = 0.

        # Loading rate-ints files
        fname_rate = glob(os.getcwd() + '/RateInts/Ramp_NRCLW' + visit + '/*' + seg + '_nrcalong_rateints.fits')[0]
        rate_ints = datamodels.open(fname_rate)

        ## Time
        times_bjd = uncal.int_times['int_mid_BJD_TDB']

        print('>>>> --- Stage 1 processing completed!!...')
        print('>>>> --- Additional correction to the data...')
        print('>>>> --- Using the rateints file: ' + fname_rate.split('/')[-1])

        print('>>>> --- Correcting errorbars (for zeros and NaNs)...')
        ## Correct errorbars
        med_err = np.nanmedian(rate_ints.err.flatten())
        ## Changing Nan's and zeros in error array with median error
        corr_err1 = np.copy(rate_ints.err)
        corr_err2 = np.where(rate_ints.err != 0., corr_err1, med_err)                     # Replacing error == 0 with median error
        corrected_errs = np.where(np.isnan(rate_ints.err) != True, corr_err2, med_err)    # Replacing error == Nan with median error
        print('>>>> --- Done!!')

        print('>>>> --- Creating a bad-pixel map...')
        ## Making a bad-pixel map
        mask_bp1 = np.ones(rate_ints.data.shape)
        mask_bp2 = np.where(rate_ints.err != 0., mask_bp1, 0.)                 # This will place 0 in mask where errorbar == 0
        mask_bp3 = np.where(np.isnan(rate_ints.err) != True, mask_bp2, 0.)     # This will place 0 in mask where errorbar is Nan
        #mask_badpix = np.where(dq == 0., mask_bp3, 0.)                               # This will place 0 in mask where darkdq != 0
        mask_badpix = mask * mask_bp3                                                 # Adding those pixels which are identified as bad by the pipeline (and hence 0)

        ## Mask with cosmic rays
        ### Essentially this mask will add 0s in the places of bad pixels...
        mask_bcr = utils.identify_crays(rate_ints.data, mask_badpix)
        non_msk_pt_fr = np.sum(mask_bcr) / (mask_bcr.shape[0] * mask_bcr.shape[1] * mask_bcr.shape[2])
        print('---- Total per cent of masked points: {:.4f} %'.format(100 * (1 - non_msk_pt_fr)))
        print('>>>> --- Done!!')

        print('>>>> --- Correcting data...')
        corrected_data = np.copy(rate_ints.data)
        corrected_data[mask_bcr == 0] = np.nan
        for i in range(corrected_data.shape[0]):
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

        np.save(p2 + '/Corrected_data_seg' + seg + '.npy', bkg_corr_data)
        np.save(p2 + '/Corrected_errors_seg' + seg + '.npy', corrected_errs)
        np.save(p2 + '/Mask_bcr_seg' + seg + '.npy', mask_bcr)
        np.save(p2 + '/Times_bjd_seg' + seg + '.npy', times_bjd)
    else:
        print('>>>> --- All is already done!!')
        print('>>>> --- Moving on to the next segment!')
    t2 = time.time()
    print('>>>> --- Total time taken: {:.4f} minutes'.format((t2-t1)/60))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> ---  Segment ' + str(seg) + ' analysis completed!!')
    print('>>>> --- --- --- --- --- --- --- --- --- ---')