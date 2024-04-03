import numpy as np
import os
from jwst.pipeline import calwebb_detector1
from glob import glob
from jwst import datamodels
from path import Path
import time

# ----------------------------
# For the final analysis
# ----------------------------

# This file is to generate corrected data from uncal files:
# Steps: Stage 1 and 2 of the JWST pipeline, without dark correction and flat fielding
# Further steps: correcting errorbars for zeros and Nan and creating a bad-pixel map

segs = []
for i in range(13):
    if i < 9:
        segs.append('00' + str(i+1))
    else:
        segs.append('0' + str(i+1))

#segs = segs[-2:]
p1 = os.getcwd() + '/Data/V3_LW'
p2 = os.getcwd() + '/Groups/Stage1'    # To store RateInts files


for i in range(len(segs)):
    t1 = time.time()
    # Segment no:
    seg = segs[i]
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> --- Working on Segment ' + str(seg))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    # See if outputs are there or not
    #jw01981033001_04103_00001-seg001_nrcalong_uncal.fits
    f1 = Path(p2 + '/jw01981033001_04103_00001-seg' + seg + '_nrcalong_linearitystep.fits')

    if f1.exists():
        all_completed = True
    else:
        all_completed = False

    if not all_completed:
        print('>>>> --- Stage 1 processing Starts...')
        #jw02084001001_04103_00001-seg001_nrcalong_uncal.fits
        fname_uncal = glob(p1 + '/*seg' + seg + '_nrcalong_uncal.fits')[0]
        uncal = datamodels.RampModel(fname_uncal)
        times_bjd = uncal.int_times['int_mid_BJD_TDB']
        groupscale_results = calwebb_detector1.group_scale_step.GroupScaleStep.call(uncal, save_results=False)
        dq_results = calwebb_detector1.dq_init_step.DQInitStep.call(groupscale_results, save_results=False)
        saturation_results = calwebb_detector1.saturation_step.SaturationStep.call(dq_results,
                                                                                save_results=False)
        superbias_results = calwebb_detector1.superbias_step.SuperBiasStep.call(saturation_results,
                                                                                save_results=False)
        refpix_results = calwebb_detector1.refpix_step.RefPixStep.call(superbias_results,
                                                                       odd_even_columns=True,
                                                                       odd_even_rows=True,
                                                                       save_results=False)
        linearity_results = calwebb_detector1.linearity_step.LinearityStep.call(refpix_results,
                                                                                save_results=True,
                                                                                output_dir=p2)
        # Saving the badpixel mask (2D array, listing bad-pixels as 0, and good pixel as 1)
        darkdq = linearity_results.pixeldq
        np.save(p2 + '/bmap_seg' + seg + '.npy', darkdq)
        np.save(p2 + '/Times_bjd_seg' + seg + '.npy', times_bjd)
        print('>>>> --- Done (Stage 1 processing)!!')
    else:
        print('>>>> --- All is already done in Stage 1 processing!!')
    t2 = time.time()
    print('>>>> --- Total time taken: {:.4f} minutes'.format((t2-t1)/60))
    print('>>>> --- --- --- --- --- --- --- --- --- ---')
    print('>>>> ---  Segment ' + str(seg) + ' analysis completed!!')
    print('>>>> --- --- --- --- --- --- --- --- --- ---')