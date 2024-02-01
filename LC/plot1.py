import numpy as np
import matplotlib.pyplot as plt
import pickle
from glob import glob
from path import Path
import os
import utils as utl
import matplotlib.gridspec as gd
from exotoolbox.utils import tdur
import juliet
from astropy.stats import mad_std

# This file is to make plots for a full eclipse model and detrended eclipse model
# For white-light lightcurve

extract = 'NRCLW'

# Input folder
pin = os.getcwd() + '/' + extract + '/Analysis/Joint'
pout = os.getcwd() + '/' + extract + '/Analysis/Figures'
if not Path(pout).exists():
    os.mkdir(pout)

# Visit Number:
instrument = 'V1'

# Loading posteriors!
f12_post = glob(pin + '/*.pkl')[0]
post = pickle.load(open(f12_post, 'rb'))
post1 = post['posterior_samples']
## Best fitted eclipse paramters
t0, per = post1['t0_p1'], 7.576863
bb, ar1 = post1['b_p1'], post1['a_p1']
rprs = post1['p_p1_V1_V2_V3']
q1, q2 = post1['q1_V1_V2_V3'], post1['q2_V1_V2_V3']

dep_qua = juliet.utils.get_quantiles((rprs**2) * 1e6)
dep_med, dep_up, dep_do = dep_qua[0], dep_qua[1]-dep_qua[0], dep_qua[0]-dep_qua[2]

# Loading the dataset
## Raw data
tim, fl, fle, fl_det, fle_det, resid, model, tmodel = np.loadtxt(pout + '/Data_' + instrument + '.dat',\
                                                                 usecols=(0,1,2,3,4,5,6,7), unpack=True)
## Binned data
### For un-detrended data
tbin, flbin, flebin, _ = utl.lcbin(time=tim, flux=fl, binwidth=0.0005)
_, resbin, resebin, _ = utl.lcbin(time=tim, flux=resid, binwidth=0.0005)

tbin2, flbin2, flebin2, _ = utl.lcbin(time=tim, flux=fl, binwidth=0.005)
_, resbin2, resebin2, _ = utl.lcbin(time=tim, flux=resid, binwidth=0.005)

### For detrended data
tbin_det, flbin_det, flebin_det, _ = utl.lcbin(time=tim, flux=fl_det, binwidth=0.0005)
tbin_det2, flbin_det2, flebin_det2, _ = utl.lcbin(time=tim, flux=fl_det, binwidth=0.005)

# For Full model
fig = plt.figure(figsize=(16/1.5,9/1.5))
gs = gd.GridSpec(2,1, height_ratios=[2,1])

# Top panel
ax1 = plt.subplot(gs[0])
ax1.errorbar(tim, fl, fmt='.', c='cornflowerblue', alpha=0.25)
ax1.errorbar(tbin, flbin, yerr=flebin, fmt='.', c='gray', alpha=0.7, zorder=50)
ax1.errorbar(tbin2, flbin2, yerr=flebin2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax1.plot(tim, model, c='k', zorder=100)
ax1.set_ylabel('Relative Flux', fontsize=14)
ax1.set_xlim(np.min(tim), np.max(tim))
ax1.set_ylim(1-0.00075,1+0.001)
ax1.text(tim[100], 1-0.0006,\
         r'Transit Depth: $' + str(int(dep_med)) + '^{+' + str(int(dep_up)) + '} _{-' + str(int(dep_do)) +'}$ ppm',\
         fontsize=15, fontweight='bold', color='maroon', zorder=1000)
plt.setp(ax1.get_xticklabels(), fontsize=12)
plt.setp(ax1.get_yticklabels(), fontsize=12)
ax1.xaxis.set_major_formatter(plt.NullFormatter())
ax1.set_title('Full lightcurve for Visit ' + str(instrument[-1]), fontsize=15)

# Bottom panel
ax2 = plt.subplot(gs[1])
ax2.errorbar(tim, resid, c='cornflowerblue', fmt='.', alpha=0.25)
ax2.errorbar(tbin, resbin, yerr=resebin, c='gray', fmt='.', alpha=0.7, zorder=50)
ax2.errorbar(tbin2, resbin2, yerr=resebin2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax2.axhline(y=0.0, c='black', ls='--', zorder=100)
ax2.set_ylabel('Residuals (ppm)', fontsize=14)
ax2.set_xlabel('Time (BJD)', fontsize=14)
ax2.set_ylim(-250,250)
ax2.set_xlim(np.min(tim), np.max(tim))
plt.setp(ax2.get_xticklabels(), fontsize=12)
plt.setp(ax2.get_yticklabels(), fontsize=12)
#plt.show()
plt.savefig(pout + '/Full_LC_' + instrument + '.png', dpi=500)

#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------


# For Detrended model
fig = plt.figure(figsize=(16/1.5,9/1.5))
gs = gd.GridSpec(2,1, height_ratios=[2,1])

# Top panel
ax1 = plt.subplot(gs[0])
ax1.errorbar(tim, fl_det, fmt='.', c='cornflowerblue', alpha=0.25)
#ax1.errorbar(tbin_det, flbin_det, yerr=flebin_det, fmt='.', c='gray', alpha=0.7, zorder=50)
ax1.errorbar(tbin_det2, flbin_det2, yerr=flebin_det2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax1.plot(tim, tmodel, c='navy', lw=2.5, zorder=100)
for i in range(50):
    tmodel1 = utl.transit_model(times=tim, per=per, tc=np.random.choice(t0), rp1=np.random.choice(rprs),\
                                ar1=np.random.choice(ar1), bb1=np.random.choice(bb),\
                                q1=np.random.choice(q1), q2=np.random.choice(q2))
    ax1.plot(tim, tmodel1, c='orangered', zorder=75, lw=1, alpha=0.3)

ax1.set_ylabel('Relative Flux', fontsize=14)
ax1.set_xlim(np.min(tim), np.max(tim))
ax1.set_ylim(1-0.00075,1+0.0006)
ax1.text(tim[100], 1+0.0005, \
            r'Eclipse Depth: $' + str(int(dep_med)) + '^{+' + str(int(dep_up)) + '} _{-' + str(int(dep_do)) +'}$ ppm',\
            fontsize=15, fontweight='bold', color='maroon', zorder=1000)
    
plt.setp(ax1.get_xticklabels(), fontsize=12)
plt.setp(ax1.get_yticklabels(), fontsize=12)
ax1.xaxis.set_major_formatter(plt.NullFormatter())
ax1.set_title('Detrended lightcurve for Visit ' + str(instrument[-1]), fontsize=15)

# Bottom panel
ax2 = plt.subplot(gs[1])
ax2.errorbar(tim, resid, c='cornflowerblue', fmt='.', alpha=0.25)
#ax2.errorbar(tbin, resbin, yerr=resebin, c='gray', fmt='.', alpha=0.7, zorder=50)
ax2.errorbar(tbin2, resbin2, yerr=resebin2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax2.axhline(y=0.0, c='navy', lw=2.5, ls='--', zorder=100)
ax2.set_ylabel('Residuals (ppm)', fontsize=14)
ax2.set_xlabel('Time (BJD)', fontsize=14)
ax2.set_ylim(-250,250)
ax2.set_xlim(np.min(tim), np.max(tim))
plt.setp(ax2.get_xticklabels(), fontsize=12)
plt.setp(ax2.get_yticklabels(), fontsize=12)
#plt.show()
plt.savefig(pout + '/Detrended_LC_' + instrument + '.png', dpi=500)

#--------------------------------------------------------
#--------------------------------------------------------
#--------------------------------------------------------

# For eclipse time
mad1 = mad_std(fl)*1e6

# For Full model
fig, ax = plt.subplots(figsize=(16/1.5,9/1.5))
# Top panel
ax.errorbar(tim, fl, fmt='.', c='cornflowerblue', alpha=0.25)
ax.errorbar(tbin, flbin, yerr=flebin, fmt='.', c='gray', alpha=0.7, zorder=50)
ax.errorbar(tbin2, flbin2, yerr=flebin2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
ax.set_ylabel('Relative Flux', fontsize=14)
ax.set_xlabel('Time (BJD)', fontsize=14)
ax.set_xlim(np.min(tim), np.max(tim))
ax.set_ylim(0.9990,1.0015)
plt.setp(ax.get_xticklabels(), fontsize=12)
plt.setp(ax.get_yticklabels(), fontsize=12)
ax.set_title('White-Light lightcurve for Visit ' + str(instrument[-1]), fontsize=15)
ax.text(tim[200], 1.00135, 'Median Absolute Deviation: ' + str(np.around(mad1, 4)) + ' ppm', fontsize=14, fontweight='bold', c='maroon', zorder=200)
#plt.show()
plt.savefig(pout + '/Full_LC_wo_model_' + instrument + '.png', dpi=500)


#--------------------------------------------------------
#--------- To only plot residuals -----------------------
#--------------------------------------------------------

fig, axs = plt.subplots(figsize=(15/1.5,5/1.5))

axs.errorbar(tim, resid, c='cornflowerblue', fmt='.', alpha=0.25)
axs.errorbar(tbin, resbin, yerr=resebin, c='gray', fmt='.', alpha=0.7, zorder=50)
axs.errorbar(tbin2, resbin2, yerr=resebin2, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150)
for i in range(13):
    plt.axvline(tim[i*(380)], ls=':', c='k')
axs.axhline(y=0.0, c='black', ls='--', zorder=100)
axs.set_ylabel('Residuals (ppm)', fontsize=16)
axs.set_xlabel('Time (BJD)', fontsize=16)
axs.set_ylim(-250,250)
axs.set_xlim(np.min(tim), np.max(tim))
plt.setp(ax2.get_xticklabels(), fontsize=14)
plt.setp(ax2.get_yticklabels(), fontsize=14)
plt.tight_layout()
#plt.show()
plt.savefig(pout + '/Residuals_' + instrument + '.png', dpi=500)