import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import ConnectionPatch
from exotoolbox.utils import tdur
import os
import utils as utl

pin = os.getcwd() + '/Groups/Outputs/'

# Loading the dataset
## Normalized lightcurves
tim, fl, fle = np.loadtxt(pin + '/White_light.dat', usecols=(0,1,2), unpack=True)

# For eclipse time
## Best fitted eclipse paramters
t0, per = 2460058.7417150857, 0.73654627
bb, ar1 = 0.358, 3.52
rprs = 0.0182
ts2 = np.array([t0 + (per/2)])

cycle_ecl = round((tim[0] - np.median(ts2))/per)
tc2 = np.median(ts2) + (cycle_ecl*per)
# Eclipse duration
t14 = tdur(per=per, ar=ar1, rprs=rprs, bb=bb)

# Running mean computation
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# Running mean
run_tim, run_fl = running_mean(tim, 4), running_mean(fl, 4)
## Binned data
tbin, flbin, flebin, _ = utl.lcbin(time=run_tim, flux=run_fl, binwidth=0.001)

# And the plots

fig = plt.figure(figsize=(16/1.5, 9/1.5), tight_layout=True)
gs = gridspec.GridSpec(2, 2, width_ratios=[1,3])

# Top panel (with all data)
ax1 = fig.add_subplot(gs[0, :])
ax1.errorbar(tim, fl, fmt='.', c='cornflowerblue', label='Raw white-light light curve')#, alpha=0.5)
#ax1.plot(run_tim, run_fl, color='maroon', zorder=70, label='Running mean')
ax1.axvline(tc2, color='orangered', ls='--', lw=2., zorder=150, label='Occultation time')
ax1.set_xlim(np.min(tim), np.max(tim))
ax1.legend(loc='best')
ax1.set_xlabel('Time (BJD)')
ax1.set_ylabel('Relative flux')

# Bottom left panel
ax2 = fig.add_subplot(gs[1, 0])
ax2.errorbar(tim, fl, fmt='.', c='cornflowerblue')#, alpha=0.5)
ax2.plot(tim, fl, color='navy', alpha=0.3)
ax2.plot(run_tim, run_fl, color='maroon', zorder=70, label='Running mean')
ax2.set_xlim([2460051.7840, 2460051.7870])
ax2.set_ylim([0.9970, 1.035])
#ax2.set_xlabel('Time (BJD)')
ax2.text(2460051.78405, run_fl[np.abs(run_tim-2460051.7840)<0.0001][0]+0.001, 'Running mean', color='maroon', fontweight='bold')
ax2.text(2460051.78405, 1.0305, 'Zoom-in on the part of the\ndata', color='orangered', fontweight='bold')

# Bottom right panel
ax3 = fig.add_subplot(gs[1, 1])
ax3.errorbar(run_tim, run_fl, fmt='.', color='maroon', zorder=70, label='Running mean')
ax3.errorbar(tbin, flbin, yerr=flebin, fmt='o', c='navy', elinewidth=2, capthick=2, capsize=3, mfc='white', zorder=150, label='Binned data')
ax3.set_xlim(np.min(run_tim), np.max(run_tim))
ax3.legend(loc='lower left')
ax3.set_xlabel('Time (BJD)')
ax3.set_ylabel('Relative flux')
ax3.set_ylim(1.00567,1.0081)
ax3.text(np.max(tim)-0.12, 1.0078, 'Running mean of the raw white-light light curve', color='maroon', fontweight='bold')

fig.suptitle('White-light light curve for the first group: GJ 341 transit data')

## Heighlighting the first zoom-in region (for ax2)
y0g, y00g = 0.9970, 1.035

x1, x2, y1, y2 = 2460051.7840, 2460051.7870, y0g, y00g
ax1.fill_between(x=np.linspace(x1, x2, 100), y1=np.ones(100)*y1, y2=np.ones(100)*y2, color='orangered', alpha=0.3, zorder=500)
ax1.plot(np.linspace(x1, x2, 100), np.ones(100)*y1, lw=1.5, c='orangered', zorder=500)
ax1.plot(np.linspace(x1, x2, 100), np.ones(100)*y2, lw=1.5, c='orangered', zorder=500)
ax1.plot(np.ones(100)*x1, np.linspace(y1, y2, 100), lw=1.5, c='orangered', zorder=500)
ax1.plot(np.ones(100)*x2, np.linspace(y1, y2, 100), lw=1.5, c='orangered', zorder=500)
# Connection Patches
con1 = ConnectionPatch(xyA=(x1, y1), coordsA=ax1.transData, xyB=(x1, y2),  coordsB=ax2.transData, color='orangered', lw=1.)
fig.add_artist(con1)
con2 = ConnectionPatch(xyA=(x2, y1), coordsA=ax1.transData, xyB=(x2, y2),  coordsB=ax2.transData, color='orangered', lw=1.)
fig.add_artist(con2)

plt.show()
#plt.savefig(pin + '/Full_whlc_Gr1.png', dpi=500)