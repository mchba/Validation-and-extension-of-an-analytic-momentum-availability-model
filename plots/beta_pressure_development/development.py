import numpy as np
import xarray
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import sys
sys.path.append('../../') # To be able to utils.py
from utils import getList


mpl.style.use('classic')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["legend.numpoints"] = 1
plt.rcParams['grid.linestyle'] = ':' # Dotted gridlines
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 17
plt.rcParams['figure.dpi'] = 300
plt.rcParams['axes.grid']=True
mpl.rcParams['axes.formatter.useoffset'] = False
yd = dict(rotation=0,ha='right',va='center') 
plt.close('all')


'''
    Plots of P(x) and beta_local(x).
'''


# PARAMETERS ########################################
D = 198.0
zh = 119.0
HF = 2.5*zh
G = 10.0
S = 5*D # inter-spacing
cases = ['H300-C5-G4',
         'H500-C5-G4',
         'H1000-C5-G4',
         ]
labels_short = ['H300','H500','H1000']
colors = ['orange','green','red']


######################################################
# LOAD WIND FARM LAYOUT #################################
######################################################
with open('../../data/H500-C5-G4/wf_setup.pkl', 'rb') as fp:
    wf = pickle.load(fp)
wfs = [wf,wf,wf]

##############################################
### LOAD UF0 #################################
###############################################
precursors = []
for i in range(len(cases)):
    sim_name = cases[i]
    with open('../../data/' + sim_name + '/precursor_avg.pkl', 'rb') as fp:
        precursors.append(pickle.load(fp))
UF0 = getList(precursors,'UF0')


###########################################################
## Load laterally-averaged data and average from z=0 to z=HF
###########################################################
avgs = []
lines = []
beta_local = []
for i in range(len(cases)):
    avgi = xarray.load_dataset('../../data/'+"%s/main_latavg.nc"%(cases[i]))
    avgs.append(avgi)    
    avgifarm = avgi.where(avgi.z < HF, drop=True)
    lines.append(avgifarm.mean(dim='z'))
    beta_local.append(lines[i].u/UF0[i])


##################################################################
########## Combined beta_local and pressure development plot for article #########3
##################################################################
reduce = 1.0
fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(reduce*7, reduce*5))
ax = ax.flatten()
plt.subplots_adjust(hspace=0.25,wspace=-0.0)

# betalocal
for i in range(len(cases)):    
    linei = lines[i]
    betalocal = linei.u/UF0[i]
    ax[0].plot((linei.x - wfs[i]['cvxstart'])/1000,betalocal,color=colors[i],label=labels_short[i])
# pressure
for i in range(len(cases)):    
    linei = lines[i]
    ax[1].plot((linei.x - wfs[i]['cvxstart'])/1000,(linei.p - linei.p[0])/G**2,color=colors[i])
ax[1].set_ylabel(r'$\dfrac{P^* - P_{\rm inlet}^*}{\rho G^2}$',**yd)
ax[0].axvspan(0, wf['L']/1000, color='gray', alpha=0.3)
# plot settings
ax[0].set_yticks([0.6,0.7,0.8,0.9,1.0])
ax[1].set_xlabel(r'$x - x_{\rm wf,start}$  [km]')
ax[0].set_ylabel(r'$\beta_{\rm local}$',**yd)
ax[0].set_xlim([-17,26])
ax[0].set_ylim([0.58,1.0])
ax[1].axvspan(0, wf['L']/1000, color='gray', alpha=0.3)
ax[1].set_ylim([-0.15,0.15])
fig.text(0.15,0.92,r"$\bf{(a)}$",ha='center', fontsize=15)
fig.text(0.15,0.48,r"$\bf{(b)}$",ha='center', fontsize=15)
fig.legend(loc='center left',fontsize=14,bbox_to_anchor=(0.91, 0.5),
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=1.5)
fig.savefig('beta_pressure_development.pdf',bbox_inches='tight')
