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
plt.rcParams['grid.linestyle'] = ':'
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 17
plt.rcParams['axes.grid']=True
plt.rcParams['figure.dpi'] = 300
mpl.rcParams['axes.formatter.useoffset'] = False
yd = dict(rotation=0,ha='right',va='center')
plt.close('all')


'''
    Script to plot beta_local(x) and P(x).
'''


# PARAMETERS ########################################
G = 10.0
H = [300,500,1000]
colors = ['orange','green','red']


######################################################
# LOAD WIND FARM LAYOUT #################################
#####################################################
with open('../../data_lanzilao2025/common/wf_setup.pkl', 'rb') as fp:
    wf = pickle.load(fp)
cvxstart = wf['cvxstart']
cvxend = wf['cvxend']
HF = wf['HF']
L = wf['L']


##############################################
### LOAD UF0 #################################
###############################################
precursors = []
for i in range(len(H)):
    sim_name = 'H%d-C5-G4'%(H[i])
    with open('../../data_lanzilao2025/' + sim_name + '/precursor_avg.pkl', 'rb') as fp:
        precursors.append(pickle.load(fp))
UF0 = getList(precursors,'UF0')



###########################################################
## Load laterally-averaged data and average from z=0 to z=HF
###########################################################
avgs = []
lines = []
beta_local = []
for i in range(len(H)):
    avgi = xarray.load_dataset('../../data_lanzilao2025/'+"%s/main_latavg.nc"%('H%d-C5-G4'%(H[i])))
    avgs.append(avgi)    
    avgifarm = avgi.where(avgi.z < HF, drop=True)
    lines.append(avgifarm.mean(dim='z'))
    beta_local.append(lines[i].u/UF0[i])




###################################################################################
########## Combined beta_local and pressure development plot for article ##########
###################################################################################
fig, ax = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(7, 5))
ax = ax.flatten()
plt.subplots_adjust(hspace=0.2,wspace=-0.1)

# betalocal
for i in range(len(H)):    
    linei = lines[i]
    betalocal = linei.u/UF0[i]
    ax[0].plot((linei.x - cvxstart)/1000,betalocal,color=colors[i],label='H%d'%(H[i]))
# pressure
for i in range(len(H)):    
    linei = lines[i]
    ax[1].plot((linei.x - cvxstart)/1000,(linei.p - linei.p[0])/G**2,color=colors[i])
ax[1].set_ylabel(r'$\dfrac{P^* - P_{\rm inlet}^*}{\rho G^2}$',**yd)
ax[1].axvspan(0, L/1000, color='gray', alpha=0.3)
# plot settings
ax[1].set_xlabel(r'$x - x_{\rm wf,start}$  [km]')
ax[0].set_ylabel(r'$\beta_{\rm local}$',**yd)
ax[0].set_xlim([-17,80])
ax[0].set_yticks(np.arange(0.6,1.01,0.1))
ax[0].axvspan(0, L/1000, color='gray', alpha=0.3)
fig.text(0.14,0.915,r"$\bf{(a)}$",ha='center', fontsize=15)
fig.text(0.14,0.48,r"$\bf{(b)}$",ha='center', fontsize=15)
fig.legend(loc='center left',fontsize=14,bbox_to_anchor=(0.91, 0.5),
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=1.5)
fig.savefig('beta_pressure_development.pdf',bbox_inches='tight')

