import numpy as np
import xarray
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import pickle
import sys
sys.path.append('../../') # To be able to utils.py
from utils import getList
import matplotlib.colors as mcolors

mpl.style.use('classic')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["legend.numpoints"] = 1
plt.rcParams['grid.linestyle'] = ':'
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 17
plt.rcParams['axes.grid']=True
mpl.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['figure.dpi'] = 300
yd = dict(rotation=0,ha='right',va='center')
plt.close('all')


'''
    Contour plot of laterally-averaged tau13 to investigate BL height (based on 5% tauw0 criterion) spatial development.
'''

###########################################################################
# PARAMETERS ########################################
###########################################################################
H = [300,500,1000]



###########################################################################
## LOAD LES DATA ##########################################################
###########################################################################

# Wind farm layout #################################
with open('../../data_lanzilao2025/common/wf_setup.pkl', 'rb') as fp:
    wf = pickle.load(fp)
cvxstart = wf['cvxstart']
cvxend = wf['cvxend']
HF = wf['HF']

# Precursor data
precursors = []
for i in range(len(H)):
    sim_name = 'H%d-C5-G4'%(H[i])
    with open('../../data_lanzilao2025/' + sim_name + '/precursor_avg.pkl', 'rb') as fp:
        precursors.append(pickle.load(fp))
tauw0 = getList(precursors,'tauw0')
UF0 = getList(precursors,'UF0')
h005 = getList(precursors,'h005')

# 5% BL height over farm 
with open('../../data_lanzilao2025/common/h005_heights.pkl', 'rb') as fp:
    bldata = pickle.load(fp)
h005_farm_tauw0 = bldata['h005_farm_tauw0'] # 5% based on tauw0
h005_farm_tauw1 = bldata['h005_farm_tauw1'] # 5% based on tauw1 

# Laterally-averaged flow fields
avgs = []
lines = []
for i in range(len(H)):
    sim_name = 'H%d-C5-G4'%(H[i])
    avgi = xarray.load_dataset('../../data_lanzilao2025/'+sim_name+"/main_latavg.nc")
    avgs.append(avgi)    
    # Average from z=0 to z=HF
    avgifarm = avgi.where(avgi.z < HF, drop=True)
    lines.append(avgifarm.mean(dim='z'))




##############################################################################
# Post-processing ###################################################################
##############################################################################
# Calculate beta (needed to compare to model) and average of 1/beta_local
beta = np.zeros_like(H,dtype=float)
invbetam = np.zeros_like(H,dtype=float)
for i in range(len(H)):
    linei = lines[i]
    linei = linei.where((linei.x > cvxstart) & (linei.x < cvxend), drop=True)
    betalocal = linei['u'].values/UF0[i]
    beta[i] = np.mean(betalocal)
    invbetam[i] = np.mean(1/betalocal)

# Ratio h/h0
hratLES0 = h005_farm_tauw0/h005
hratModel = 1/beta
herror0 = (hratModel-hratLES0)/hratLES0*100


    


##############################################################################
# PLOTTING ###################################################################
##############################################################################
# Plotting template
def plot_stress(ax,X,Z,VAR,varlim='None',lvls=20,filename='None',
                 title='None',cl='None'):
    # Take values of the xarray and transpose to fit the meshgrid style
    VAR = VAR.values.T
    varmin = np.min(VAR); varmax = np.max(VAR)
    if varlim == 'None':
        vmin = varmin; vmax = varmax
    else:
        vmin = varlim[0]; vmax = varlim[1];
    lvls = np.linspace(vmin, vmax, lvls)
    # Structured data in matrix form
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    p = ax.contourf(X, Z, VAR, lvls, cmap=cm.viridis, norm=norm, extend="both")
    
    if cl != 'None':
        # Only make contour lines above this height index (this is to avoid drawing contour lines inside farm)
        zidx = 25 
        pcl = ax.contour(X[:,zidx:],Z[:,zidx:],VAR[:,zidx:],[cl],colors='k',linewidths=0.8,linestyles='-')
    
    # Set axes    
    ax.set_ylabel('$z$ [km]',**yd)
    ax.set_xlim([0,100])
    ax.set_ylim([0,1.3])
    ax.set_yticks([0,0.5,1.0])
    
    # Write information about min and max (useful for setting limits on colorbar)
    if False:
        info = r'%s'%(title) #+ ',  ' \
               #r'varmin $= %.2f$'%(varmin) + ',  ' \
             #+ r'varmax $= %.2f$'%(varmax)
        ax.text(2, 1.1, info, fontsize=14,
                   verticalalignment='top', horizontalalignment='left', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
    
    return pcl, p
    
# Subplots of stress contours    
X, Z = np.meshgrid(avgs[0]['x']/1000, avgs[0]['z']/1000,indexing='ij')
reduce = 1.3
fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(7.5*reduce, 5*reduce))
ax = ax.flatten()
plt.subplots_adjust(hspace=0.3,wspace=0.15)
if True:
    # Style 1
    vmin = -1; vmax = 7.5
else:
    # Style 2
    vmin = 0; vmax = 1
pcl = []
labels = [r"$\bf{(a)}$ ", r"$\bf{(b)}$ ",r"$\bf{(c)}$ "]
for i in range(len(H)):
    pcl0, p0 = plot_stress(ax[i], X, Z, -avgs[i]['uw_tot']/tauw0[i],
                           varlim=[vmin,vmax],lvls=21, cl=0.05)
                                 #varlim=[vmin,vmax],lvls=21,title=labels[i]+'H%d'%H[i], cl=0.05)
    pcl.append(pcl0)
    # Theoretical model
    ax[i].plot(lines[i].x/1000, h005[i]*UF0[i]/lines[i]['u']/1000, '--', color='k')
    # Mark farm region
    ax[i].axvspan(cvxstart/1000, cvxend/1000, color='gray', alpha=0.4)
    # Error text
    ax[i].text(15, 1.05, r'Error: %.1f'%(herror0[i]) + '%', fontsize=15,
               verticalalignment='bottom', horizontalalignment='left',color='white')
    ax[i].set_title('H%d'%(H[i]), fontsize=15)
ax[2].set_xlabel('$x$ [km]')
fig.text(0.14,0.915,r"$\bf{(a)}$",ha='center', fontsize=15)
fig.text(0.14,0.625,r"$\bf{(b)}$",ha='center', fontsize=15)
fig.text(0.14,0.34,r"$\bf{(c)}$",ha='center', fontsize=15)
# Colorbar
fig.subplots_adjust(right=0.8)
cbar = fig.add_axes([0.85, 0.1, 0.03, 0.8])
comcolRANS = plt.colorbar(p0, cax=cbar, aspect=100)
comcolRANS.set_ticks(np.linspace(vmin,vmax,5))
cbar.set_ylabel(r'$\dfrac{\tau_{13}}{\tau_{w0}}$',rotation=0,va='center',ha='left')
fig.savefig('h_contour.png',bbox_inches='tight',dpi=600)


