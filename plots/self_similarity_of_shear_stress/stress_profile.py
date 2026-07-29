import numpy as np
import xarray
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import sys
sys.path.append('../../') # To be able to utils.py
from utils import getList
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from scipy.interpolate import interp1d

mpl.style.use('classic')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["legend.numpoints"] = 1
plt.rcParams['grid.linestyle'] = ':' # Dotted gridlines
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 17
plt.rcParams['axes.grid']=True
mpl.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['figure.dpi'] = 300
yd = dict(rotation=0,ha='right',va='center') 
plt.close('all')


'''
   Check of self-similarity of streamwise shear stress profiles of ABL/main simulation pairs.
'''



### PARAMETERS ########################################
cases = ['H300-C5-G4',
         'H500-C5-G4',
         'H1000-C5-G4'
    ]
short_name = ['H300','H500','H1000']
labels_short = cases
colors = ['orange','green','red']

### LOAD WIND FARM LAYOUTS #################################
with open('../../data/H500-C5-G4/wf_setup.pkl', 'rb') as fp:
    wf = pickle.load(fp)
zh = 119.0
wf['HF'] = 2.5*zh
wfs = [wf,wf,wf]
HF = getList(wfs,'HF')


### Load precursor data #################################
precursors = []
for i in range(len(cases)):
    sim_name = cases[i]
    with open('../../data/' + sim_name + '/precursor_avg.pkl', 'rb') as fp:
        precursors.append(pickle.load(fp))
tauw0 = getList(precursors,'tauw0')
h005 = getList(precursors,'h005') # 5% ABL height


### Load M values #########################
M_datasets = []
for i in range(len(cases)):
    sim_name = cases[i]
    with open('../../data/' + sim_name + '/Mdata.pkl', 'rb') as fp:
        M_datasets.append(pickle.load(fp))
M = getList(M_datasets,'M')
tauw1 = M*tauw0


### Load laterally-averaged data and average from z=0 to z=HF ##############
avgs = []
for i in range(len(HF)):
    sim_name = cases[i]
    avgi = xarray.load_dataset('../../data/'+sim_name+"/main_latavg.nc")
    avgs.append(avgi)    
    avgifarm = avgi.where(avgi.z < HF[i], drop=True)


### Horizontally-averaged (over CV area) flow profiles. ##############
havgs = []
for i in range(len(HF)):
    avgi = avgs[i]
    # Only consider data in the farm region
    avgic = avgi.where((avgi.x > wfs[i]['cvxstart']) & (avgi.x < wfs[i]['cvxend']), drop=True)
    # Average over length of farm to get farm profile
    havgs.append(avgic.mean(dim='x'))


# Extrapolate tau(z) to cleaner profile near the ground
tauw = []
for i in range(len(HF)):    
    havgi = havgs[i]
    # Linear extrapolate to ground
    widx1 = 1
    widx2 = 2
    z1 = float(havgi['z'][widx1]); z2 = float(havgi['z'][widx2])
    uw1 = float(havgi['uw_tot'][widx1]); uw2 = float(havgi['uw_tot'][widx2])
    uw_w = uw1 + (uw2-uw1)/(z2-z1) * (-z1)     
    tauw.append(-uw_w)
# Make arrays with the extrapolated tauw value
tau_ext = []
z_ext = []
for i in range(len(HF)):    
    havgi = havgs[i]
    tauraw = -havgi['uw_tot'][widx1:]
    tauadd = np.insert(tauraw,0,tauw[i])
    tau_ext.append(tauadd)
    zraw = havgi.z[widx1:]
    zadd = np.insert(zraw,0,0)
    z_ext.append(zadd)



############################################################################
### Calculate farm BL height with 5% criterion ##############
### (i)  using tauw0
### (ii) using tauw1
############################################################################
zb = 0.8*HF[0] 
hfarm = np.zeros_like(HF)
hfarm_tauw0 = np.zeros_like(HF)
for i in range(len(HF)):
    idxb = int(np.abs(z_ext[i] - zb).argmin().values)
    zp = z_ext[i][idxb:].values
    taup = tau_ext[i][idxb:].values
    z_of_tauw1 = interp1d(taup/tauw1[i], zp, kind='linear', fill_value='extrapolate')
    z_of_tauw0 = interp1d(taup/tauw0[i], zp, kind='linear', fill_value='extrapolate')

    # Now get z where tau = 0.05
    hfarm[i] = z_of_tauw1(0.05)
    hfarm_tauw0[i] = z_of_tauw0(0.05)
    

# Write bl_height.pkl file
# for i in range(len(HF)):
#     sim_name = cases[i]
#     heightdata = {'h0': h0[i],          # ABL height from fit of polynomial
#                   'h005': h005[i],      # 5% ABL height
#                   'h_tauw1': hfarm[i], # Height where tau(z)=0.05*tauw1
#                   'h_tauw0': hfarm_tauw0[i]  # Height where tau(z)=0.05*tauw0
#           }
#     with open('../../data/' + sim_name + '/bl_height.pkl', 'wb') as fp:
#         pickle.dump(heightdata, fp)
#         print('heightdata saved successfully to file')




############################################################################
### Linear extrapolate tau_0(z), i.e. precursor, near the ground to get cleaner profile ##############
############################################################################
tau0_ext = []
z_ext0 = []
widxp = 4
for i in range(len(HF)):    
    precursori = precursors[i]
    tau0raw = -precursori['uw_tot'][widxp:]
    tau0add = np.insert(tau0raw,0,tauw0[i])
    tau0_ext.append(tau0add)
    zraw0 = precursori['z'][widxp:]
    zadd0 = np.insert(zraw0,0,0)
    z_ext0.append(zadd0)



##########################################################################
### CHECK LINEAR EXTRAPOLATIONS ##########################################
##########################################################################
def base_plot(xlabel='$S/G$',ylabel='$z$ [m]',xlim='None',turbine='on'):
    fig = plt.figure(figsize=(7,5))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel,**yd)
    if xlim == 'None':
        pass
    else:
        plt.xlim(xlim)
    plt.ylim([0,1000])
    return fig
# Precursor
fig = base_plot(xlabel=r'$\tau_0$ [m2/s2]', xlim=[0,0.085])  
for i in range(len(HF)):
    precursor = precursors[i]
    plt.plot(-precursor['uw_tot'], precursor['z'],'o-',color=colors[i],label=labels_short[i],markersize=3)
    plt.plot(tau0_ext[i], z_ext0[i],'--',color=colors[i])
plt.title(r'Precursors',fontsize=17)
plt.ylim([0,100])

# Main
fig = base_plot(xlabel=r'$\tau$ [m2/s2]', xlim=[0,0.1])
for i in range(len(HF)):    
    havgi = havgs[i]
    plt.plot(-havgi['uw_tot'][:], havgi.z[:],'o-',color=colors[i],label=labels_short[i],markersize=3)
    plt.plot(tau_ext[i][0:5],z_ext[i][0:5],linestyle='--',color=colors[i], zorder=9)
plt.ylim([0,100])
plt.title(r'Full: averaged over farm region'+ '\n' + 'Dotted: extrapolation',fontsize=17)
fig.legend(loc='center left',fontsize=14,bbox_to_anchor=(0.91, 0.5),
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=1.5)


#####################################################################
# Subplot for article
#####################################################################
reduce = 0.9
fig, ax = plt.subplots(1, 2, sharex=False, sharey=False, figsize=(reduce*10, reduce*4))
plt.subplots_adjust(wspace=0.35)
ax = ax.flatten()

# Create zoomed inset
axins = zoomed_inset_axes(ax[0], zoom=5, loc='upper right')  # zoom=level of zoom, loc=position

# Left subplot
for i in range(len(HF)):    
    havgi = havgs[i]
    precursori = precursor
    ax[0].plot(tau_ext[i]/tauw0[i], z_ext[i],color=colors[i],label=short_name[i])
    ax[0].plot(tau0_ext[i]/tauw0[i], z_ext0[i],'--',color=colors[i])
    #ax[0].plot((tauw1[i] - precursori['z']/h1[i])/tauw0[i], precursori['z'],':',color=colors[i])
    # Inlet
    axins.plot(tau_ext[i]/tauw0[i], z_ext[i],color=colors[i])
    axins.plot(tau0_ext[i]/tauw0[i], z_ext0[i],'--',color=colors[i])

# Define zoom limits
x1, x2 = 0, 1.05  # Example x-limits of the zoom
y1, y2 = 0, 75 # Example y-limits of the zoom
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticks([0,1])
axins.set_yticks([0,25,50,75])
# Draw rectangle and connecting lines between inset and main plot
mark_inset(ax[0], axins, loc1=2, loc2=4, fc="none", ec="0.5")


# Right subplot (tau0/tauw0 and tau/tauw1)
for i in range(len(HF)):    
    havgi = havgs[i]
    precursori = precursor
    ax[1].plot(tau_ext[i]/tauw1[i], z_ext[i]/hfarm[i],color=colors[i])
    ax[1].plot(tau0_ext[i]/tauw0[i], z_ext0[i]/h005[i],'--',color=colors[i])



# Plot settings
ax[0].set_ylim([0,800])
ax[0].set_xlim([0,12])
ax[1].set_ylim([0,1.2])
ax[1].set_xlim([0,1.05])
ax[0].set_xlabel(r'$\tau_0/\tau_{w0}$ and $\tau/\tau_{w0}$')
ax[1].set_xlabel(r'$\tau_0/\tau_{w0}$ and $\tau/\tau_{w1}$')
ax[0].set_ylabel(r'$z$ [m]',**yd)
fig.text(0.14,0.92,r"$\bf{(a)}$",ha='center', fontsize=15)
fig.text(0.58,0.92,r"$\bf{(b)}$",ha='center', fontsize=15)
ax[1].set_ylabel(r'$\frac{z}{h_0}$ '+'\nand'+'\n'+r'$\frac{z}{h}$  ',**yd)
fig.legend(loc='lower center',fontsize=14,bbox_to_anchor=(0.5, 0.95),
          ncol=4, fancybox=True, shadow=True,scatterpoints=1, handlelength=1.5)
fig.savefig('stress_profiles.pdf',bbox_inches='tight')



