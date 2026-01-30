import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import sys
sys.path.append('../../../') # To be able to utils.py
from utils import power_law_fit_iterative

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
    Fit shear stress profiles with power function.

    A small correction (linear extrapolation) of the shear stress profiles close to the wall is applied 
    before any fitting is done. This is because the LES wall function gives some wiggles in the first
    few wall-adjacents.
'''

datapath = '../../../data_lanzilao2025/'


#####################################################################
### PARAMETERS ########################################
#####################################################################
G = 10.0
fc = 1.14*10**(-4)
z0 = 1e-4
H = [300,500,1000]
colors = ['orange','green','red']



#####################################################################
### Load precursor data #################################
#####################################################################
precursors = []
for i in range(len(H)):
    sim_name = 'H%d-C5-G4'%(H[i])
    with open(datapath + sim_name + '/precursor_avg.pkl', 'rb') as fp:
        precursors.append(pickle.load(fp))



#####################################################################
##  Correct shear stress profiles close to wall ######################
#######################################################################
tauxw0 = np.zeros((len(H)))
tauyw0 = np.zeros((len(H)))
tautotw0 = np.zeros((len(H)))
uw0_ext = []
vw0_ext = []
tautot0_ext = []
z_ext0 = []
widxp = 4
for i in range(len(H)):    
    precursori = precursors[i]
    # Estimate tauxw0 from linear extrapolation of point widxp and widxp+1
    tauxw0[i] = -(  precursori['uw_tot'][widxp+1] - precursori['uw_tot'][widxp]) / ( precursori['z'][widxp+1] - precursori['z'][widxp] ) * (0 - precursori['z'][widxp] ) - precursori['uw_tot'][widxp]
    # Estimate tauyw0 from linear extrapolation of point widxp and widxp+1
    tauyw0[i] = -(  precursori['vw_tot'][widxp+1] - precursori['vw_tot'][widxp]) / ( precursori['z'][widxp+1] - precursori['z'][widxp] ) * (0 - precursori['z'][widxp] ) - precursori['vw_tot'][widxp]
    # Total wall shear stress
    tautotw0[i] = np.sqrt(tauxw0[i]**2 + tauyw0[i]**2)
    # Fix uw
    uw0raw = -precursori['uw_tot'][widxp:]
    uw0add = np.insert(uw0raw,0,tauxw0[i])
    uw0_ext.append(uw0add)
    # Fix vw
    vw0raw = -precursori['vw_tot'][widxp:]
    vw0add = np.insert(vw0raw,0,tauyw0[i])
    vw0_ext.append(vw0add)
    # Fix z
    zraw0 = precursori['z'][widxp:]
    zadd0 = np.insert(zraw0,0,0)
    z_ext0.append(zadd0)
    # Tautot 
    tautot0_ext.append(np.sqrt(uw0_ext[i]**2 + vw0_ext[i]**2))

# Corrected streamwise shear stress profiles
taux = uw0_ext



#################################################################
############ PLOTTING (to check of wall extrapolation) ###
#################################################################
def base_plot(ylabel='$z$ [m]',xlim='None',turbine='on'):
    fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)
    axes[0].set_ylabel(ylabel,**yd)
    axes[0].set_xlabel(r'$\tau_x$ [m$^2$ s$^{-2}$]')
    axes[1].set_xlabel(r'$\tau_y$ [m$^2$ s$^{-2}$]')
    for ax in axes:
        if xlim != 'None':
            ax.set_xlim(xlim)
        ax.set_ylim([0,1000])
    return fig, axes


# tau_x and tau_y (zoom-in with linear wall extrapolation)
fig, axes = base_plot(xlim=[0,0.085])
for i in range(len(H)):    
    precursori = precursors[i]
    axes[0].plot(-precursori['uw_tot'], precursori['z'],'o-',color=colors[i],label='H%d'%(H[i]),markersize=3)
    axes[0].plot(uw0_ext[i], z_ext0[i],'--',color=colors[i])
    axes[1].plot(-precursori['vw_tot'], precursori['z'],'o-',color=colors[i],markersize=3)
    axes[1].plot(vw0_ext[i], z_ext0[i],'--',color=colors[i])
axes[0].set_title(r'$\tau_x$ - Raw (solid) and extrapolated (dashed)',fontsize=17)
axes[0].set_xlim([0.05,0.1])
axes[1].set_title(r'$\tau_y$ - Raw (solid) and extrapolated (dashed)',fontsize=17)
axes[1].set_xlim([-0.01,0.005])
for ax in axes:
    ax.set_ylim([0,100])
fig.legend(loc='center left',fontsize=14,bbox_to_anchor=(0.91, 0.5),
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=1.5)

# tau_x and tau_y (corrected)
fig, axes = base_plot(xlim=[0,0.1])
for i in range(len(H)):    
    axes[0].plot(uw0_ext[i], z_ext0[i],'-',color=colors[i],label='H%d'%(H[i]))
    axes[1].plot(vw0_ext[i], z_ext0[i],'-',color=colors[i])
axes[0].set_title(r'$\tau_x$ - Precursor corrected',fontsize=17)
axes[1].set_title(r'$\tau_y$ - Precursor corrected',fontsize=17)
for ax in axes:
    ax.set_ylim([0,1000])
axes[1].set_xlim([-0.05,0.005])
fig.legend(loc='center left',fontsize=14,bbox_to_anchor=(0.91, 0.5),
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=1.5)




#################################################################
############ Fit tautot/tautotw0 with analytical expression:
###########    tau/tauw0 = (1 - z/h0)^p   ###
########## Therer are two parameters, h0 and p ##########
#################################################################
# Fit with iterative method
h0fit = np.zeros((len(H)))
p_fit = np.zeros((len(H)))
for i in range(len(H)):
    zi = z_ext0[i]
    tautoti = tautot0_ext[i]
    tautotw0i = tautotw0[i]
    h0fit[i], p_fit[i] = power_law_fit_iterative(zi, tautoti, tautotw0i)


#################################################################
############ Fit taux/tauxw0 with analytical expression:
###########    taux/tauxw0 = (1 - z/h0)^p   ###
########## Therer are two parameters, h0 and p ##########
#################################################################
h0fit_taux = np.zeros((len(H)))
p_fit_taux = np.zeros((len(H)))
for i in range(len(H)):
    h0fit_taux[i], p_fit_taux[i] = power_law_fit_iterative(z_ext0[i], taux[i], tauxw0[i])



#################################################################
############ Plot fits #####################################
#################################################################
# normalized total momentum flux (left) and normalized taux (right)
fig, axes = plt.subplots(1, 2, figsize=(12,4), sharey=True)
axes[0].set_xlabel(r'$|\tau|_0/|\tau|_{w0}$')
axes[0].set_ylabel(r'$z$ [m]', **yd)
axes[1].set_xlabel(r'$\tau_{x0}/\tau_{xw0}$')
# LES
for i in range(len(H)):
    # left: total momentum flux (data + fit)
    axes[0].plot(tautot0_ext[i]/tautotw0[i], z_ext0[i], '-', color=colors[i], label='H%d LES' % (H[i]))
    axes[0].plot((1 - z_ext0[i]/h0fit[i])**(p_fit[i]), z_ext0[i], '--', color=colors[i],
                 label=r'Fit: $p$=%.2f, $h_0$=%.0f m' % (p_fit[i], h0fit[i]))
    # right: tau_x (data + fit)
    axes[1].plot(taux[i]/tauxw0[i], z_ext0[i], '-', color=colors[i], label='H%d LES' % (H[i]))
    axes[1].plot((1 - z_ext0[i]/h0fit_taux[i])**(p_fit_taux[i]), z_ext0[i], '--', color=colors[i],
                 label=r'Fit: $p_x$=%.2f, $h_{x0}$=%.0f m' % (p_fit_taux[i], h0fit_taux[i]))
for ax in axes:
    ax.set_ylim([0,1250])
    ax.set_xlim([0,1])
axes[0].legend(loc='upper right', fontsize=14,
           ncol=1, fancybox=True, shadow=True, scatterpoints=1, handlelength=1.5)
axes[1].legend(loc='upper right', fontsize=14,
           ncol=1, fancybox=True, shadow=True, scatterpoints=1, handlelength=1.5)

# Add a label (e.g. (a), (b), etc.) to each subplot in the top-left corner
for i in range(2):
    axes[i].text(0.01, 1.07, f'({chr(97 + i)})', transform=axes[i].transAxes,
               fontsize=14, fontweight='bold', va='top')
fig.savefig('article_shear_stress_power_law_fit.pdf', bbox_inches='tight')


#################################################################
############ Save fitted coefficients #####################################
#################################################################
if False:
    Ro_h0 = np.zeros((len(H)))
    for i in range(len(H)):
        Ro_h0[i] = G/(fc*h0fit[i])
    fit_data = {
        'Ro_h0': Ro_h0,
        'h0fit': h0fit,
        'p_fit': p_fit,
        'h0fit_taux': h0fit_taux,
        'p_fit_taux': p_fit_taux,
    }
    with open('lanzilao2025fit.pkl', 'wb') as fp:
        pickle.dump(fit_data, fp)










