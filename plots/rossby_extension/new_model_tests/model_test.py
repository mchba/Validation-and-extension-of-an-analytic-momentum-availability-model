import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import pickle
import sys
sys.path.append('../../../') # To be able to utils.py
from utils import getList, twoscale

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
yd = dict(rotation=0,ha='right',va='center') # I couldn't find a way to customize these, so use a dict everytime..
plt.close('all')




# PARAMETERS ########################################
D = 198.0
zh = 119.0
HF = 2.5*zh
Ad = np.pi*(D/2)**2
G = 10.0
rho = 1
fc = 1.14*10**(-4)

short_name = ['H300', 'H500', 'H1000','Aligned','Double-spacing','Half-farm']
cases = ['H300-C5-G4',
         'H500-C5-G4',
         'H1000-C5-G4',
         'H500-C5-G4_aligned',
         'H500-C5-G4_double_spacing',
         'H500-C5-G4_half_farm'
    ]


##########################################################
##### Input files ########################################
##########################################################

# Wind farm layout #################################
L = np.zeros(len(cases))
lam = np.zeros(len(cases))
for i in range(len(cases)):
    sim_name = cases[i]
    with open('../../../data/' + sim_name + '/wf_setup.pkl', 'rb') as fp:
        wf = pickle.load(fp)
    L[i] = wf['L']
    lam[i] = wf['lam']

## Load precursor data
precursors = []
for i in range(len(cases)):
    sim_name = cases[i]
    with open('../../../data/' + sim_name + '/precursor_avg.pkl', 'rb') as fp:
        precursors.append(pickle.load(fp))
Cf0 = getList(precursors,'Cf0')
h0 = getList(precursors,'h0')


## Load beta ##############################
beta_datasets = []
for i in range(len(cases)):
    sim_name = cases[i]
    with open('../../../data/' + sim_name + '/beta_pressure_data.pkl', 'rb') as fp:
        beta_datasets.append(pickle.load(fp))
beta = getList(beta_datasets,'beta')

### Load M values #########################
M_datasets = []
for i in range(len(cases)):
    sim_name = cases[i]
    with open('../../../data/' + sim_name + '/Mdata.pkl', 'rb') as fp:
        M_datasets.append(pickle.load(fp))
M_LES = getList(M_datasets,'M')
CT_star = getList(M_datasets,'CT_star')
CP_star = getList(M_datasets,'CP_star')
zeta_LES = (M_LES - 1)/(1 - beta)



############################################################
########## Old M model ########################
############################################################
zeta_approx = (1.18 + 2.18*h0/(L*Cf0))




############################################################
########## New M model ########################
############################################################
invRo_h0 = fc*h0/G
px = 1 + 70*invRo_h0
hx0_to_h0 = np.exp(-(invRo_h0/0.02)**3.0)
hx0 = h0*hx0_to_h0
til_hx0 = HF + px**(-1.25)*(hx0 - HF)
zeta_new = (1.18 + 2.18*til_hx0/(L*Cf0))





############################################################
########## Calculate beta with quadratic equation (assuming gamma=2) ########################
############################################################
beta_approx_calc = np.zeros_like(beta)
beta_new_calc = np.zeros_like(beta)
for i in range(len(cases)):
    beta_approx_calc[i] = twoscale(CT_star[i], lam[i], Cf0[i], zeta_approx[i])
    beta_new_calc[i] = twoscale(CT_star[i], lam[i], Cf0[i], zeta_new[i])

############################################################
########## Calculate M from beta and zeta ########################
############################################################
M_new = 1 + zeta_new*(1 - beta_new_calc)
M_KDN_approx = 1 + zeta_approx*(1 - beta_approx_calc)

############################################################
########## Calculate C_PG ##################################
############################################################
C_PG_LES = beta**3 * CP_star
C_PG_approx = beta_approx_calc**3 * CP_star
C_PG_new = beta_new_calc**3 * CP_star




############################################################
#### Bar plot zeta #########################################
############################################################
fig_zeta, axes_zeta = plt.subplots(1, 6, figsize=(10, 3.5), sharey=True)

# Make colors with colormap viridis
cmap = cm.get_cmap('viridis', 5)
colors_bar = [cmap(i) for i in range(cmap.N)]
col_les = 'k'
col_old = colors_bar[1]
col_new = colors_bar[2]

# Bar plot parameters
barx = np.array([0])
barw = 0.2
baro = 0.09
b2off = 3*baro
b3off = 0
b4off = -3*baro

# Create bar plots for each H
for i in range(len(short_name)):
    ax = axes_zeta[i]
    
    # Bar 1 (LES)
    ax.bar(barx-b2off, zeta_LES[i], color=col_les, width=barw, label=r'LES')
    
    # Bar 2 (Approx model)
    ax.bar(barx-b3off, zeta_approx[i], color=col_old, width=barw, 
           label=r'Kirby et al. (2023)')
    rel_diff = (zeta_approx[i] - zeta_LES[i]) / zeta_LES[i] * 100
    ax.text(barx-b3off, zeta_approx[i], f'{rel_diff:+.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Bar 3 (New model)
    ax.bar(barx-b4off, zeta_new[i], color=col_new, width=barw, 
           label=r'New model')
    rel_diff = (zeta_new[i] - zeta_LES[i]) / zeta_LES[i] * 100
    ax.text(barx-b4off, zeta_new[i], f'{rel_diff:+.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks([])
    ax.set_xlim([-0.5, 0.5])
    ax.set_title('%s'%(short_name[i]),fontsize=14)
    
    if i == 0:
        ax.set_ylabel(r'$\zeta$', **yd)
        ax.legend(loc='upper left', fontsize=9, ncol=1, fancybox=True, 
                 shadow=True, handlelength=1.5)
        
    # Add a label (e.g. (a), (b), etc.) to each subplot in the top-left corner
    ax.text(0.00, 1.15, f'({chr(97 + i)})', transform=ax.transAxes,
               fontsize=11, fontweight='bold', va='top')

plt.tight_layout()
fig_zeta.subplots_adjust(wspace=0.35)
fig_zeta.savefig('article_zeta_comparison.png', bbox_inches='tight')


############################################################
#### Bar plot CPG #########################################
############################################################
fig_cpg, axes_cpg = plt.subplots(1, 6, figsize=(10, 3.5), sharey=True)

# Bar plot parameters (reuse from zeta plot)
barx = np.array([0])
barw = 0.2
baro = 0.09
b2off = 3*baro
b3off = 0
b4off = -3*baro

# Create bar plots for each case
for i in range(len(short_name)):
    ax = axes_cpg[i]
    
    # Bar 1 (LES)
    ax.bar(barx-b2off, C_PG_LES[i], color=col_les, width=barw, label=r'LES')
    
    # Bar 2 (Approx model)
    ax.bar(barx-b3off, C_PG_approx[i], color=col_old, width=barw, 
           label=r'Kirby et al. (2023)')
    rel_diff = (C_PG_approx[i] - C_PG_LES[i]) / C_PG_LES[i] * 100
    ax.text(barx-b3off, C_PG_approx[i], f'{rel_diff:+.1f}%', ha='center', va='bottom', fontsize=10)
    
    # Bar 3 (New model)
    ax.bar(barx-b4off, C_PG_new[i], color=col_new, width=barw, 
           label=r'New model')
    rel_diff = (C_PG_new[i] - C_PG_LES[i]) / C_PG_LES[i] * 100
    ax.text(barx-b4off, C_PG_new[i], f'{rel_diff:+.1f}%', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks([])
    ax.set_xlim([-0.5, 0.5])
    ax.set_title('%s'%(short_name[i]),fontsize=14)
    
    if i == 0:
        ax.set_ylabel(r'$C_{PG}$', **yd)
        ax.legend(loc='upper left', fontsize=9, ncol=1, fancybox=True, 
                 shadow=True, handlelength=1.5)
        
    # Add a label (e.g. (a), (b), etc.) to each subplot in the top-left corner
    ax.text(0.00, 1.15, f'({chr(97 + i)})', transform=ax.transAxes,
               fontsize=11, fontweight='bold', va='top')

plt.tight_layout()
fig_cpg.subplots_adjust(wspace=0.35)
fig_cpg.savefig('article_CPG_comparison.png', bbox_inches='tight')


