import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import pickle
import sys
sys.path.append('../../../') # To be able to utils.py
from utils import getList, twoscale
from matplotlib import cm

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




# PARAMETERS ########################################
zh = 119.0
HF = 2.5*zh
G = 10.0
fc = 1.14*10**(-4)

short_name = ['H300', 'H500', 'H1000','Aligned','Double-spacing','Half-farm']
wf_path = ['../../../data_lanzilao2025/common/wf_setup.pkl',
           '../../../data_lanzilao2025/common/wf_setup.pkl',
           '../../../data_lanzilao2025/common/wf_setup.pkl',
           '../../../data_lanzilao2024/H500-C5-G4_aligned/wf_setup.pkl',
           '../../../data_lanzilao2024/H500-C5-G4_double_spacing/wf_setup.pkl',
           '../../../data_lanzilao2024/H500-C5-G4_half_farm/wf_setup.pkl'
           ]
prec_path = ['../../../data_lanzilao2025/H300-C5-G4/precursor_avg.pkl',
             '../../../data_lanzilao2025/H500-C5-G4/precursor_avg.pkl',
             '../../../data_lanzilao2025/H1000-C5-G4/precursor_avg.pkl',
             '../../../data_lanzilao2024/common/precursor_avg.pkl',
             '../../../data_lanzilao2024/common/precursor_avg.pkl',
             '../../../data_lanzilao2024/common/precursor_avg.pkl',
           ]


##########################################################
##### LES DATA ########################################
##########################################################
# Wind farm layout #################################
L = np.zeros(len(wf_path))
lam = np.zeros(len(wf_path))
for i in range(len(wf_path)):
    with open(wf_path[i], 'rb') as fp:
        wf = pickle.load(fp)
    L[i] = wf['L']
    lam[i] = wf['lam']

## Precursor data
precursors = []
for i in range(len(wf_path)):
    with open(prec_path[i], 'rb') as fp:
        precursors.append(pickle.load(fp))
Cf0 = getList(precursors,'Cf0')
h0 = getList(precursors,'h0')

## beta
with open('../../../data_lanzilao2025/common/beta_pressure_data.pkl', 'rb') as fp:
    beta_data1 = pickle.load(fp)
with open('../../../data_lanzilao2024/common/beta_pressure_data.pkl', 'rb') as fp:
    beta_data2 = pickle.load(fp)
beta = np.concatenate([np.array(beta_data1['beta']), np.array(beta_data2['beta'])])

# M, CT_star and CP_star
with open('../../../data_lanzilao2025/common/Mdata.pkl', 'rb') as fp:
    M_data1 = pickle.load(fp)
with open('../../../data_lanzilao2024/common/Mdata.pkl', 'rb') as fp:
    M_data2 = pickle.load(fp)
M_LES = np.concatenate([np.array(M_data1['M']), np.array(M_data2['M'])])
zeta_LES = (M_LES - 1)/(1 - beta)
CT_star = np.concatenate([np.array(M_data1['CT_star']), np.array(M_data2['CT_star'])])
CP_star = np.concatenate([np.array(M_data1['CP_star']), np.array(M_data2['CP_star'])])


############################################################
########## Old M model (M_KDN3) ########################
############################################################
zeta_old = (1.18 + 2.18*h0/(L*Cf0))


############################################################
########## New M model (M_BNK) ########################
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
beta_old_calc = np.zeros_like(beta)
beta_new_calc = np.zeros_like(beta)
for i in range(len(wf_path)):
    beta_old_calc[i] = twoscale(CT_star[i], lam[i], Cf0[i], zeta_old[i])
    beta_new_calc[i] = twoscale(CT_star[i], lam[i], Cf0[i], zeta_new[i])


############################################################
########## Calculate C_PG ##################################
############################################################
C_PG_LES = beta**3 * CP_star
C_PG_old = beta_old_calc**3 * CP_star
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
    ax.bar(barx-b3off, zeta_old[i], color=col_old, width=barw, 
           label=r'Kirby et al. (2023)')
    rel_diff = (zeta_old[i] - zeta_LES[i]) / zeta_LES[i] * 100
    ax.text(barx-b3off, zeta_old[i], f'{rel_diff:+.1f}%', ha='center', va='bottom', fontsize=10)
    
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
fig_zeta.savefig('article_zeta_comparison.pdf', bbox_inches='tight')



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
    ax.bar(barx-b3off, C_PG_old[i], color=col_old, width=barw, 
           label=r'Kirby et al. (2023)')
    rel_diff = (C_PG_old[i] - C_PG_LES[i]) / C_PG_LES[i] * 100
    ax.text(barx-b3off, C_PG_old[i], f'{rel_diff:+.1f}%', ha='center', va='bottom', fontsize=10)
    
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
fig_cpg.savefig('article_CPG_comparison.pdf', bbox_inches='tight')


