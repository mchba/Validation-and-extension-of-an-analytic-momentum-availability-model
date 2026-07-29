import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import sys
sys.path.append('../../') # To use utils.py
from utils import getList
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
D = 198.0
zh = 119.0
HF = 2.5*zh
Ad = np.pi*(D/2)**2
G = 10.0
rho = 1
fc = 1.14*10**(-4)

short_name = ['H300', 'H500', 'H1000','H500_aligned','H500_double-spacing','H500_half-farm']
wf_path = ['../../data_lanzilao2025/common/wf_setup.pkl',
           '../../data_lanzilao2025/common/wf_setup.pkl',
           '../../data_lanzilao2025/common/wf_setup.pkl',
           '../../data_lanzilao2024/H500-C5-G4_aligned/wf_setup.pkl',
           '../../data_lanzilao2024/H500-C5-G4_double_spacing/wf_setup.pkl',
           '../../data_lanzilao2024/H500-C5-G4_half_farm/wf_setup.pkl'
           ]
prec_path = ['../../data_lanzilao2025/H300-C5-G4/precursor_avg.pkl',
             '../../data_lanzilao2025/H500-C5-G4/precursor_avg.pkl',
             '../../data_lanzilao2025/H1000-C5-G4/precursor_avg.pkl',
             '../../data_lanzilao2024/common/precursor_avg.pkl',
             '../../data_lanzilao2024/common/precursor_avg.pkl',
             '../../data_lanzilao2024/common/precursor_avg.pkl',
           ]


##########################################################
##### Input files ########################################
##########################################################

# Wind farm layout #################################
L = np.zeros(len(wf_path))
for i in range(len(wf_path)):
    with open(wf_path[i], 'rb') as fp:
        wf = pickle.load(fp)
    L[i] = wf['L']

## Load precursor data
precursors = []
for i in range(len(wf_path)):
    with open(prec_path[i], 'rb') as fp:
        precursors.append(pickle.load(fp))
Cf0 = getList(precursors,'Cf0')
h0 = getList(precursors,'h0')
tauw0 = getList(precursors,'tauw0')
taut0 = getList(precursors,'taut0')


## Load beta
with open('../../data_lanzilao2025/common/beta_pressure_data.pkl', 'rb') as fp:
    beta_data1 = pickle.load(fp)
with open('../../data_lanzilao2024/common/beta_pressure_data.pkl', 'rb') as fp:
    beta_data2 = pickle.load(fp)
beta = np.concatenate([np.array(beta_data1['beta']), np.array(beta_data2['beta'])])


############################################################
########## M_exact ########################
############################################################
# Load M and deltaM terms from LES
with open('../../data_lanzilao2025/common/Mdata.pkl', 'rb') as fp:
    M_data1 = pickle.load(fp)
with open('../../data_lanzilao2024/common/Mdata.pkl', 'rb') as fp:
    M_data2 = pickle.load(fp)
M_exact = np.concatenate([np.array(M_data1['M']), np.array(M_data2['M'])])


############################################################
########## M_KDN1 ########################
############################################################
M_KDN1 = (1 + HF/(L*Cf0)*(1-beta**2) - taut0/tauw0)/(beta*(1 - taut0/tauw0))


############################################################
########## M_KDN2 ########################
############################################################
M_KDN2 = (1 + h0/(L*Cf0)*(1-beta**2))/beta

############################################################
########## M_KDN3 ########################
############################################################
zeta_KDN3 = (1.18 + 2.18*h0/(L*Cf0))
M_KDN3 = 1 + zeta_KDN3*(1-beta)  

############################################################
########## M_lin with (with zeta=10) ########################
############################################################
M_lin = 1 + 10*(1-beta)

############################################################
########## M_one ########################
############################################################
M_one = np.ones(len(beta))



############################################################
#### Bar plot of M #########################################
############################################################
fig, axes = plt.subplots(2, 3, figsize=(11, 7), sharey=True)
axes = axes.flatten()


# Make colors with colormap viridis
Nmodels = 5
cmap = cm.get_cmap('viridis', Nmodels)
colors_bar = [cmap(i) for i in range(cmap.N)]
col_les = 'k'

# Bar plot parameters
barx = np.array([0])
barw = 0.12
baro = 0.08
b1off = 5*baro
b2off = 3*baro
b3off = baro
b4off = -baro
b5off = -3*baro
b6off = -5*baro

# Create bar plots for each case
for i in range(len(short_name)):
    ax = axes[i]
    
    # Bar 1 (LES)
    ax.bar(barx-b1off, M_exact[i], color=col_les, width=barw, label=r'$M_{\mathrm{exact}}$')
    
    # Bar 2 (KDN1)
    ax.bar(barx-b2off, M_KDN1[i], color=colors_bar[0], width=barw, 
           label=r'$M_{\mathrm{KDN1}}$')
    rel_diff = (M_KDN1[i] - M_exact[i]) / M_exact[i] * 100
    ax.text(barx-b2off, M_KDN1[i], f'{rel_diff:+.1f}%', ha='center', va='bottom', fontsize=9)

    # Bar 3 (KDN2)
    ax.bar(barx-b3off, M_KDN2[i], color=colors_bar[1], width=barw, 
           label=r'$M_{\mathrm{KDN2}}$')
    rel_diff = (M_KDN2[i] - M_exact[i]) / M_exact[i] * 100
    ax.text(barx-b3off, M_KDN2[i], f'{rel_diff:+.1f}%', ha='center', va='bottom', fontsize=9)

    # Bar 4 (KDN3)
    ax.bar(barx-b4off, M_KDN3[i], color=colors_bar[2], width=barw, 
           label=r'$M_{\mathrm{KDN3}}$')
    rel_diff = (M_KDN3[i] - M_exact[i]) / M_exact[i] * 100
    ax.text(barx-b4off, M_KDN3[i], f'{rel_diff:+.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Bar 5 (lin, zeta=10)
    ax.bar(barx-b5off, M_lin[i], color=colors_bar[3], width=barw, 
           label=r'$M_{\mathrm{lin}} (\zeta=10)$')
    rel_diff = (M_lin[i] - M_exact[i]) / M_exact[i] * 100
    ax.text(barx-b5off, M_lin[i], f'{rel_diff:+.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Bar 6 (one)
    ax.bar(barx-b6off, M_one[i], color=colors_bar[4], width=barw, 
           label=r'$M=1$')
    rel_diff = (M_one[i] - M_exact[i]) / M_exact[i] * 100
    ax.text(barx-b6off, M_one[i], f'{rel_diff:+.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_xticks([])
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([0, 26])
    ax.set_title('%s'%(short_name[i]),fontsize=16)
    
    if i % 3 == 0:
        ax.set_ylabel(r'$M$', **yd)
    
    if i == 0:
        ax.legend(loc='upper left', fontsize=9, ncol=1, fancybox=True, 
                 shadow=True, handlelength=1.5)
        
    # Add a label (e.g. (a), (b), etc.) to each subplot in the top-left corner
    ax.text(0.01, 1.08, f'({chr(97 + i)})', transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top')

plt.subplots_adjust(hspace=0.3, wspace=0.2)
fig.savefig('out_test_of_M.pdf', bbox_inches='tight')
fig.savefig('out_test_of_M.png', dpi=300, bbox_inches='tight')
