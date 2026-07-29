import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

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



############################################################
# NAMING ########################################
############################################################
short_name = ['H300', 'H500', 'H1000','Aligned','Double-spacing','Half-farm']



############################################################
########## LES DATA ########################
############################################################
# Load momentum sink data
with open('../../data_lanzilao2025/common/Mdata.pkl', 'rb') as fp:
    M_data1 = pickle.load(fp)
with open('../../data_lanzilao2024/common/Mdata.pkl', 'rb') as fp:
    M_data2 = pickle.load(fp)
T_LES = np.concatenate([np.array(M_data1['farmT_Scv_tauw0']), np.array(M_data2['farmT_Scv_tauw0'])])
tau_LES = np.concatenate([np.array(M_data1['tauw_tauw0']), np.array(M_data2['tauw_tauw0'])])




############################################################
#### Bar plot of momentum sinks #########################################
############################################################
fig, axes = plt.subplots(1, 6, figsize=(8, 3.5), sharey=True)

# Colors
col_T = 'k'
col_tau = 'gray'


# Bar plot parameters
barx = np.array([0])
barw = 0.35
baro = 0.09

# Create bar plots for each H
for i in range(len(short_name)):
    ax = axes[i]
    
    # Bar 1 (LES)
    ax.bar(barx, T_LES[i], color=col_T, width=barw, label=r'$\frac{T}{S_{\rm cv} \tau_{w0}}$')
    ax.bar(barx,tau_LES[i],bottom=T_LES[i],color=col_tau,width=barw,label=r'$\frac{\tau_{w}}{ \tau_{w0}}$')

    # Annotate the percentage of T_LES over the total (T_LES + tau_LES) with white text within the bar
    total_LES = T_LES[i] + tau_LES[i]
    perc_T_LES = T_LES[i] / total_LES * 100
    ax.text(barx, T_LES[i]/2, f'{perc_T_LES:.0f}%', ha='center', va='center', fontsize=10, color='white')

    # Also annotate the percentage of tau_LES over the total (T_LES + tau_LES) with white text within the bar
    perc_tau_LES = tau_LES[i] / total_LES * 100
    ax.text(barx, T_LES[i] + tau_LES[i]/2, f'{perc_tau_LES:.0f}%', ha='center', va='center', fontsize=10, color='white')

    ax.set_xticks([])
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([0, 12.0])
    ax.set_title('%s'%(short_name[i]),fontsize=14)
    
    if i == 0:
        ax.set_ylabel(r'$\frac{T + S_{\rm cv} \tau_{w}}{S_{\rm cv} \tau_{w0}}$', **yd)
        fig.legend(loc='upper left', fontsize=10, ncol=1, fancybox=True, 
             shadow=True, handlelength=1.5, bbox_to_anchor=(0.02, 0.95))

    # Add a label (e.g. (a), (b), etc.) to each subplot in the top-left corner
    ax.text(0.00, 1.15, f'({chr(97 + i)})', transform=ax.transAxes,
               fontsize=11, fontweight='bold', va='top')

plt.tight_layout()
fig.subplots_adjust(wspace=0.35)
fig.savefig('out_momentum_sinks.pdf', bbox_inches='tight')

