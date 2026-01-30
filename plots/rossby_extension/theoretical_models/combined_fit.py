import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
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


'''
    All fitted data plotted together with theoretical models.
'''


###############################################
### Main dataset #################################
###############################################
# Lanzilao 2025
with open('../power_function_fits/lanzilao2025fit.pkl', 'rb') as fp:
    data_l2025 = pickle.load(fp)
H = [300,500,1000]
colors = ['orange','green','red']


###############################################
### Other datasets #################################
###############################################
Nother = 6
grayscale = cm.gray_r(np.linspace(0.2,0.6,Nother))
# Berg et al. (2020)
with open('berg2020fit.pkl', 'rb') as fp:
    data_berg = pickle.load(fp)
bergsty = {'color':grayscale[0],'marker':'d','label':'Berg et al. (2020)','markersize':6,'linestyle':'None'}

# Baungaard et al. (2024)
with open('baungaard2024fit.pkl', 'rb') as fp:
    data_baungaard = pickle.load(fp)
baungaardsty = {'color':grayscale[1],'marker':'X','label':'Baungaard et al. (2024)','markersize':6,'linestyle':'None'}

# Liu et al. (2024)
with open('liu2024fit.pkl', 'rb') as fp:
    data_liu = pickle.load(fp)
liu_sty = {'color':grayscale[2],'marker':'.','label':'Liu et al. (2024)','markersize':6,'linestyle':'None'}

# Lanzilao 2024
with open('lanzilao2024fit.pkl', 'rb') as fp:
    data_l2024 = pickle.load(fp)
l2024sty = {'color':grayscale[3],'marker':'^','label':'Lanzilao et al. (2024)','markersize':6,'linestyle':'None'}

# Heck 2025
with open('heck2025fit.pkl', 'rb') as fp:
    data_heck = pickle.load(fp)
hecksty = {'color':grayscale[4],'marker':'o','label':'Heck et al. (2025)','markersize':6,'linestyle':'None'}



###############################################
### Theory #################################
###############################################
iRo_h0_theory = np.linspace(0,0.15,500)
Ro_h0_theory = 1/iRo_h0_theory
px_theory = 70*iRo_h0_theory + 1
hx0_ho_theory = np.exp(-(iRo_h0_theory/0.02)**3)



###############################################
### Plot #################################
###############################################
# 1x2 subplot with h0x/h0 (left) and px (right) vs 1/Ro_h0
xticks = np.array([0,0.005,0.01,0.015])
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,2.5))
plt.subplots_adjust(wspace=0.35)

# Left plot
ax1.set_xlabel(r'$Ro_{h_0}^{-1}=f_c h_0/G$')
ax1.set_ylabel(r'$h_{x0}/h_0$',**yd)
# Plot Lanzilao2025 data
for i in range(len(data_l2025['h0fit'])):
    ax1.plot(1/data_l2025['Ro_h0'][i], data_l2025['h0fit_taux'][i]/data_l2025['h0fit'][i], marker='s', color=colors[i],
            label='H%d'%(H[i]), markersize=8, linestyle='None',zorder=99)
# Plot Berg2020 data
ax1.plot(1/data_berg['Ro_h0'], data_berg['h0fit_taux']/data_berg['h0fit'], **bergsty)
# Plot Baungaard2024 data
ax1.plot(1/data_baungaard['Ro_h0'], data_baungaard['h0fit_taux']/data_baungaard['h0fit'], **baungaardsty)
# Plot Liu2024 data
ax1.plot(1/data_liu['Ro_h0'], data_liu['h0fit_taux']/data_liu['h0fit'], **liu_sty)
# Plot Lanzilao2024 data
ax1.plot(1/data_l2024['Ro_h0'], data_l2024['h0fit_taux']/data_l2024['h0fit'], **l2024sty)
# Plot Heck2025 data
ax1.plot(1/data_heck['Ro_h0'], data_heck['h0fit_taux']/data_heck['h0fit'], **hecksty)
# Plot theory
ax1.plot(iRo_h0_theory, hx0_ho_theory, 'k-')
ax1.set_xlim([0,0.015])
ax1.set_ylim([0.6,1.1])
ax1.set_xticks(xticks)

# Right plot
ax2.set_xlabel(r'$Ro_{h_0}^{-1}=f_c h_0/G$')
ax2.set_ylabel(r'$p_x$',**yd)
# Plot Lanzilao2025 data
for i in range(len(data_l2025['h0fit'])):
    ax2.plot(1/data_l2025['Ro_h0'][i], data_l2025['p_fit_taux'][i], marker='s', color=colors[i],
            label='H%d'%(H[i]), markersize=8, linestyle='None',zorder=99)
# Plot Berg2020 data
ax2.plot(1/data_berg['Ro_h0'], data_berg['p_fit_taux'], **bergsty)
# Plot Baungaard2024 data
ax2.plot(1/data_baungaard['Ro_h0'], data_baungaard['p_fit_taux'], **baungaardsty)
# Plot Liu2024 data
ax2.plot(1/data_liu['Ro_h0'], data_liu['p_fit_taux'], **liu_sty)
# Plot Lanzilao2024 data
ax2.plot(1/data_l2024['Ro_h0'], data_l2024['p_fit_taux'], **l2024sty)
# Plot Heck2025 data
ax2.plot(1/data_heck['Ro_h0'], data_heck['p_fit_taux'], **hecksty)
# Plot theory
ax2.plot(iRo_h0_theory, px_theory, 'k-', label='Theory')
ax2.set_xlim([0,0.015])
ax2.set_ylim([1,2.2])
ax2.set_xticks(xticks)

# Add a label (e.g. (a), (b), etc.) to each subplot in the top-left corner
axes = [ax1, ax2]
for i in range(2):
    axes[i].text(0.01, 1.11, f'({chr(97 + i)})', transform=axes[i].transAxes,
               fontsize=14, fontweight='bold', va='top')
# Remove duplicates from legend
handles, labels = ax2.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='center left', fontsize=11, bbox_to_anchor=(0.91, 0.51),
          ncol=1, fancybox=True, shadow=True, scatterpoints=1, handlelength=1.5)
fig.savefig('article_invRoh0.pdf', bbox_inches='tight')


