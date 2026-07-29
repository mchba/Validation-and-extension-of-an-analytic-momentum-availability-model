import numpy as np
import xarray
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import sys
sys.path.append('../../') # To be able to utils.py
from utils import getList, cv_surface
from matplotlib import cm
import matplotlib.gridspec as gridspec

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



#####################################################
# PARAMETERS ########################################
#####################################################
D = 198.0
zh = 119.0
HF = 2.5*zh
Nt = 160
Ad = np.pi*(D/2)**2
G = 10.0
S = 5*D # inter-spacing
rho = 1
fc = 1.14*10**(-4)
H = [300,500,1000]
colors = ['orange','green','red']



###########################################################################
## LOAD LES DATA ##########################################################
###########################################################################
# Wind farm layout #################################
with open('../../data_lanzilao2025/common/wf_setup.pkl', 'rb') as fp:
    wf = pickle.load(fp)
W = wf['W']
L = wf['L']
Scv = wf['Scv']
Vcv = Scv*HF

# Precursor data ###################################
precursors = []
for i in range(len(H)):
    sim_name = 'H%d-C5-G4'%(H[i])
    with open('../../data_lanzilao2025/' + sim_name + '/precursor_avg.pkl', 'rb') as fp:
        precursors.append(pickle.load(fp))
tauw0 = getList(precursors,'tauw0')
taut0 = getList(precursors,'taut0')
Cf0 = getList(precursors,'Cf0')
h0 = getList(precursors,'h0')     # ABL height from fit of polynomial
h005_abl = getList(precursors,'h005') # 5% ABL height
dpdx0 = getList(precursors,'dpdx0')
Xadv0 = getList(precursors,'Xadv0')
Xstr0 = getList(precursors,'Xstr0')
C0 = getList(precursors,'C0')
UF0 = getList(precursors,'UF0')
    
# Turbine data ##################################################
turbines = []
for i in range(len(H)):
    sim_name = 'H%d-C5-G4'%(H[i])
    with open('../../data_lanzilao2025/' + sim_name + '/turbines_avg.pkl', 'rb') as fp:
        turbines.append(pickle.load(fp))
farmT = getList(turbines,'farmT')*1e6 # Actually, this is T/rho
farmP = getList(turbines,'farmP')*1e6 # Actually, this is P/rho

# 5% BL height over farm ##########################################
with open('../../data_lanzilao2025/common/h005_heights.pkl', 'rb') as fp:
    bldata = pickle.load(fp)
h005_farm_tauw0 = bldata['h005_farm_tauw0']
h005_farm_tauw1 = bldata['h005_farm_tauw1']

# 3D flow fields in CV region #################################
datasets = []
for i in range(len(H)):
    sim_name = 'H%d-C5-G4'%(H[i])
    # Load dataset
    datai = xarray.load_dataset('../../data_lanzilao2025/' + sim_name+"/main_cv.nc")
    # Add to list
    datasets.append(datai)   
    
## beta and pressure data 
with open('../../data_lanzilao2025/common/beta_pressure_data.pkl', 'rb') as fp:
    beta_data = pickle.load(fp)
beta = np.array(beta_data['beta'])
beta0 = np.array(beta_data['beta0'])
betaL = np.array(beta_data['betaL'])
Pin = np.array(beta_data['Pin'])



#############################################################################
## Calculate CV surface averages #################################
############################################################################
cv_outs = []
for i in range(len(H)):
    # Load data
    datai = datasets[i]
    # Calculate surface averages
    cv_outi = cv_surface(datai,wf,method='midpoint')
    cv_outs.append(cv_outi)

# Advection terms (positive = flow into the box, negative = flow out of the box.)
front = getList(cv_outs,'front')
rear = getList(cv_outs,'rear')
south = getList(cv_outs,'south')
north = getList(cv_outs,'north')
top = getList(cv_outs,'top')
total = getList(cv_outs,'total')

# Stress terms
fronts = getList(cv_outs,'fronts')
rears = getList(cv_outs,'rears')
souths = getList(cv_outs,'souths')
norths = getList(cv_outs,'norths')
tops = getList(cv_outs,'tops')
totals = getList(cv_outs,'totals')

# Pressure on front and rear
frontp = getList(cv_outs,'frontp')
rearp = getList(cv_outs,'rearp')

# Wall shear stress
tauw = getList(cv_outs,'tauw')



####################################################################
## deltaM_adv ######################################################
####################################################################
deltaM_advnorm = (total - Vcv*Xadv0)/(W*L*tauw0)

# Model of deltaM_adv
madv_terms = [UF0**2*beta0**2*HF*W,                  # front
              -UF0**2*betaL**2*HF*W,                 # rear
              UF0*0,                                 # south
              UF0*0,                                 # north
              W*HF*UF0**2*0.5*(betaL**2 - beta0**2), # top
              
    ]
mdeltaM_advnorm = 0.5*UF0**2*HF*W*(beta0**2 - betaL**2)/(W*L*tauw0)

adv_terms = [front,rear,north,south,top]
adv_labels = ['front', 'rear', 'north', 'south', 'top']
adv_colors = plt.cm.viridis(np.linspace(0,1,len(adv_terms)))

# Combined subplot for article
fig = plt.figure(figsize=(9, 4))
gs = gridspec.GridSpec(1, 3)
gs.update(wspace=1)
ax = [plt.subplot(gs[0, :2]),  plt.subplot(gs[0, 2:])]

for i in range(len(adv_terms)):
    ax[0].plot(H, adv_terms[i]/(W*L*tauw0),'s-',label=adv_labels[i], color=adv_colors[i])
    ax[0].plot(H, madv_terms[i]/(W*L*tauw0),'<--', color=adv_colors[i], lw=4)
ax[0].set_xlabel('$H$ [m]')
ax[0].set_ylabel(r'$\dfrac{- \int \rho U_1 U_j dA_j}{X_{F0}}$',**yd,labelpad=-10)
ax[0].legend(loc='center right',fontsize=14,bbox_to_anchor=(1, 0.68),
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=2)
bars = 1.
barw = 0.3
barx = np.array([bars,2*bars,3*bars])
barcol = plt.cm.viridis(np.linspace(0,1,5))
ax[1].bar(barx-0.2,deltaM_advnorm,color=barcol[1],width=barw,label=r'LES')
ax[1].bar(barx+0.2,mdeltaM_advnorm,color=barcol[1],width=barw,label=r'Model',hatch='//')
ax[1].set_xticks(barx, ['H300', 'H500', 'H1000'],rotation=-45)
ax[1].set_ylabel(r'$\Delta M_{\rm adv}$',**yd)
ax[1].set_yticks([0,1,2,3,4])
ax[1].legend(loc='upper left',fontsize=10,
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=2)
madv_error = (mdeltaM_advnorm-deltaM_advnorm)/deltaM_advnorm*100
ax[1].text(1.1,3.2,r"%.1f"%(madv_error[0]) + "%",ha='center', fontsize=13)
ax[1].text(2.1,2.85,r"%.1f"%(madv_error[1]) + "%",ha='center', fontsize=13)
ax[1].text(3.1,4.3,"+" + r"%.1f"%(madv_error[2]) + "%",ha='center', fontsize=13)
ax[1].set_ylim([0,5])
fig.text(0.14,0.92,r"$\bf{(a)}$",ha='center', fontsize=15)
fig.text(0.75,0.92,r"$\bf{(b)}$",ha='center', fontsize=15)
fig.savefig('advection_combined.pdf',bbox_inches='tight')



#################################################################
## deltaM_PGF ###################################################
#################################################################
dpdx_avg = []     # Calculate with volume integral 
deltaM_PGF = []   
dpdx_avg2 = []    # Calculated with area integral
deltaM_PGF2 = []   

for i in range(len(H)):
    # Volume integral of dpdx
    datai = datasets[i]
    dpdx_avg.append(datai['dpdx'].mean())
    dpdx_avg2.append((rearp[i] - frontp[i])/L )
    # Total (remember dpdx is actually dP*/dx, hence precursor is not subtracted below)
    deltaM_PGF.append(Vcv*(-dpdx_avg[i]))
    deltaM_PGF2.append(Vcv*(-dpdx_avg2[i]))
deltaM_PGF = np.array(deltaM_PGF)
deltaM_PGFnorm = deltaM_PGF/(W*L*tauw0)

# Model of deltaM_PGF
deltaPfront = 0.5*UF0**2*(1 - beta0**2)
mdeltaM_PGFnorm = HF*W*2*deltaPfront/(W*L*tauw0)



# PGF figure for article
fig = plt.figure(figsize=(9, 4))
gs = gridspec.GridSpec(1, 3)
gs.update(wspace=1)
ax = [plt.subplot(gs[0, :2]),  plt.subplot(gs[0, 2:])]
ax[0].plot(H,(frontp-Pin)/G**2,'s-',color=adv_colors[0],label=r'front')
ax[0].plot(H,(0.5*UF0**2*(1-beta0**2))/G**2,'--',color=adv_colors[0])
ax[0].plot(H,(rearp-Pin)/G**2,'s-',color=adv_colors[1],label=r'rear')
ax[0].plot(H,(-0.5*UF0**2*(1-beta0**2))/G**2,'--',color=adv_colors[1])
ax[0].set_xlabel('$H$ [m]')
ax[0].set_ylabel(r'$\dfrac{P_{\rm}^* - P_{\rm inlet}^*}{\rho G^2}$',**yd,labelpad=-10)
ax[0].legend(loc='center right',fontsize=14,
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=2)
bars = 1.
barw = 0.3
barx = np.array([bars,2*bars,3*bars])
barcol = plt.cm.viridis(np.linspace(0,1,5))
ax[1].bar(barx-0.2,deltaM_PGFnorm,color=barcol[2],width=barw,label=r'LES')
ax[1].bar(barx+0.2,mdeltaM_PGFnorm,color=barcol[2],width=barw,label=r'Model',hatch='//')
ax[1].set_xticks(barx, ['H300', 'H500', 'H1000'],rotation=-45)
ax[1].set_ylabel(r'$\Delta M_{\rm PGF}$',**yd)
ax[1].set_yticks([0,1,2,3,4,5,6,7])
ax[1].legend(loc='upper left',fontsize=10,
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=2)
mPGF_error = (mdeltaM_PGFnorm-deltaM_PGFnorm)/deltaM_PGFnorm*100
ax[1].text(1.1,6.2,"+" + r"%.1f"%(mPGF_error[0]) + "%",ha='center', fontsize=13)
ax[1].text(2.1,5,"+" + r"%.1f"%(mPGF_error[1]) + "%",ha='center', fontsize=13)
ax[1].text(3.1,3,r"%.1f"%(mPGF_error[2]) + "%",ha='center', fontsize=13)
ax[1].set_ylim([0,8])
fig.text(0.14,0.92,r"$\bf{(a)}$",ha='center', fontsize=15)
fig.text(0.75,0.92,r"$\bf{(b)}$",ha='center', fontsize=15)
fig.savefig('pgf_combined.pdf',bbox_inches='tight')



#######################################################################
## Advection-pressure (AP) approximation ##############################
#######################################################################
msadeltaM_advpgfnorm = HF/(Cf0*L)*(1 - beta**2)
sa_Merror = (msadeltaM_advpgfnorm - (mdeltaM_PGFnorm+mdeltaM_advnorm))/(mdeltaM_PGFnorm+mdeltaM_advnorm)*100
sa_error = ((1 + beta**2) - (betaL**2 + beta0**2))/(betaL**2 + beta0**2)*100

fig = plt.figure(figsize=(9, 4))
gs = gridspec.GridSpec(1, 10)
gs.update(wspace=10)
ax = [plt.subplot(gs[0, :5]),  plt.subplot(gs[0, 7:])]
ax[0].plot(H,(betaL**2 + beta0**2),'s-',color='r',label=r'$\beta_{\rm local}(L)^2 + \beta_{\rm local}(0)^2$')
ax[0].plot(H,(1 + beta**2),'s--',color='r',label=r'$1+\beta^2$')
ax[0].plot(H,1-beta0,'s-',color='b',label=r'$1-\beta_{\rm local}(0)$')
ax[0].plot(H,betaL-beta,'s--',color='b',label=r'$\beta_{\rm local}(L)-\beta$')
ax[0].set_xlabel('$H$ [m]')
ax[0].text(310,1.3,r"%.1f"%(sa_error[0]) + "%",ha='left', fontsize=13)
ax[0].annotate("",xy=(300, 1.48), xytext=(300, 1.2),arrowprops=dict(arrowstyle="->", color='red', lw=1))
ax[0].text(510,1.38,r"%.1f"%(sa_error[1]) + "%",ha='left', fontsize=13)
ax[0].annotate("",xy=(500, 1.53), xytext=(500, 1.3),arrowprops=dict(arrowstyle="->", color='red', lw=1))
ax[0].text(890,1.42,r"%.1f"%(sa_error[2]) + "%",ha='left', fontsize=13)
ax[0].annotate("",xy=(1000, 1.55), xytext=(1000, 1.35),arrowprops=dict(arrowstyle="->", color='red', lw=1))
#ax[0].set_ylabel(r'$\dfrac{P_{\rm}^* - P_{\rm in}^*}{\rho }$' + r'$\left[\frac{{\rm m}^2}{{\rm s}^2}\right]$',**yd,labelpad=-10)
ax[0].legend(loc='center right',fontsize=13,
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=2)
bars = 1.
barw = 0.23
bari = 0.28
barx = np.array([bars,2*bars,3*bars])
barcol = plt.cm.viridis(np.linspace(0,1,5))
ax[1].bar([1],[0], edgecolor='black', color='None',label=r'LES')
ax[1].bar([1],[0], edgecolor='black', color='None',label=r'Model',hatch='//')
ax[1].bar([1],[0], edgecolor='black', color='None',label=r'AP approximation',hatch='xx')
ax[1].bar(barx-bari,deltaM_advnorm,color=barcol[1],width=barw)
ax[1].bar(barx-bari,deltaM_PGFnorm,bottom=deltaM_advnorm,color=barcol[2],width=barw)
ax[1].bar(barx+0.0,mdeltaM_advnorm,color=barcol[1],width=barw,hatch='//')
ax[1].bar(barx+0.0,mdeltaM_PGFnorm,bottom=mdeltaM_advnorm,color=barcol[2],width=barw,hatch='//')
ax[1].bar(barx+bari,msadeltaM_advpgfnorm,color='blue',width=barw,hatch='xx')
ax[1].set_xticks(barx, ['H300', 'H500', 'H1000'],rotation=-45)
ax[1].set_ylabel(r'$\Delta M_{\rm adv} + \Delta M_{\rm PGF}$',**yd,fontsize=15)
ax[1].set_yticks([0,2,4,6,8,10,12])
# Make empty hatches for plot

ax[1].legend(loc='upper left',fontsize=10,
          ncol=1, fancybox=True, shadow=True,handlelength=2)
mPGF_error = (mdeltaM_PGFnorm-deltaM_PGFnorm)/deltaM_PGFnorm*100
ax[1].text(1.1,9,r"%.1f"%(sa_Merror[0]) + "%",ha='center', fontsize=13)
ax[1].text(2.1,7.8,r"%.1f"%(sa_Merror[1]) + "%",ha='center', fontsize=13)
ax[1].text(3.1,7,r"%.1f"%(sa_Merror[2]) + "%",ha='center', fontsize=13)
ax[1].set_ylim([0,13])
fig.text(0.14,0.92,r"$\bf{(a)}$",ha='center', fontsize=15)
fig.text(0.74,0.92,r"$\bf{(b)}$",ha='center', fontsize=15)
fig.savefig('ap_combined.pdf',bbox_inches='tight')



################################################################
## deltaM_coriolis #############################################
################################################################
vavg = []
vavg0 = []
deltaM_cor = []

for i in range(len(H)):
    # Volume integral of V (main)
    datai = datasets[i]
    vavg.append(datai.mean()['v'])
    
    # Volume integral of V (precursor)
    precursori = precursors[i]
    vc = precursori['v'][precursori['z'] < HF]
    vavg0.append(np.mean(vc))
    
    # Calculate the momentum addition
    deltaM_cor.append(Vcv*rho*fc*(vavg[i] - vavg0[i]))

deltaM_cor = np.array(deltaM_cor)
deltaM_cornorm = deltaM_cor/(W*L*tauw0)

# Print LaTex table data for Coriolis force
print('------- Cor prediction -------')
for i in range(len(H)):
    print('H%d'%(H[i]) + ' & ' +
          '$%.2f$'%((2*fc*HF)/(Cf0[i]*UF0[i])) + ' & ' +  # 2/(Cf0*Ro_F0)
          '$%.2f$'%((vavg[i]-vavg0[i])/UF0[i]) + ' & ' +
          '$%.2f$'%(deltaM_cornorm[i]) + ' & ' +
          '$%.2f$'%(0*deltaM_cornorm[i]) + '\\' +'\\')
print(' ')




######################################################################
## deltaM_turb (aka deltaM_str) ######################################
######################################################################
# Contributions of the surfaces ########
fig = plt.figure(figsize=(7,4))
str_terms = [fronts,rears,norths,souths,tops]
str_labels = ['front', 'rear', 'north', 'south', 'top']
str_colors = plt.cm.viridis(np.linspace(0,1,len(str_terms)))
for i in range(len(str_terms)):
    plt.plot(H, str_terms[i]/(W*L*tauw0),'s-',label=str_labels[i], color=str_colors[i])
plt.plot(H,-taut0/tauw0,'ms--',label='top (precursor)')
plt.xlabel('$H$ [m]')
plt.ylabel(r'$\dfrac{\int \tau_{1j} dA_j}{X_{F0}}$',**yd)
fig.legend(loc='center left',fontsize=14,bbox_to_anchor=(0.91, 0.5),
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=2)
fig.savefig('stress_parts.pdf',bbox_inches='tight')

# Total stress contribution
deltaM_strnorm = (totals - Vcv*Xstr0)/(W*L*tauw0)


## deltaM_turb sub-model #########################
M3 = msadeltaM_advpgfnorm             # Model of (Madv+Mpgf)/MF0 with AP approximation
h1 = h005_abl/h005_farm_tauw1         # Exact h0/h (used in Fig.13b in article of shear stress profiles)
h2 = beta                             # Model h0/h

# Calculate model deltaM_turb
def mstress(Min,hin):
    # Total momentum availability factor
    tmpM = (1 + Min - taut0/tauw0)/((1 - taut0/tauw0)*hin)
    # Contribution from stress divergence normalized by MF0
    tmpMstr = tmpM + tmpM*(taut0/tauw0 - 1)*hin - taut0/tauw0
    return tmpMstr
mdeltaM_strnorm = mstress(M3,h2)


# deltaM_turb plot for article
fig = plt.figure(figsize=(9, 4))
gs = gridspec.GridSpec(1, 3)
gs.update(wspace=1)
ax = [plt.subplot(gs[0, :2]),  plt.subplot(gs[0, 2:])]
ax[0].plot(H,h1,'ks-',color=adv_colors[0], label=r'LES ($h$ based on 5% of $\tau_{w1}$)')
ax[0].plot(H,h005_abl/h005_farm_tauw0,'ks:',color=adv_colors[0], label=r'LES ($h$ based on 5% of $\tau_{w0}$)')
ax[0].plot(H,beta,'ks--',color=adv_colors[0], label=r'Model')
ax[0].set_xlabel('$H$ [m]')
ax[0].set_ylim([0.6,1.3])
ax[0].set_ylabel(r'$\dfrac{h_0}{h}$',**yd,labelpad=0)
ax[0].legend(loc='upper left',fontsize=14,
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=2)
bars = 1.
barw = 0.3
barx = np.array([bars,2*bars,3*bars])
barcol = plt.cm.viridis(np.linspace(0,1,5))
ax[1].bar(barx-0.2,deltaM_strnorm,color=barcol[3],width=barw,label=r'LES')
ax[1].bar(barx+0.2,mdeltaM_strnorm,color=barcol[3],width=barw,label=r'Model',hatch='xx')
ax[1].set_xticks(barx, ['H300', 'H500', 'H1000'],rotation=-45)
ax[1].set_ylabel(r'$\Delta M_{\rm turb}$',**yd)
ax[1].set_yticks([0,1,2,3,4,5,6,7])
ax[1].legend(loc='upper left',fontsize=10,
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=2)
mstr_error = (mdeltaM_strnorm-deltaM_strnorm)/deltaM_strnorm*100
ax[1].text(1.1,3.8,"+" + r"%.0f"%(mstr_error[0]) + "%",ha='center', fontsize=13)
ax[1].text(2.1,5.5,"+" + r"%.0f"%(mstr_error[1]) + "%",ha='center', fontsize=13)
ax[1].text(3.1,6.7,r"%.0f"%(mstr_error[2]) + "%",ha='center', fontsize=13)
ax[1].set_ylim([0,8])
fig.text(0.14,0.92,r"$\bf{(a)}$",ha='center', fontsize=15)
fig.text(0.75,0.92,r"$\bf{(b)}$",ha='center', fontsize=15)
fig.savefig('Mturb_bars.pdf',bbox_inches='tight')



##################################################################################
## Linearizations (to go from M_KDN1 to M_KDN2 and M_KDN3) #######################
##################################################################################
M_KDN1 = (1 + HF/(L*Cf0)*(1-beta**2) - taut0/tauw0)/(beta*(1 - taut0/tauw0))          # M_KDN1
M_KDN2 = (1 + h0/(L*Cf0)*(1-beta**2))/beta                                            # M_KDN2
M_KDN2_i = 1 + 1.18*(1-beta) + h0/(L*Cf0)*(2*(1-beta) - (1-beta)**2)/(1-(1-beta))     # Add linearization (i)
M_KDN2_i_and_ii = 1 + 1.18*(1-beta) + h0/(L*Cf0)*(2.18*(1-beta))                      # Add linearization (i) and (ii): This is M_KDN3.

fig, axes = plt.subplots(1, 3, figsize=(6, 3.5), sharey=True)

# Make colors with colormap viridis
cmap = cm.get_cmap('viridis', 6)
colors_bar = [cmap(i) for i in range(cmap.N)]

# Bar plot parameters
barx = np.array([0])
barw = 0.17
b1off = 0.33
b2off = 0.11
b3off = -0.11
b4off = -0.33

# Create bar plots for each H
for i in range(len(H)):
    ax = axes[i]
    
    # Bar 0 (M_KDN1)
    ax.bar(barx-b1off,M_KDN1[i],color=colors_bar[0],width=barw,label=r'$M_{\rm KDN1}$' if i==0 else "")

    # Bar 1 (M_KDN2)
    ax.bar(barx-b2off,M_KDN2[i],color=colors_bar[1],width=barw,label=r'$M_{\rm KDN2}$' if i==0 else "")

    # Bar 2 (M_KDN2 + lin. (i))
    ax.bar(barx-b3off,M_KDN2_i[i],color=colors_bar[2],width=barw,label=r'$M_{\rm KDN2}$ + lin. (i)' if i==0 else "")

    # Bar 3 (M_KDN2 + lin. (i) + lin. (ii) = M_KDN3)
    ax.bar(barx-b4off,M_KDN2_i_and_ii[i],color=colors_bar[3],width=barw,label=r'$M_{\rm KDN2}$ + lin. (i) + (ii) (= $M_{\rm KDN3}$)' if i==0 else "")

    # Settings
    ax.set_xticks([])
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([0, 26.0])
    ax.set_title('H%d'%(H[i]),fontsize=14)
    ax.grid(axis='x', visible=False)

    if i == 0:
        ax.set_ylabel(r'$M$', **yd)

    # Add a label (e.g. (a), (b), etc.) to each subplot in the top-left corner
    ax.text(0.00, 1.07, f'({chr(97 + i)})', transform=ax.transAxes,
               fontsize=11, fontweight='bold', va='top')

fig.legend(loc='center left',fontsize=12,bbox_to_anchor=(0.92, 0.5),
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=1.5)

fig.subplots_adjust(wspace=0.35)
fig.savefig('appendix_model.pdf', bbox_inches='tight')



##########################################################################
######## SUMMARY OF M_KDN1 BAR PLOT ######################################
##########################################################################
# M_exact summed from its contributions
totRHS = 1+deltaM_advnorm+deltaM_PGFnorm+deltaM_strnorm+deltaM_cornorm

# Plot
fig, axes = plt.subplots(1, 3, figsize=(5, 3.5), sharey=True)

# Make colors with colormap viridis
cmap = cm.get_cmap('viridis', 5)
colors_bar = [cmap(i) for i in range(cmap.N)]
col_les = 'k'
col_old = colors_bar[1]
col_new = colors_bar[2]

# Bar plot parameters
barone = np.ones_like(H)
barx = np.array([0])
barw = 0.25
baro = 0.09
b2off = 2*baro
b3off = 0
b4off = -2*baro

# Create bar plots for each H
for i in range(len(H)):
    ax = axes[i]
    
    # Bar 1 (LES)
    ax.bar(barx-b2off,barone[i],color=barcol[0],width=barw,label='$1$' if i==0 else "")
    ax.bar(barx-b2off,deltaM_advnorm[i],bottom=barone[i],color=barcol[1],width=barw, label=r'$\Delta M_{\rm adv}$ (LES)' if i==0 else "")
    ax.bar(barx-b2off,deltaM_PGFnorm[i],bottom=barone[i]+deltaM_advnorm[i],color=barcol[2],width=barw, label=r'$\Delta M_{\rm PGF}$ (LES)' if i==0 else "")
    ax.bar(barx-b2off,deltaM_strnorm[i],bottom=barone[i]+deltaM_advnorm[i]+deltaM_PGFnorm[i],color=barcol[3],width=barw, label=r'$\Delta M_{\rm turb}$ (LES)' if i==0 else "")
    ax.bar(barx-b2off,deltaM_cornorm[i],bottom=barone[i]+deltaM_advnorm[i]+deltaM_PGFnorm[i]+deltaM_strnorm[i],color=barcol[4],width=barw, label=r'$\Delta M_{\rm cor}$ (LES)' if i==0 else "")

    # Bar 2 (model)
    b4h = 'xx'
    ax.bar(barx-b4off,barone[i],color=barcol[0],width=barw, hatch=b4h)
    ax.bar(barx-b4off,msadeltaM_advpgfnorm[i],bottom=barone[i],color='blue',width=barw, hatch=b4h, label=r'$\Delta M_{\rm adv} + \Delta M_{\rm PGF}$ (KDN1)' if i==0 else "")
    ax.bar(barx-b4off,mdeltaM_strnorm[i],bottom=barone[i]+msadeltaM_advpgfnorm[i],color=barcol[3],width=barw, hatch=b4h, label=r'$\Delta M_{\rm turb}$ (KDN1)' if i==0 else "")

    # Annotate the error of the model bar compared to LES bar
    total_les = totRHS[i]
    total_model = barone[i] + msadeltaM_advpgfnorm[i] + mdeltaM_strnorm[i]
    model_error = (total_model - total_les) / total_les * 100
    ax.text(barx[0]-b4off, total_model + 0.5, f"{model_error:+.1f}%", ha='center', fontsize=11)

    # Use "$M_{exact}$" and "$M_{model}$" as xlabels under the bars
    ax.set_xticks([barx[0]-b2off, barx[0]-b4off], labels=[r'$M_{\rm exact}$', r'$M_{\rm KDN1}$'], rotation=-45)

    # Settings
    ax.set_xlim([-0.4, 0.4])
    ax.set_ylim([0, 14.0])
    ax.set_title('H%d'%(H[i]),fontsize=14)
    ax.grid(axis='x', visible=False)

    if i == 0:
        ax.set_ylabel(r'$M$', **yd)

    # Add a label (e.g. (a), (b), etc.) to each subplot in the top-left corner
    ax.text(0.00, 1.07, f'({chr(97 + i)})', transform=ax.transAxes,
               fontsize=11, fontweight='bold', va='top')

fig.legend(loc='center left',fontsize=12,bbox_to_anchor=(0.94, 0.5),
          ncol=1, fancybox=True, shadow=True,scatterpoints=1, handlelength=1.5)

fig.subplots_adjust(wspace=0.35)
fig.savefig('M_split.pdf', bbox_inches='tight')



##################################################################################
########## Print LaTex table data (table 2 in article) ###########################
##################################################################################
for i in range(len(H)):
    print('H%d'%(H[i]) + ' & ' +
          '$%.2f$'%(beta[i]) + ' & ' +                                     # beta
          '$%.2f$'%(beta0[i]) + ' & ' +                                    # beta_local(0)
          '$%.2f$'%(betaL[i]) + ' & ' +                                    # beta_local(L) 
          '$%.2f$'%(np.log(tauw[i]/tauw0[i])/np.log(beta[i])) + ' & ' +    # gamma
          '$%.2f$'%(farmT[i]/(0.5*Nt*Ad*(beta[i]*UF0[i])**2)) + ' & ' +    # C_T^*
          #'$%.2f$'%(farmP[i]/(0.5*Nt*Ad*(beta[i]*UF0[i])**3)) + ' & ' +   # C_P^*
          #'$%.2f$'%(farmP[i]/(0.5*Nt*Ad*(UF0[i])**3)) + ' & ' +           # C_PG
          '$%.3f$'%(dpdx_avg2[i]*L/G**2) + '\\' +'\\')                     # DeltaP^*/(rho*G^2)

