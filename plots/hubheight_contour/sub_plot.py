import numpy as np
import xarray
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.style.use('classic')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["legend.numpoints"] = 1
plt.rcParams['grid.linestyle'] = ':' # Dotted gridlines
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 17
plt.rcParams['axes.grid']=True
plt.rcParams['figure.dpi'] = 300
yd = dict(rotation=0,ha='right',va='center') 
plt.close('all')

'''
    Hub-height contour plots of velocity and pressure for the cases:
    - H300
    - H500
    - H1000
    - H500 aligned
    - H500 double-spacing
    - H500 half-farm
'''


########### PARAMETERS ########################
D = 198
G = 10 # Geostrophc wind speed [m/s]



##################### Flow data at hub height ###################
cases = ['H300','H500','H1000','H500_aligned','H500_double-spacing','H500_half-farm']
paths = ['../../data_lanzilao2025/H300-C5-G4/main_hubheight.nc',
         '../../data_lanzilao2025/H500-C5-G4/main_hubheight.nc',
         '../../data_lanzilao2025/H1000-C5-G4/main_hubheight.nc',
         '../../data_lanzilao2024/H500-C5-G4_aligned/main_hubheight.nc',
         '../../data_lanzilao2024/H500-C5-G4_double_spacing/main_hubheight.nc',
         '../../data_lanzilao2024/H500-C5-G4_half_farm/main_hubheight.nc']
flows = {}
P_in = np.zeros(len(cases))
for i in range(len(cases)):
    case = cases[i]
    flows[case] = xarray.open_dataset(paths[i])
    P_in[cases.index(case)] = flows[case]['p'].isel(x=0).mean().values



############ Make a 4x3 subplot (with velocity in the first two rows and pressure in last two row) ######################
fig, axes = plt.subplots(4, 3, figsize=(12, 14), sharex=False, sharey='row')
plt.subplots_adjust(hspace=0.5, wspace=0.2)


# Flatten axes for easy indexing
ax = axes.flatten()

# Loop through cases and create subplots
for i, case in enumerate(cases):
    U = flows[case]['u'].values
    P = flows[case]['p'].values
    X, Y = np.meshgrid(flows[case]['x'].values/1000, flows[case]['y'].values/1000)

    # Streamwise velocity
    im_vel = ax[i].pcolormesh(X, Y, U/G, cmap='viridis',
                              shading='auto', vmin=5/G, vmax=10/G,   
                              rasterized=True)
    ax[i].set_title(f'{case}',fontsize=15)
    ax[i].axis('scaled')
    ax[i+6].axis('scaled')
    ax[i].set_xlabel('$x$ [km]')
    ax[i+6].set_xlabel('$x$ [km]')
    ax[i].set_xticks(np.arange(0,41,10))
    ax[i+6].set_xticks(np.arange(0,41,10))
    
    if i < 3:
        ax[i].set_xlim([0, 40])
        ax[i].set_ylim([35, 65])
        ax[i+6].set_xlim([0, 40])
        ax[i+6].set_ylim([35, 65])
    if i < 6 and i >=3:
        ax[i].set_xlim([0, 40])
        ax[i].set_ylim([0, 30])
        ax[i+6].set_xlim([0, 40])
        ax[i+6].set_ylim([0, 30])
    if i == 0 or i ==3:
        ax[i].set_ylabel('$y$ [km]', **yd)
        ax[i+6].set_ylabel('$y$ [km]', **yd)
    if i == 0:
        ax[i].set_yticks(np.arange(35,70,10))
        ax[i+6].set_yticks(np.arange(35,70,10))
    if i ==3:
        ax[i].set_yticks(np.arange(0,35,10))
        ax[i+6].set_yticks(np.arange(0,35,10))

    # Pressure
    im_pres = ax[i+6].pcolormesh(X, Y, (P - P_in[i])/G**2, cmap='seismic',
                                shading='auto',
                                rasterized=True, vmin=-20/G**2, vmax=20/G**2)
    ax[i+6].set_title(f'{case}',fontsize=15)

for i in range(12):
    # Add a label (e.g. (a), (b), etc.) to each subplot in the top-left corner
    ax[i].text(0.02, 1.12, f'({chr(97 + i)})', transform=ax[i].transAxes,
               fontsize=14, fontweight='bold', va='top')
    
# Colorbars
cbar_vel = fig.colorbar(im_vel, ax=ax[0:6], orientation='vertical', fraction=0.046, pad=0.03, extend='both')
cbar_vel.set_label('$U/G$', fontsize=15, rotation=0, ha='left')
cbar_pres = fig.colorbar(im_pres, ax=ax[6:12], orientation='vertical', fraction=0.046, pad=0.03, extend='both')
cbar_pres.set_label(r'$\dfrac{P^* - P_{\rm inlet}^*}{\rho G^2}$', fontsize=15, rotation=0, ha='left')
    
# Save figure
fig.savefig('out_subplot_hubheight_contour.png', bbox_inches='tight', dpi=300)







