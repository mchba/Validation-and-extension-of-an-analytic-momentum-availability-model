import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import cm
import sys
sys.path.append('../../../') # To be able to utils.py

mpl.style.use('classic')
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["legend.scatterpoints"] = 1
plt.rcParams["legend.numpoints"] = 1
plt.rcParams['grid.linestyle'] = ':' # Dotted gridlines
mpl.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 17
plt.rcParams['axes.grid']=False
mpl.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['figure.dpi'] = 300
yd = dict(rotation=0,ha='right',va='center') 
plt.close('all')


'''
    Sketched used to illustate different shear stress profiles in paper
'''


### Example plot to show idea of connection between hx0 and hx0tilde
fig = plt.figure(figsize=(5,4))
z = np.linspace(0,1000,100)
HF_example = 2.5*119.0
alpha_example = 1.4
h0_example = 1000
alphax_example = 1.8
hx0_example = 800
hx0tilde_example = HF_example/(1 - (1 - (HF_example/hx0_example))**alphax_example)

xticks = [0,1]
yticks = [HF_example, hx0tilde_example, hx0_example, h0_example]

cmap = cm.get_cmap('viridis', 3)
colors = [cmap(i) for i in range(cmap.N)]

plt.ylabel('$z$ [m]', **yd)
plt.plot((1-z/h0_example)**alpha_example, z, color=colors[0], label=r'$\frac{|\tau|_{0}(z)}{|\tau|_{w0}} = \left(1-\frac{z}{h_{0}}\right)^{p}$')
plt.plot((1-z/hx0_example)**alphax_example, z, color=colors[1], label=r'$\frac{\tau_{x0}(z)}{\tau_{xw0}} = \left(1-\frac{z}{h_{x0}}\right)^{p_x}$')
plt.plot((1 - z/hx0tilde_example), z, color=colors[2], label=r'$\frac{\tilde{\tau}_{x0}(z)}{\tau_{xw0}} = 1 - \frac{z}{\tilde{h}_{x0}}$')
plt.annotate('Control volume', xy=(0.1, HF_example), xytext=(0.1, HF_example-110), fontsize=14)
plt.legend(fontsize=14)
plt.ylim([0, h0_example+100])
plt.xlim(left=0)
plt.xticks(xticks)
plt.yticks(yticks,[r'$H_F$',r'$\tilde{h}_{x0}$',r'$h_{x0}$',r'$h_{0}$'])
plt.axhspan(0, HF_example, color='gray', alpha=0.3, hatch='//')
plt.savefig('article_sketch_shear_stress.pdf', bbox_inches='tight')

