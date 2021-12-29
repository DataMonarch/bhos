import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

x1 = pd.read_excel('x1.xlsx')
x1

x1[x1 == -999] = np.NaN
x1[x1 == ' '] = np.NaN
x1.info()
x1.describe()

%matplotlib qt

x1[['core_interconnected_porosity', 'phi_dens', 'phi_sonic']].corr()

fig, [ax, ax1, ax2] = plt.subplots(1, 3, sharey=True)
ax.plot( x1['GR'], x1['log_depth_tvd'], 'k', linewidth=0.5)
ax.set_ylabel('Depth TVD')
ax.set_xlabel('GR')
ax.grid(which='major', color='#DDDDDD', linewidth=1)
ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
ax.minorticks_on()

ax1.plot(x1['sonic'], x1['log_depth_tvd'], 'r', linewidth=0.5)
ax1.set_xlabel('SONIC')
ax1.grid(which='major', color='#DDDDDD', linewidth=1)
ax1.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
ax1.minorticks_on()


ax2.plot(x1['phi_dens'], x1['log_depth_tvd'], 'b', linewidth=0.5, label='phi_dens')
ax2.plot(x1['phi_sonic'], x1['log_depth_tvd'], 'r', linewidth=0.5, label='phi_sonic')
ax2.plot(x1['core_interconnected_porosity'], x1['core_depth_tvd'], 'g^', linewidth=2, alpha=0.5, label='core_interconnected_porosity')
ax2.set_xlabel('Porosity')

ax2.grid(which='major', color='#DDDDDD', linewidth=1)
ax2.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
ax2.minorticks_on()
ax2.legend()

plt.show()
