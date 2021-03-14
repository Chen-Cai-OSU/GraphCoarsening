# Created at 2020-06-18
# Summary: test tight layout

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d

fig = plt.figure()
ax = fig.gca(projection='3d')

N = 100
x = np.random.random(N)
y = np.random.random(N)
z = np.random.random(N)

ax.scatter(x, y, z)

# The fix
# for spine in ax.spines.values():
#     spine.set_visible(False)

plt.tight_layout()

plt.savefig('scatter.png')