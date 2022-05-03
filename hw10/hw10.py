# %%
# CS6300 HW10 Ryan Dalby

import matplotlib.pyplot as plt
import numpy as np

# %% 
# 2.3
p_vals = np.linspace(0,1,100)
V_stay = (1/2) * p_vals - (1/4)
V_go = (-1/3) * p_vals + (2/3) 
plt.figure()
plt.plot(p_vals, V_stay, label='V_stay = (1/2)(p_1) - (1/4)')
plt.plot(p_vals, V_go, label='V_go = (-1/3)(p_1) + (2/3)')
plt.xlim((0,1))
plt.xlabel('p_1')
plt.legend()
plt.savefig('./latex/images/2_3.png')
plt.show()
