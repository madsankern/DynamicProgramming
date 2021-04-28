# Load packages
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# load user written modules
import tools
import model
import vfi
import egm

# Setup
par = model.setup()

egm = egm.solve_EGM_2d(par)