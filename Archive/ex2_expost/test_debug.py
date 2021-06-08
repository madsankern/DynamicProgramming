# load general packages
import numpy as np
import matplotlib.pyplot as plt


# load modules related to this exercise
import tools 
import model_dc

par = model_dc.setup()
par = model_dc.create_grids(par)
sol = model_dc.solve(par)
