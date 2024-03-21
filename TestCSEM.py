import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

import emg3d
import pygimli as pg

# Requires currently branch inversion:
# pip install git+https://github.com/emsig/emg3d@inversion
from emg3d.inversion.pygimli import Inversion

print('Importing models and survey')
start_data = emg3d.load('ModSynth_CSEM_start.h5') 
true_data = emg3d.load('ModSynth_CSEM_true.h5')
survey_data = emg3d.load('Surv_3x9_rot10_27src_W123_TxW_withofflineData_CF_24-03-06.h5')

start_model= start_data['model']
true_model = true_data['model']
survey = survey_data['survey']

survey = survey.select(sources = ['RxW11', 'RxW31', 'RxW51', 'RxW73', 'RxW93'], 
                       frequencies='f-1')

rec_coords = survey.receiver_coordinates()
src_coords = survey.source_coordinates()

start_grid = start_model.grid
true_grid = true_model.grid

print()
print('Grid:')
print(start_grid)

print()
print('Defining conductivity models')
# The model is resistivity. Change it to conductivity
# TODO: make this internally happen, so that pyGIMLi
# always gets a conductivity model!
start_LgCon_model = emg3d.Model(start_grid, start_model.property_x, mapping='LgConductivity')
true_LgCon_model = emg3d.Model(true_grid, true_model.property_x, mapping='LgConductivity')

# Convert to conductivity
StartModCond = 10**start_model.property_x
TrueModCond = 10**true_model.property_x

start_con_model = emg3d.Model(start_grid, StartModCond, mapping='Conductivity')
true_con_model = emg3d.Model(true_grid, TrueModCond, mapping='Conductivity')

print()
print('Defining gridding options')
gopts = {
    'properties': [0.3, 10, 1., 0.3],
    'min_width_limits': (160, 160, 20),
    'stretching': (None, None, [1.05, 1.5]),
    'domain': (
        [rec_coords[0].min()-100, rec_coords[0].max()+100],
        [rec_coords[1].min()-100, rec_coords[1].max()+100],
        [-2000,0]
    ),
    'center_on_edge': False,
    }

# Create an emg3d Simulation instance
sim = emg3d.simulations.Simulation(
    survey=survey,
    model=start_con_model,
    gridding='both', #'same',  # I would like to make that more flexible in the future
    gridding_opts= gopts, #{'vector': 'xyz'},
    max_workers=50,    # Adjust as needed
    receiver_interpolation='linear',  # Currently necessary for the gradient
    ### Tolerance: TODO: different ones for forward and adjoints
    solver_opts={'tol': 1e-3},                # Just for dev-purpose
    tqdm_opts=False,  # Switch off verbose progress bars
    #verb=3,
)

print()
print(sim)

print()
print('Defining inversion')

INV = Inversion(fop=sim, verbose=True) #, debug=True)
INV.inv.setCGLSTolerance(10)  # is _absolute_, not _relative_
INV.inv.setMaxCGLSIter(30)

#INV.dataTrans = pg.trans.TransSymLog(sim.survey.noise_floor)

# Create region markers
# Set marker -> air is 2, water is 1, subsurface is 0 ?

for c in INV.inv_mesh.cells():
    if c.center()[2] >= 200:
        c.setMarker(2)
    elif (c.center()[2] > 0) & (c.center()[2] < 200):
        c.setMarker(1)

# , zWeight=0.2
# , correlationLengths=[1000, 1000, 300])

INV.setRegularization(0, limits = (3e-3, 3e-1), zWeight=0.2) #, startModel=3e-2)
INV.setRegularization(1, limits = (3.43, 3.45), startModel=3.44) # , single=True)
INV.setRegularization(2, limits = (2e-8, 9e-7), startModel=1e-8)

# 0 only damping (minimum length; only useful with good starting model and isReference=True in INV)
# 1 1st deriv - smoothing
# 2 2nd deriv
# 10, 20 - mixed form

errmodel = INV.run(
    maxIter=1, # just to test
    lam=1,  # btw 1-100
    verbose=True,
    startModel=start_con_model.property_x.ravel('F'),
    isReference=True,
)

# Add inversion result to data;
# I should wrap these few lines into a function, as they are used above as well
idata = np.ones(sim.survey.shape, dtype=sim.data.observed.dtype)*np.nan
x = np.asarray(INV.response)
idata[sim.survey.isfinite] = x[:x.size//2] + 1j*x[x.size//2:]
sim.survey.data['inv'] = sim.survey.data.observed.copy(data=idata)

# Compute the 2-layer model as comparison
sim.clean('computed')
sim.model = start_con_model
sim.compute()  # <- puts it into 'synthetic'

print()
print('Storing results')
# Store final model as model to be saved with the simulation
sim.model = emg3d.Model(start_grid, np.array(INV.model), mapping='Conductivity')
sim.to_file('Simulation-CSEM-StartModel_Ref_5s.h5')



