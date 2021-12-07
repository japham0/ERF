# ------------------  INPUTS TO MAIN PROGRAM  -------------------
max_step = 20

amrex.fpe_trap_invalid = 1

fabarray.mfiter_tile_size = 1024 1024 1024

# PROBLEM SIZE & GEOMETRY
geometry.is_periodic = 1 1 1
geometry.prob_lo     =  0     0     0
geometry.prob_hi     =  1     1     1    
amr.n_cell           = 16     16    16

# >>>>>>>>>>>>>  BC KEYWORDS <<<<<<<<<<<<<<<<<<<<<<
# Interior, UserBC, Symmetry, SlipWall, NoSlipWall
# >>>>>>>>>>>>>  BC KEYWORDS <<<<<<<<<<<<<<<<<<<<<<
erf.lo_bc       = "Interior"   "Interior"   "Interior"
erf.hi_bc       = "Interior"   "Interior"   "Interior"

# TIME STEP CONTROL
erf.fixed_dt       = 5e-5    # fixed time step
#erf.cfl            = 0.9     # cfl number for hyperbolic system
erf.init_shrink    = 1.0     # scale back initial timestep
erf.change_max     = 1.05    # scale back initial timestep
erf.dt_cutoff      = 5.e-20  # level 0 timestep below which we halt

# DIAGNOSTICS & VERBOSITY
erf.sum_interval   = 1       # timesteps between computing mass
erf.v              = 1       # verbosity in ERF.cpp
amr.v                = 1       # verbosity in Amr.cpp
amr.data_log         = datlog

# REFINEMENT / REGRIDDING
amr.max_level       = 0       # maximum level number allowed
amr.ref_ratio       = 2 2 2 2 # refinement ratio
amr.regrid_int      = 2 2 2 2 # how often to regrid
amr.blocking_factor = 4       # block factor in grid generation
amr.max_grid_size   = 64
amr.n_error_buf     = 2 2 2 2 # number of buffer cells in error est

# CHECKPOINT FILES
amr.checkpoint_files_output = 0
amr.check_file      = chk        # root name of checkpoint file
amr.check_int       = 100        # number of timesteps between checkpoints

# PLOTFILES
amr.plot_files_output = 1
amr.plot_file       = plt        # root name of plotfile
amr.plot_int        = 20        # number of timesteps between plotfiles
amr.plot_vars        =  density rhoadv_0
amr.derive_plot_vars = pressure temp theta x_velocity y_velocity z_velocity

# SOLVER CHOICE
erf.use_state_advection = false
erf.use_momentum_advection = false
erf.use_thermal_diffusion = false
erf.alpha_T = 0.0
erf.use_scalar_diffusion = true
erf.alpha_C = 1.0
erf.use_momentum_diffusion = false
erf.use_pressure = false
erf.use_gravity = false

erf.les_type         = "None"
erf.dynamicViscosity = 0.0

erf.spatial_order = 2

# PROBLEM PARAMETERS
prob.rho_0 = 1.0
prob.A_0 = 1.0
prob.B_0 = 0.0
prob.u_0 = 0.0
prob.v_0 = 0.0
prob.uRef = 0.0
prob.prob_type = 10

# INTEGRATION
## integration.type can take on the following values:
## 0 = Forward Euler
## 1 = Explicit Runge Kutta
integration.type = 1

## Explicit Runge-Kutta parameters
#
## integration.rk.type can take the following values:
### 0 = User-specified Butcher Tableau
### 1 = Forward Euler
### 2 = Trapezoid Method
### 3 = SSPRK3 Method
### 4 = RK4 Method
integration.rk.type = 3

## If using a user-specified Butcher Tableau, then
## set nodes, weights, and table entries here:
#
## The Butcher Tableau is read as a flattened,
## lower triangular matrix (but including the diagonal)
## in row major format.
integration.rk.weights = 1
integration.rk.nodes = 0
integration.rk.tableau = 0.0