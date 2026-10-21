# Finalised

# This code produces main effects plots for unscaled kernels
# Adjust kernel settings as needed

KERNEL = "RBF"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import InputStandardize

# INITISALIING GP MODEL ====================================

# Kernel aka covar_module is exchangeable, all Scaled ARD
def initialize_GPmodel(train_X, train_y, covar_type=None, noise=None): 
    """Initialize a GP model where
    trian_X.shape is [N, d]
    train_y.shape is [N, 1]
    Return the fitted GP model with selected kernel and noise level.
    """
    # Get the dimensions of train_X and train_y to standardize inputs
    train_X_dim = train_X.shape[-1]
    train_y_dim = train_y.shape[-1]

    # ard_num_dims=train_X.shape[-1]

    # Take out ARD
    ard_num_dims = None

    if covar_type == "RBF":
        covar_module = RBFKernel(ard_num_dims=ard_num_dims)
    elif covar_type == "Matern_52":
        covar_module = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims)
    elif covar_type == "Matern_32":
        covar_module = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims)
    else:
        print("Invalid kernel.")

    # Customise noise levels
    if noise == None:
        likelihood = None
    else:
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(noise))

    # Train the GP model
    gp = SingleTaskGP(
        train_X, train_y,
        input_transform=InputStandardize(d=train_X_dim),
        outcome_transform=Standardize(m=train_y_dim),
        covar_module=covar_module,
        # Override likelihood if noise level is specified
        likelihood=likelihood
    )

    # Objective function
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

    return mll, gp

# OBTAIN OVERALL MLL AND FINAL FITTED GP MODEL =======================

def final_fit(inputs, outputs, covar_type=None, noise=None):

    # Initialise gp model and train given data
    mll, gp = initialize_GPmodel(inputs, outputs, covar_type=covar_type, noise=noise)

    # Fit model
    fit_gpytorch_mll(mll)

    # Evaluate scalar mll
    with torch.no_grad():
        output_MVN = gp(X)
        scalar_mll = mll(output_MVN, outputs).sum()
    
    return mll, gp, scalar_mll.item()

# SAMPLING FOR MAIN EFFECTS PLOT =======================================================

# Gets main effect plot data for Time
def gp_main_effect_time(model, Na_value, n_points: int=100):
    """
    Evaluate GP predictions while varying Time, Na is specified.
    Cu = Fe = Mn = Ni = 0.25 (fixed value).
    Returns input_values, mean, std.
    """

    # Generates input values in order (Time, Na, Cu, Fe, Mn, Ni_
    # Varies Time (column=0) between 3 and 12
    inputs = np.array([
        np.linspace(3, 12, n_points), # Time
        np.full(n_points, Na_value), # Na
        np.full(n_points, 0.25), # Cu
        np.full(n_points, 0.25), # Fe
        np.full(n_points, 0.25), # Mn
        np.full(n_points, 0.25) # Ni
    ])

    # Transpose and convert to tensor to pass through model
    inputs_torch = torch.tensor(inputs.T, dtype=torch.double)

    model.eval()

    with torch.no_grad():
        posterior = model.posterior(inputs_torch)

    mean = posterior.mean.squeeze().numpy()
    std = posterior.variance.sqrt().squeeze().numpy()

    return inputs, mean, std

# Gets main effect plot data for Na
def gp_main_effect_na(model, Time_value, n_points: int=100):
    """
    Evaluate GP predictions while varying Na, whiel Time is specified.
    Cu = Fe = Mn = Ni = 0.25 (fixed value).
    Returns input_values, mean, std.
    """

    # Generates input values in order (Time, Na, Cu, Fe, Mn, Ni)
    # Varies Na (column=1) between 0.5 and 1.0
    inputs = np.array([
        np.full(n_points, Time_value), # Time
        np.linspace(0.5, 1.0, n_points), # Na
        np.full(n_points, 0.25), # Cu
        np.full(n_points, 0.25), # Fe
        np.full(n_points, 0.25), # Mn
        np.full(n_points, 0.25) # Ni
    ])

    # Transpose and convert to tensor to pass through model
    inputs_torch = torch.tensor(inputs.T, dtype=torch.double)

    model.eval()

    with torch.no_grad():
        posterior = model.posterior(inputs_torch)

    mean = posterior.mean.squeeze().numpy()
    std = posterior.variance.sqrt().squeeze().numpy()

    return inputs, mean, std

# Gets main effect plot data for a specified TM, given a value of Na and Time
def gp_main_effect_tm(model, tm, Time_value, Na_value, n_points: int=100):
    """
    Evaluate GP predictions while varying Na. Time can be varied.
    Cu = Fe = Mn = Ni = 0.25 (fixed value).
    Returns: input_values, mean, std.
    """

    # Set TM to correct column
    if tm == "Cu":
        tm_idx = 2
    elif tm == "Fe":
        tm_idx = 3
    elif tm == "Mn":
        tm_idx = 4
    elif tm == "Ni":
        tm_idx = 5

    tm_values = np.linspace(0, 1.0, n_points)
    other_tm_values = (1 - tm_values) / 3

    # Generate input values in order (Time, Na, Cu, Fe, Mn, Ni)
    # Start with default values for all rows
    inputs = np.array([
        np.full(n_points, Time_value), # Time
        np.full(n_points, Na_value), # Na
        other_tm_values, # Cu
        other_tm_values, # Fe
        other_tm_values, # Mn
        other_tm_values # Ni
    ])

    # Change selected TM row to sweep through feature space
    inputs[tm_idx] = tm_values

    # Transpose and convert to tensor to pass through model
    inputs_torch = torch.tensor(inputs.T, dtype=torch.double)

    model.eval()

    with torch.no_grad():
        posterior = model.posterior(inputs_torch)

    mean = posterior.mean.squeeze().numpy()
    std = posterior.variance.sqrt().squeeze().numpy()

    return inputs, mean, std

# LOAD FULL DATASET ======================================

df = pd.read_csv("active_learning_model/original_dataset/df_exp_semi_refined.csv")

df_success = df[df["Cat_Success"] == "Successful"]

dropped_columns = ["P2O3ratio", "UID", "Chemical", "O", "Cat_Success", "Temperature"]
X = torch.tensor(df_success.drop(columns=dropped_columns).to_numpy())

dropped_columns2 = ["P2O3ratio", "UID", "Chemical", "O", "Cat_Success", "Temperature", "Na", "Cu"]
X_removed = torch.tensor(df_success.drop(columns=dropped_columns2).to_numpy())

# set output value to P2O3 ratio
y = torch.tensor(df_success["P2O3ratio"].to_numpy()).reshape(-1, 1)

# FITTING GP MODEL =======================================================

# Refit GP with all data for scalar MLL value
mll, gp, scalar_mll = final_fit(X, y, covar_type="KERNEL")

# GENERATE DATA FOR TIME PLOTS ========================================================

Na_values = [0.5, 0.75, 1.0]

for value in Na_values:
    Time_inputs, Time_mean, Time_std = gp_main_effect_time(gp, Na_value=value)

    Time_data = pd.DataFrame({
        "Time": Time_inputs[0],
        "Na": Time_inputs[1],
        "Cu": Time_inputs[2],
        "Fe": Time_inputs[3],
        "Mn": Time_inputs[4],
        "Ni": Time_inputs[5],
        "P2O3ratio": Time_mean,
        "P2O3std": Time_std,
    })
    Time_data.index.name = "ID"

    Time_data.to_csv(f"active_learning_model/surrogate_design/{KERNEL}_unscaled_main_effects_plots/data/{KERNEL}_unscaled_main_effects_time_{value}Na_data.csv")

# GENERATE DATA FOR NA PLOTS ========================================================

Time_values = [3.0, 6.0, 12.0]

for value in Time_values:
    Na_inputs, Na_mean, Na_std = gp_main_effect_na(gp, Time_value=value)

    Na_data = pd.DataFrame({
        "Time": Na_inputs[0],
        "Na": Na_inputs[1],
        "Cu": Na_inputs[2],
        "Fe": Na_inputs[3],
        "Mn": Na_inputs[4],
        "Ni": Na_inputs[5],
        "P2O3ratio": Na_mean,
        "P2O3std": Na_std,
    })
    Na_data.index.name = "ID"

    Na_data.to_csv(f"active_learning_model/surrogate_design/{KERNEL}_unscaled_main_effects_plots/data/{KERNEL}_unscaled_main_effects_Na_{value}time_data.csv")

# GENERATING TM DATA FOR MAIN EFFECTS PLOTS ======================================================

tm_values = ["Cu", "Fe", "Mn", "Ni"]

for tm in tm_values:
    for time in Time_values:
        for na in Na_values:

            TM_inputs, TM_mean, TM_std = gp_main_effect_tm(gp, tm=tm, Time_value=time, Na_value=na)
            
            TM_data = pd.DataFrame({
                "Time": TM_inputs[0],
                "Na": TM_inputs[1],
                "Cu": TM_inputs[2],
                "Fe": TM_inputs[3],
                "Mn": TM_inputs[4],
                "Ni": TM_inputs[5],
                "P2O3ratio": TM_mean,
                "P2O3std": TM_std,
            })
            TM_data.index.name = "ID"

            TM_data.to_csv(f"active_learning_model/surrogate_design/{KERNEL}_unscaled_main_effects_plots/data/{KERNEL}_unscaled_main_effects_{tm}_{na}Na_{time}time_data.csv")
