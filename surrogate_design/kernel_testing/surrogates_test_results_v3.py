# Finalised

# Saves scores from kernel comparisons
# Gets data for P2O3_pred against P2O3_actual for all kernels
# Tests RBF, Matern 32, Matern 52 with and without: ARD, scaling, noise

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch

import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

from sklearn.model_selection import GroupKFold

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import InputStandardize

# DATASET ========================================================================================================

# dataset
df = pd.read_csv("active_learning_model/original_dataset/df_exp_semi_refined.csv")

df_success = df[df["Cat_Success"] == "Successful"]

dropped_columns = ["P2O3ratio", "UID", "Chemical", "O", "Cat_Success", "Temperature"]
X = torch.tensor(df_success.drop(columns=dropped_columns).to_numpy())

y = torch.tensor(df_success["P2O3ratio"].to_numpy()).reshape(-1, 1)

# define groups w/ same chemical formula
groups = df_success["Chemical"].values

# INITISALIING GP MODEL WITH ALL KERNEL VARIANTS =======================================================================

# interchangeable kernel
def initialize_GPmodel(train_X, train_y, scaled=True, ard=True, covar_type=None, noise=None): 
    """Initialize a GP model where
    trian_X.shape is [N, d]
    train_y.shape is [N, 1]
    Return the fitted GP model with selected kernel and noise level.
    """
    # Get the dimensions of train_X and train_y to standardize inputs
    train_X_dim = train_X.shape[-1]
    train_y_dim = train_y.shape[-1]

    # Number of ARD lengthscales is the number of dimensions for each input
    ard_num_dims=train_X.shape[-1]

    if scaled == True:
        if covar_type == "RBF":
            covar_module = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims if ard == True else None))
        elif covar_type == "Matern_52":
            covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=ard_num_dims if ard == True else None))
        elif covar_type == "Matern_32":
            covar_module = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=ard_num_dims if ard == True else None))
        else:
            print("Invalid kernel.")
    else:
        if covar_type == "RBF":
            covar_module = RBFKernel(ard_num_dims=ard_num_dims if ard == True else None)
        elif covar_type == "Matern_52":
            covar_module = MaternKernel(nu=2.5, ard_num_dims=ard_num_dims if ard == True else None)
        elif covar_type == "Matern_32":
            covar_module = MaternKernel(nu=1.5, ard_num_dims=ard_num_dims if ard == True else None)
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

# ===================== FUNCTIONS TO CALCULATE METRICS FOR CV =================================================================

# compute R^2 between predicted and test values, returns R^2 as scalar tensor ----------
def r2(y_test: torch.Tensor,
       y_pred: torch.Tensor
       ):

    ss_res = torch.sum((y_test - y_pred) ** 2)
    ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)

    if ss_tot == 0:
        return torch.tensor(float('nan'))

    r2 = 1 - ss_res / ss_tot
    return r2

# function for RMSE with tensor compatibility --------------------------------------------------------
def rmse(y_test: torch.Tensor,
       y_pred: torch.Tensor
       ):
    rmse = torch.sqrt(torch.mean((y_test - y_pred) ** 2))
    return rmse

# Generate grouped k folds
def sklearn_gkf_split(X, y, groups, n_splits, random_state=42):
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return gkf.split(X, y, groups)

# CV TO EVALUATE GP ======================================

# This version is modified to alter scaling, ARD and noise levels
def run_cv_gp_mod(
        X: torch.Tensor,
        y: torch.Tensor,
        groups: torch.Tensor,
        random_state: int=20,
        scaled: bool=True,
        ard: bool=True,
        covar_type=None,
        noise=None,
        n_splits: int=5
):
    """Takes inputs and output in tensor form.
    Applies GKFolds and calculates metrics.
    """
    print("Running cross validation for GP...")

    train_r2_scores = []
    train_rmse_scores = []
    test_r2_scores = []
    test_rmse_scores = []
    mll_values = []
    # train_prop = []
    # test_prop = []
    total_y_actual = []
    total_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(sklearn_gkf_split(X, y, groups, n_splits, random_state=random_state)):

        print(f"GP Fold {fold}...")

        X_train, X_test = X[train_idx], X[test_idx] # already tensors
        y_train, y_test = y[train_idx], y[test_idx]

        ### Don't need fold profiles in this case
        # O3_train, mixed_train, P2_train, O3_test, mixed_test, P2_test = fold_profile(y_train, y_test)
        # train_prop.append([O3_train, mixed_train, P2_train])
        # test_prop.append([O3_test, mixed_test, P2_test])

        # Train the model on folds
        mll_fold, gp_fold = initialize_GPmodel(X_train, y_train, scaled=scaled, ard=ard, covar_type=covar_type, noise=noise)

        # Fit GP model
        fit_gpytorch_mll(mll_fold)

        # Evaluate model scores
        with torch.no_grad():
            pred_X_test = gp_fold.posterior(X_test)
            
            ### Test error
            y_pred_unclipped = pred_X_test.mean

            # Clip y_pred by changing any predicted P2O3 values above or below (0,1) to 0 or 1 (physically possible values)
            y_pred = torch.clamp(y_pred_unclipped, min=0.0, max=1.0)

            test_r2_scores.append(r2(y_test, y_pred).item())
            test_rmse_scores.append(rmse(y_test, y_pred).item())

            ### Training error
            pred_X_train = gp_fold.posterior(X_train)
            y_pred_train_unclipped = pred_X_train.mean
            y_pred_train = torch.clamp(y_pred_train_unclipped, min=0.0, max=1.0)

            # calculate R^2 and RMSE per fold and update list of scores
            train_r2_scores.append(r2(y_train, y_pred_train).item())
            train_rmse_scores.append(rmse(y_train, y_pred_train).item())

            # update data to plot
            total_y_actual.extend(y_test.tolist())
            total_y_pred.extend(y_pred.tolist())

            # obtain negative MLL value per fold
            loss_fold = mll_fold(gp_fold(X_train), y_train).sum()
            mll_values.append(loss_fold)

    gp_results = ({
        "all_train_r2": train_r2_scores,
        "all_train_rmse": train_rmse_scores,
        # "fold_1_train_O3_mixed_P2_prop": train_prop[0],
        # "fold_2_train_O3_mixed_P2_prop": train_prop[1],
        # "fold_3_train_O3_mixed_P2_prop": train_prop[2],
        # "fold_4_train_O3_mixed_P2_prop": train_prop[3],
        # "fold_5_train_O3_mixed_P2_prop": train_prop[4],
        "train_avg_r2": np.mean(train_r2_scores),
        "train_avg_rmse": np.mean(train_rmse_scores),
        "all_test_r2": test_r2_scores,
        "all_test_rmse": test_rmse_scores,
        "test_avg_r2": np.mean(test_r2_scores),
        "test_avg_rmse": np.mean(test_rmse_scores),
        "avg_mll": np.mean(mll_values),
        # "fold_1_test_O3_mixed_P2_prop": test_prop[0],
        # "fold_2_test_O3_mixed_P2_prop": test_prop[1],
        # "fold_3_test_O3_mixed_P2_prop": test_prop[2],
        # "fold_4_test_O3_mixed_P2_prop": test_prop[3],
        # "fold_5_test_O3_mixed_P2_prop": test_prop[4]
    })

    return gp_results, total_y_actual, total_y_pred

# OBTAIN OVERALL MLL AND FINAL FITTED GP MODEL =======================

# modified to alter scaling, ARD and noise levels
def final_fit_mod(X, y, scaled: bool=True, ard: bool=True, covar_type=None, noise=None):

    # Initialise gp model and train given data
    mll, gp = initialize_GPmodel(X, y, scaled=scaled, ard=ard, covar_type=covar_type, noise=noise)

    # Fit model to obtain MLL
    fit_gpytorch_mll(mll)

    # Evaluate the MLL to get the actual objective value at all points
    with torch.no_grad():
        output_MVN = gp(X)
        scalar_mll = mll(output_MVN, y).sum()
    
    return mll, gp, scalar_mll.item()

# DEFINE KERNELS TO TEST ==========================================================================================

all_kernel_settings = [
    ("RBF", {"type": "RBF", "scaled": False, "ard": False}),
    ("RBF ARD", {"type": "RBF", "scaled": False, "ard": True}),
    ("Scaled RBF", {"type": "RBF", "scaled": True, "ard": False}),
    ("Scaled RBF ARD", {"type": "RBF", "scaled": True, "ard": True}),

    ("Matern 32", {"type": "Matern_32", "scaled": False, "ard": False}),
    ("Matern 32 ARD", {"type": "Matern_32", "scaled": False, "ard": True}),
    ("Scaled Matern 32", {"type": "Matern_32", "scaled": True, "ard": False}),
    ("Scaled Matern 32 ARD", {"type": "Matern_32", "scaled": True, "ard": True}),

    ("Matern 52", {"type": "Matern_52", "scaled": False, "ard": False}),
    ("Matern 52 ARD", {"type": "Matern_52", "scaled": False, "ard": True}),
    ("Scaled Matern 52", {"type": "Matern_52", "scaled": True, "ard": False}),
    ("Scaled Matern 52 ARD", {"type": "Matern_52", "scaled": True, "ard": True})
    ]

# COLLECT RESULTS ===========================================================================================

results = []
hyperparams = []
lengthscales_ard = []
keys = ["Time", "Na", "Cu", "Fe", "Mn", "Ni"] # must stay in this order

# Default white noise
for name, kernel_settings in all_kernel_settings:

    name = f"{name} Low Noise"

    # Set default white noise level which is 1E-04
    white_noise_threshold = 1E-04

    print(f"Evaluating kernel: {name}")

    # Run CV and get model metrics and values to plot
    cv_results, plot_y_actual, plot_y_pred = run_cv_gp_mod(X,
                                                       y,
                                                       groups,
                                                       scaled=kernel_settings["scaled"],
                                                       ard = kernel_settings["ard"],
                                                       covar_type=kernel_settings["type"],
                                                       noise=white_noise_threshold)

    # Fit with all data and calculate scalar MLL
    mll, gp, scalar_mll = final_fit_mod(X,
                                        y,
                                        scaled=kernel_settings["scaled"],
                                        ard = kernel_settings["ard"],
                                        covar_type=kernel_settings["type"],
                                        noise=white_noise_threshold)

    # Get kernel hyperparameters
    covar = gp.covar_module

    # account for ScaleKernel vs unscaled kernels
    if isinstance(covar, ScaleKernel):
        base_kernel = covar.base_kernel
        outputscale = covar.outputscale.item()
    else:
        base_kernel = covar
        outputscale = np.nan

    lengthscale = base_kernel.lengthscale.detach().cpu().numpy().ravel() # will be vector (ARD) or scalar (no ARD)

    # Update scores per kernel
    results.append({
        "kernel_name": name,
        "train_avg_r2": float(cv_results["train_avg_r2"]), # need to convert from tensors
        "train_avg_rmse": float(cv_results["train_avg_rmse"]),
        "test_avg_r2": float(cv_results["test_avg_r2"]),
        "test_avg_rmse": float(cv_results["test_avg_rmse"]),
        "avg_mll": float(cv_results["avg_mll"]),
        "scalar_mll": float(scalar_mll)
    })

    hyperparams.append({
    "kernel_name": name,
    "noise_constraint": white_noise_threshold,
    "outputscale": outputscale,
    "noise": gp.likelihood.noise.item(),
    "lengthscale": lengthscale.tolist(), # stores ARD vector or scalar BUT AS A STRING SO NOT USEFUL, hence new df below
    })

    if len(lengthscale) > 1:
        lengthscales_ard.append({"kernel_name": name, **dict(zip(keys, lengthscale))})

    plot_data = pd.DataFrame({
        "act_P2O3": np.array(plot_y_actual).ravel(),
        "pred_P2O3": np.array(plot_y_pred).ravel()
    })
    plot_data.to_csv(f"active_learning_model/surrogate_design/kernel_testing/kernel_plot_data/{name}_data.csv", index=False)
    print("Saved csv file of results.")

    print(gp)

# Constrained white noise
for name, kernel_settings in all_kernel_settings:
    
    name = f"{name} High Noise"
    
    # Increase white noise constraint
    white_noise_threshold = 0.001

    print(f"Testing kernel: {name}")

    # Run CV and get model metrics and values to plot
    cv_results, plot_y_actual, plot_y_pred = run_cv_gp_mod(X,
                                                       y,
                                                       groups,
                                                       scaled=kernel_settings["scaled"],
                                                       ard = kernel_settings["ard"],
                                                       covar_type=kernel_settings["type"],
                                                       noise=white_noise_threshold)

    # Fit with all data and calculate scalar MLL
    mll, gp, scalar_mll = final_fit_mod(X,
                                        y,
                                        scaled=kernel_settings["scaled"],
                                        ard = kernel_settings["ard"],
                                        covar_type=kernel_settings["type"],
                                        noise=white_noise_threshold)

    # Get kernel hyperparameters
    covar = gp.covar_module

    # account for ScaleKernel vs unscaled kernels
    if isinstance(covar, ScaleKernel):
        base_kernel = covar.base_kernel
        outputscale = covar.outputscale.item()
    else:
        base_kernel = covar
        outputscale = np.nan

    lengthscale = base_kernel.lengthscale.detach().cpu().numpy().ravel() # will be vector (ARD) or scalar (no ARD)

    # Update scores per kernel
    results.append({
        "kernel_name": name,
        "train_avg_r2": float(cv_results["train_avg_r2"]), # need to convert from tensors
        "train_avg_rmse": float(cv_results["train_avg_rmse"]),
        "test_avg_r2": float(cv_results["test_avg_r2"]),
        "test_avg_rmse": float(cv_results["test_avg_rmse"]),
        "avg_mll": float(cv_results["avg_mll"]),
        "scalar_mll": float(scalar_mll)
    })

    # Update hyperparameters per kernel
    hyperparams.append({
        "kernel_name": name,
        "noise_constraint": white_noise_threshold,
        "outputscale": outputscale,
        "noise": gp.likelihood.noise.item(),
        "lengthscale": lengthscale.tolist(), # stores ARD vector or scalar BUT AS A STRING SO NOT USEFUL, hence new df below
        })

    if len(lengthscale) > 1:
        lengthscales_ard.append({"kernel_name": name, **dict(zip(keys, lengthscale))})
    
    plot_data = pd.DataFrame({
        "act_P2O3": np.array(plot_y_actual).ravel(),
        "pred_P2O3": np.array(plot_y_pred).ravel()
    })
    plot_data.to_csv(f"active_learning_model/surrogate_design/kernel_testing/kernel_plot_data/{name}_data.csv", index=False)
    print("Saved csv file of results.")

    print(gp)


# Save scores
results = pd.DataFrame(results)
results.to_csv("active_learning_model/surrogate_design/kernel_testing/surrogate_test_results_v3.csv", index=False)
print("Saved csv file of results.")

hyperparams = pd.DataFrame(hyperparams)
hyperparams.to_csv("active_learning_model/surrogate_design/kernel_testing/kernel_hyperparameters_v3.csv", index=False)

lengthscales_ard = pd.DataFrame(lengthscales_ard)
lengthscales_ard.to_csv("active_learning_model/surrogate_design/kernel_testing/kernel_hyperparams_lengthscales_ard_v3.csv", index=False)







