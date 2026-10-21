# Finalised

# tests synthetic datasets, systematic + random sampling
# trains GP every 10 datapoints and returns metrics

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# DATASET FUNCTIONS ===========================

# turns a dataframe into suitable features and target values (tensors) and groups (list)
# Dataframe columns should contain (Time,Na,Cu,Fe,Mn,Ni,P2O3ratio,Cat_Success) at least
def rearrage_dataset(df):

    # only take successful datapoints
    df_success = df[df["Cat_Success"] == "Successful"]
    
    columns_to_keep = ["Time", "Na", "Cu", "Fe", "Mn", "Ni"]
    X = torch.tensor(df_success[columns_to_keep].to_numpy())
    
    y = torch.tensor(df_success["P2O3ratio"].to_numpy()).reshape(-1, 1)

    groups = df_success["Chemical"].values

    return X, y, groups

# SPLIT DATASET =======================

# function that splits the inital ordered data
def split_per_batch(
    df: pd.DataFrame,
    batch_number: int,
    initial_data_size: int=135,
    batch_size: int=20
) -> pd.DataFrame: # should return a df of reordered data

    # returning df with batch_number of batches attached
    current_data = df.iloc[:(initial_data_size + batch_size * batch_number)]

    return current_data

# GP MODEL =============================

# INITIALISING GP MODEL ===================================================

# Kernel aka covar_module is exchangeable, all Scaled ARD
def initialize_GPmodel(train_X, train_y, covar_type=None, noise=None): 
    """Initialize a GP model where
    trian_X.shape: [N, d]
    train_y.shape: [N, 1]
    Return the fitted GP model with selected kernel and noise level.
    """
    # Get the dimensions of train_X and train_y to standardize inputs
    train_X_dim = train_X.shape[-1]
    train_y_dim = train_y.shape[-1]

    # Number of ARD lengthscales is the number of dimensions for each input
    ard_num_dims=train_X.shape[-1]

    if covar_type == "RBF":
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))
    elif covar_type == "Matern_52":
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=ard_num_dims))
    elif covar_type == "Matern_32":
        covar_module = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=ard_num_dims))
    else:
        print("Invalid kernel.")

    # Customise noise levels
    if noise == None:
        likelihood = None
    else:
        likelihood = GaussianLikelihood(noise_constraint=GreaterThan(noise))

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

# ===================== A FEW EXTRAS TO HELP ====================

# compute R^2 between predicted and test values, returns R^2 as scalar tensor ----------
def r2(y_test: torch.Tensor,
       y_pred: torch.Tensor
       ):

    ss_res = torch.sum((y_test - y_pred) ** 2)
    ss_tot = torch.sum((y_test - torch.mean(y_test)) ** 2)

    # avoid division by zero, just in case although probs don't need this
    if ss_tot == 0:
        return torch.tensor(float('nan'))

    r2 = 1 - ss_res / ss_tot
    return r2

# another function for RMSE, tensor compatibility -------------------
def rmse(y_test: torch.Tensor,
       y_pred: torch.Tensor
       ):
    rmse = torch.sqrt(torch.mean((y_test - y_pred) ** 2))
    return rmse

# Obtain groups (tensor compatibility)
def sklearn_gkf_split(X, y, groups, n_splits, random_state=42):
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return gkf.split(X, y, groups)

# Calculate the proportion of P2, O3 and mixed values in each fold
def fold_profile(y_train, y_test):
    if torch.is_tensor(y_train):
        O3_train = (y_train == 0).float().mean().item()
        mixed_train = ((y_train > 0) & (y_train < 1)).float().mean().item()
        P2_train = (y_train == 1).float().mean().item()

        O3_test = (y_test == 0).float().mean().item()
        mixed_test = ((y_test > 0) & (y_test < 1)).float().mean().item()
        P2_test = (y_test == 1).float().mean().item()
    else:
        O3_train = (y_train == 0).mean().item()
        mixed_train = ((y_train > 0) & (y_train < 1)).mean().item()
        P2_train = (y_train == 1).mean().item()

        O3_test = (y_test == 0).mean().item()
        mixed_test = ((y_test > 0) & (y_test < 1)).mean().item()
        P2_test = (y_test == 1).mean().item()
    return O3_train, mixed_train, P2_train, O3_test, mixed_test, P2_test

# CV TO EVALUATE GP ======================================

def run_cv_gp(
        X: torch.Tensor,
        y: torch.Tensor,
        groups: torch.Tensor,
        covar_type=None,
        random_state: int=42,
        noise=None,
        n_splits: int=5
):
    print("Running cross validation for GP...")

    train_r2_scores = []
    train_rmse_scores = []
    test_r2_scores = []
    test_rmse_scores = []
    mll_values = []
    train_prop = []
    test_prop = []
    total_y_actual = []
    total_y_pred = []

    for fold, (train_idx, test_idx) in enumerate(sklearn_gkf_split(X, y, groups, n_splits, random_state=random_state,)):

        print(f"GP Fold {fold}...")

        X_train, X_test = X[train_idx], X[test_idx] # already tensors
        y_train, y_test = y[train_idx], y[test_idx]

        O3_train, mixed_train, P2_train, O3_test, mixed_test, P2_test = fold_profile(y_train, y_test)
        train_prop.append([O3_train, mixed_train, P2_train])
        test_prop.append([O3_test, mixed_test, P2_test])

        # Train the model based on fold training data
        mll_fold, gp_fold = initialize_GPmodel(X_train, y_train, covar_type=covar_type, noise=noise)

        # Fit GP model
        fit_gpytorch_mll(mll_fold)

        # Evaluate model scores
        with torch.no_grad():
            pred_X_test = gp_fold.posterior(X_test) # returns the posterior distribution, sampled from X_test
            
            ### Test error
            y_pred_unclipped = pred_X_test.mean

            # Clip y_pred by changing any predicted P2O3 values above or below (0,1) to 0 or 1 (physically possible values)
            y_pred = torch.clamp(y_pred_unclipped, min=0.0, max=1.0)

            # calculate test R^2 and RMSE per fold and updates list of scores
            test_r2_scores.append(r2(y_test, y_pred).item())
            test_rmse_scores.append(rmse(y_test, y_pred).item())

            ### Train error
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
        "fold_1_train_O3_mixed_P2_prop": train_prop[0],
        "fold_2_train_O3_mixed_P2_prop": train_prop[1],
        "fold_3_train_O3_mixed_P2_prop": train_prop[2],
        "fold_4_train_O3_mixed_P2_prop": train_prop[3],
        "fold_5_train_O3_mixed_P2_prop": train_prop[4],
        "train_avg_r2": np.mean(train_r2_scores),
        "train_avg_rmse": np.mean(train_rmse_scores),
        "all_test_r2": test_r2_scores,
        "all_test_rmse": test_rmse_scores,
        "test_avg_r2": np.mean(test_r2_scores),
        "test_avg_rmse": np.mean(test_rmse_scores),
        "avg_mll": np.mean(mll_values),
        "fold_1_test_O3_mixed_P2_prop": test_prop[0],
        "fold_2_test_O3_mixed_P2_prop": test_prop[1],
        "fold_3_test_O3_mixed_P2_prop": test_prop[2],
        "fold_4_test_O3_mixed_P2_prop": test_prop[3],
        "fold_5_test_O3_mixed_P2_prop": test_prop[4]
    })

    return gp_results, total_y_actual, total_y_pred
    
# FUNCTION TO OBTAIN OVERALL MLL AND FINAL FITTED GP MODEL =======================

def final_fit(inputs, outputs, covar_type=None, noise=None):

    # Initialise gp model and train given data
    mll, gp = initialize_GPmodel(inputs, outputs, covar_type=covar_type, noise=noise)

    # Fit model
    fit_gpytorch_mll(mll)

    # Evaluate scalar mll
    with torch.no_grad():
        output_MVN = gp(inputs)
        scalar_mll = mll(output_MVN, outputs).sum()
    
    return mll, gp, scalar_mll.item()


# CALCULATING R^2 AND LML FOR EACH DATASET SIZE ==============================

def get_batch_scores(df, covar_type=None):
    print("Getting batch scores...")

    results = []

    for batch_number in range(NO_BATCHES):
        print(f"Batch number: {batch_number}")
    
        current_df = split_per_batch(df=df,
                                    batch_number=batch_number,
                                    initial_data_size=INITIAL_SIZE,
                                    batch_size=BATCH_SIZE,
                                    )

        X, y, groups = rearrage_dataset(current_df)

        # Calculate scores
        gp_results, _, _ = run_cv_gp(X, y, groups, covar_type=covar_type)

        # Refit GP with all data for scalar MLL value
        _, _, scalar_mll = final_fit(X, y, covar_type=covar_type)

        results.append({
            "no_samples": len(current_df),
            "batch_number": batch_number,
            "test_avg_r2": gp_results["test_avg_r2"],
            "test_avg_rmse": gp_results["test_avg_rmse"],
            "scalar_mll": scalar_mll
        })
    
    return pd.DataFrame(results)


# EXPERIMENTAL AND SIMULATED DATASETS FOR SYSTEMATIC SAMPLING ============

# Experimental
experimental_df = pd.read_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_expr.csv")

# SOBOL
sobol_df = pd.read_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_sobol2.csv")

# LHS
lhs_df = pd.read_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_lhs.csv")

# Ranodm
rand_df = pd.read_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_random.csv")

### RESULTS =================================================

COVAR_TYPE = "RBF"
BATCH_SIZE = 10
INITIAL_SIZE = 50
NO_BATCHES = (400 - INITIAL_SIZE) // BATCH_SIZE

print("Evaluating LHS data.")
lhs_results_df = (get_batch_scores(lhs_df, covar_type=COVAR_TYPE))
lhs_results_df.to_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_scores_lhs.csv", index=False)
print("Saved LHS results.")

print("Evaluating experimental data.")
exp_results_df = (get_batch_scores(experimental_df, covar_type=COVAR_TYPE))
exp_results_df.to_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_scores_exp.csv", index=False)
print("Saved experimental results.")

print("Evaluating SOBOL data.")
sobol_results_df = (get_batch_scores(sobol_df, covar_type=COVAR_TYPE))
sobol_results_df.to_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_scores_sobol2.csv", index=False)
print("Saved Sobol results.")

print("Evaluating random data.")
rand_results_df = (get_batch_scores(rand_df, covar_type=COVAR_TYPE))
rand_results_df.to_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_scores_random.csv", index=False)
print("Saved random results.")