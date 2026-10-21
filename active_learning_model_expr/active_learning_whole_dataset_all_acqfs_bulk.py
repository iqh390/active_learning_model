# Finalised

# Kernel is interchangeable (Scaled ARD with default noise): RBF, Matern32, Matern52
# acqf is interchangeable: qNIPV, qUCB, qPSD
# Initial dataset is full experimental dataset

import time
import numpy as np
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

from botorch.sampling import SobolQMCNormalSampler

from botorch.acquisition import qNegIntegratedPosteriorVariance, qUpperConfidenceBound, qPosteriorStandardDeviation
from botorch.optim import optimize_acqf

# DATASET ======================================

def rearrage_dataset(df):
    """ Transforms dataframe into suitable features and target values (tensors) and groups (list).
    Dataframe columns should contain (Time,Na,Cu,Fe,Mn,Ni,P2O3ratio,Cat_Success).
    """
    df_success = df[df["Cat_Success"] == "Successful"]
    
    columns_to_keep = ["Time", "Na", "Cu", "Fe", "Mn", "Ni"]
    X = torch.tensor(df_success[columns_to_keep].to_numpy())
    
    y = torch.tensor(df_success["P2O3ratio"].to_numpy()).reshape(-1, 1)

    groups = df_success["Chemical"].values

    return X, y, groups

# INITIALISING GP MODEL ===================================================

# Kernel aka covar_module is exchangeable, all scaled ARD
def initialize_GPmodel(train_X, train_y, covar_type=None, noise=None): 
    """Initializes a GP model where

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

    # Train the GP model with the standardized inputs
    gp = SingleTaskGP(
        train_X, train_y,
        input_transform=InputStandardize(d=train_X_dim),
        outcome_transform=Standardize(m=train_y_dim),
        covar_module=covar_module,
        # Override likelihood if noise level is specified
        likelihood=likelihood
    )

    # Objective defined
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

    return mll, gp
    
# FUNCTION TO OBTAIN OVERALL MLL AND FINAL FITTED GP MODEL =======================

def final_fit(inputs, outputs, covar_type=None, noise=None):

    # Initialise gp model and train given data
    mll, gp = initialize_GPmodel(inputs, outputs, covar_type=covar_type, noise=noise)

    # Fit model to obtain MLL, this trains the hyperparams
    fit_gpytorch_mll(mll)

    # Calculate scalar mll
    with torch.no_grad():
        output_MVN = gp(inputs)
        scalar_mll = mll(output_MVN, outputs).sum()
    
    return mll, gp, scalar_mll.item()

# ACTIVE LEARNING TO RETURN CANDIDATE COMPOSITIONS ======================

# INITIALISE ACQUISITION FUNCTION ==============================================

# Initialise and specified acquisition functions and return candidate values
def initialize_and_optimize_acq_function(
        fitted_model,
        current_X,
        acqf,
        num_restarts: int=10, # These happen in parallel
        raw_samples: int=128, # Number of random candidate samples the acqf optimiser will start with
        candidates: int=10,
        ):
    
    # Define space within which candidates will be chosen
    dims = current_X.shape[-1]

    # Set boundary constraints to 0,1 for all compositional features and 3,12 for Time
    lower = torch.zeros(dims, dtype=current_X.dtype)
    upper = torch.ones(dims, dtype=current_X.dtype)    
    lower[0], upper[0] = 3.0, 12.0 # Preserve different boundaries for Time only
    bounds = torch.stack((lower, upper))
    
    # Define constraints for candidate compositions
    # Note feature order is: Time, Na, Cu, Fe, Mn, Ni
    equality_constraints = [(
        torch.tensor([2, 3, 4, 5]), # feature indices to add --> only TMs
        torch.ones(4, dtype=current_X.dtype), # coefficients are all 1.
        1.0 # RHS of equation: fractional compositions must add to 1
    )]
    
    inequality_constraints = [(
        torch.tensor([1]), # feature index for Na
        torch.tensor([1.], dtype=current_X.dtype), # coefficient of 1.
        0.5 # set Na >= 0.5
    )]

    if acqf == "qNIPV":
        # Generates MC samples from a GP posterior w/ Sobol quasi-random sequences
        # Dimension of sampler needs to match features hence torch.Size([dims])
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([dims]), seed=42)

        with torch.no_grad():
            # Rescaling outputs with fitted_model.posterior as we initially scaled inputs
            # Obtains a posterior object (defines probability distribution, not actual sampled values)
            pred_current_X = fitted_model.posterior(current_X)

        qmc_samples = qmc_sampler(pred_current_X)
        
        # construct aqcf to reduce integrated posterior variance
        print("Obtaining qNIPV candidates...")
        qNIPV = qNegIntegratedPosteriorVariance(model=fitted_model, mc_points=qmc_samples)
        # optimise the acqf and return the next samples
        next_samples, acq_value = optimize_acqf(
        qNIPV,
        bounds=bounds,
        q=candidates,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints)

    elif acqf == "qUCB":
        # Construct aqcf to reduce integrated posterior variance
        print("Obtaining qUCB candidates...")
        qUCB = qUpperConfidenceBound(model=fitted_model, beta=1000) # Can try larger
        # Optimise the acqf and return the next samples
        next_samples, acq_value = optimize_acqf(
        qUCB,
        bounds=bounds,
        q=candidates,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints)

    elif acqf == "qPSD":
        # Construct aqcf to reduce integrated posterior variance
        print("Obtaining qPSD candidates...")
        qPSD = qPosteriorStandardDeviation(model=fitted_model)
        # optimise the acqf and return the next samples
        next_samples, acq_value = optimize_acqf(
        qPSD,
        bounds=bounds,
        q=candidates,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints)
        
    else:
        print("Invalid acquisition function.")

    return next_samples, acq_value

# Format candidates nicely
def make_experimental(raw_candidates):
    """Takes raw_candidates as a dataframe and returns experimentally plausible rounded values.
    Note these are:
    Integer time
    Na to 1dp
    Cu, Fe, Mn, Ni to 4dp
    """
    reasonable_candidates = raw_candidates.copy()
    reasonable_candidates["Time"] = reasonable_candidates["Time"].round(decimals=0)
    reasonable_candidates["Na"] = reasonable_candidates["Na"].round(decimals=1)
    reasonable_candidates[["Cu", "Fe", "Mn", "Ni"]] = reasonable_candidates[["Cu", "Fe", "Mn", "Ni"]].round(decimals=4)
    return reasonable_candidates

# produce df of candidates, suitably formatted and rounded
def candidates_tensor_to_expr_pd(candidates):
    """Takes tensor of candidates from optimised acquisition function.
    Returns a nicely formatted dataframe with rounded values.
    """
    features = ["Time", "Na", "Cu", "Fe", "Mn", "Ni"]
    df_candidates = pd.DataFrame(candidates.detach().cpu().numpy(), columns=features)
    df_candidates = make_experimental(df_candidates)
    return df_candidates

# predict P2O3 value for candidates using GP model
def predict_P2O3_gp(candidates, fitted_model):

    predicted_candidates = candidates.copy()
    tensor_candidates = torch.tensor(candidates.copy().to_numpy())

    # Obtain P2O3 from posterior mean
    # Values are not clipped for this model
    p2o3_distr = fitted_model.posterior(tensor_candidates)
    p2o3_values = p2o3_distr.mean
    print(p2o3_values)

    predicted_candidates["pred_P2O3"] = p2o3_values.detach().cpu().numpy().round(decimals=2)

    return predicted_candidates

# FUNCTION TO UPDATE DATASET =========================================

# both df and updated_candidates should be dataframes
def update_dataset(df, updated_candidates):
    updated_dataset = pd.concat((df, updated_candidates), axis=0, ignore_index=True)
    return updated_dataset

# ==========================================================================
# ACTIVE LEARNING MODEL
# ==========================================================================

# 1. Initial dataset (dataframe) taken as input
# 2. Dataset is split into features/target as tensors
# 3. GP model is trained and evaluated
# 4. Acquisition function is applied to produce candidates (tensor)
# 5. Candidate values are predicted with the GP model (dataframe) and returned
# 6. NO LOOP


# SETTINGS ===
N_ITERATIONS = 1
BATCH_SIZE = 10
kernel = "Matern_32"

# DATASET ============================

# load file for inital full dataset
df = pd.read_csv("active_learning_model/original_dataset/df_exp_semi_refined.csv")

X, y , groups = rearrage_dataset(df)

# TRAIN AND FIT GP MODEL W/ ALL DATA ========================

# Iniitalise new gp model and fit and obtain scalar mll with RBF
mll_all, gp_all, scalar_mll = final_fit(X, y, kernel)

# OPTIMISE ACQUISITION FUNCTION + PRODUCE CANDIDATES =====================

acqfs = ["qNIPV",
         "qUCB",
         "qPSD"]


for i in range(len(acqfs)):
    acqf = acqfs[i]

    for run in range(3):
        # Reindex runs for ease
        run += 1

        # Initialise df to store candidates
        all_candidates = pd.DataFrame()
        
        # start timer
        start = time.perf_counter()

        print(f"Starting run {run} for {acqf}.")

        # Produce candidates for experiment/simulation
        candidates, acqf_value = initialize_and_optimize_acq_function(
            fitted_model=gp_all, current_X=X, acqf=acqf, candidates=BATCH_SIZE
            )

        print(type(acqf_value))
        print(acqf_value)

        print(f"Successfully optimised {acqf} acqf...")

        # format candidates for predictions
        df_candidates = candidates_tensor_to_expr_pd(candidates)

        # simluate values
        df_updated_candidates = predict_P2O3_gp(candidates=df_candidates, fitted_model=gp_all)

        # store results and add a column for batch number
        all_candidates = pd.concat([all_candidates, df_updated_candidates], axis=0, ignore_index=True)

        print("Candidates predicted from GP.")
        print(df_updated_candidates)

        # end timer
        end = time.perf_counter()
        minutes, seconds = divmod(end - start, 60)
        print(f"{acqf} Run {run} Time: {int(minutes)}m {seconds:2f}s")

        # SAVE CANDIDATES ====================================================
        
        all_candidates.to_csv(f"active_learning_model/active_learning_model_expr/{kernel}/{kernel}_whole_dataset_{acqf}_bulk_candidates_run{run}.csv",
                            index=False)