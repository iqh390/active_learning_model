# Finalised

# varyies batch size for qNIPV, using the 50 as initial dataset size
# 2, 5, 10, 20, 50, 100
# returns (1) a csv with compositions & (2) a csv with model scores to plot for each batch size

import time
import numpy as np
import pandas as pd

import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

import sklearn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import root_mean_squared_error

import sys
sys.path.append("msc_project")
from ApprOxiMate.approx.approximate import ApprOXimate
from ApprOxiMate.approx.feature_engineering import MaterialFeatureExtractor

from mendeleev.fetch import fetch_table

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
from botorch.models.transforms.input import InputStandardize

from botorch.sampling import SobolQMCNormalSampler

from botorch.acquisition import qNegIntegratedPosteriorVariance, qUpperConfidenceBound, qPosteriorStandardDeviation
from botorch.optim import optimize_acqf

# DATASET ======================================

# Function which turns a dataframe into suitable features and target values (tensors) and groups (list)
# Dataframe columns should contain (Time,Na,Cu,Fe,Mn,Ni,P2O3ratio,Cat_Success) at least
def rearrage_dataset(df):

    df_success = df[df["Cat_Success"] == "Successful"]
    
    columns_to_keep = ["Time", "Na", "Cu", "Fe", "Mn", "Ni"]
    X = torch.tensor(df_success[columns_to_keep].to_numpy())
    
    y = torch.tensor(df_success["P2O3ratio"].to_numpy()).reshape(-1, 1)

    groups = df_success["Chemical"].values

    return X, y, groups

# TRAINING CAT_SUCCESS AND P2O3RATIO MODELS (these are static) ========================

# featurised datasets to train
cl_train_ft = pd.read_csv("active_learning_model/original_dataset/cat_success_model/cl_app_train_split.csv")
cl_val_ft = pd.read_csv("active_learning_model/original_dataset/cat_success_model/cl_app_val_split.csv")
cl_test_ft = pd.read_csv("active_learning_model/original_dataset/cat_success_model/cl_app_test_split.csv")

cl_frames = [cl_train_ft, cl_val_ft, cl_test_ft]
cl_df = pd.concat(cl_frames)
cl_df["Cat_Success"] = cl_df["Cat_Success"].astype("category") # set successful to 0, unsuccessful to 1
categories = cl_df["Cat_Success"].cat.categories
cl_df["Cat_Success"] = cl_df["Cat_Success"].cat.set_categories(categories)
cl_df["Cat_Success"] = cl_df["Cat_Success"].cat.codes

feature_cols_cl = [col for col in cl_df.columns if col not in ['UID','Chemical','P2O3ratio', 'formula', 'Cat_Success']]
X_cl = cl_df[feature_cols_cl]
y_cl = cl_df['Cat_Success']

rg_train_ft = pd.read_csv("active_learning_model/original_dataset/p2o3ratio_model/rg_app_train_split.csv")
rg_val_ft = pd.read_csv("active_learning_model/original_dataset/p2o3ratio_model/rg_app_val_split.csv")
rg_test_ft = pd.read_csv("active_learning_model/original_dataset/p2o3ratio_model/rg_app_test_split.csv")

rg_frames = [rg_train_ft, rg_val_ft, rg_test_ft]
rg_df = pd.concat(rg_frames)

feature_cols_rg = [col for col in rg_df.columns if col not in ['UID','Chemical','P2O3ratio', 'formula', 'Cat_Success']]
X_rg = rg_df[feature_cols_rg]
y_rg = rg_df['P2O3ratio']

# FITTING CLASSIFICATION AND REGRESSION RF MODELS =====================================

cat_success_model_params = ({'bootstrap': True,
                             'ccp_alpha': 0.0,
                             'class_weight': None,
                             'criterion': 'gini',
                             'max_depth': None,
                             'max_features': 'sqrt',
                             'max_leaf_nodes': None,
                             'max_samples': None,
                             'min_impurity_decrease': 0.0,
                             'min_samples_leaf': 1,
                             'min_samples_split': 2,
                             'min_weight_fraction_leaf': 0.0,
                             'monotonic_cst': None,
                             'n_estimators': 100,
                             'n_jobs': None, 
                             'oob_score': False,
                             'random_state': 20,
                             'verbose': 0,
                             'warm_start': False})

cat_success_model = RandomForestClassifier(**cat_success_model_params)
print("CatSuccess_model")
print(cat_success_model)
cat_success_model.fit(X_cl, y_cl)

# Scale and normalize data before fitting (RF regression)
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
scaler = StandardScaler()
scaled_X_rg = normalize(scaler.fit_transform(X_rg))

p2o3ratio_model_params = ({'random_state': 0})
p2o3ratio_model = RandomForestRegressor(**p2o3ratio_model_params)
print("P2O3ratio_model")
print(p2o3ratio_model)
p2o3ratio_model.fit(scaled_X_rg, y_rg)

# FUNCTION TO FEATURISE CANDIDATE COMPOSITIONS w/ APPROXIMATE + UPDATE FROM SKLEAN MODELS ================================

def featurize_df(df, extractor, formula_col="Chemical"):
    formulas = df[formula_col].tolist()
    feature_rows = extractor.featurize_many(formulas)
    feat_df = pd.DataFrame(feature_rows, index=df.index)

    # Keep only rows that featurized successfully
    feature_cols = feat_df.drop(columns=["formula"], errors="ignore")
    if feature_cols.shape[1] == 0:
        return df.iloc[0:0].copy()

    valid_rows = feature_cols.notna().any(axis=1)
    df_valid = df.loc[valid_rows]
    feat_valid = feat_df.loc[valid_rows]

    failed_formulas = df.loc[~valid_rows, formula_col]
    for formula in failed_formulas:
        print("Formula Fail:", formula)

    return df_valid.join(feat_valid)

def feat_candidates(candidates):
    """Takes processed candidates as a dataframe
    Returns featurised version to ready to input into the RF models."""

    # create "Chemical" column for featurisation
    element_ratios = candidates.drop(columns=(['Time']))

    # also add "O2" at the end to complete formula
    candidates["Chemical"] = element_ratios.apply(lambda row: "".join(
                f"{col}{val:.6g}" for col, val in row.items() if pd.notna(val)
            ) + "O2",axis=1)
    
    # initialize modules
    approx = ApprOXimate()
    ptable = fetch_table("elements")
    extractor = MaterialFeatureExtractor(approx, ptable, mode="both", ionic_radius_unit="pm",
                                         enable_tm_potential=True, tm_anion="O", tm_cation='Na') # Na cathode specific features
    
    # run featurization
    candidates_feat = featurize_df(candidates, extractor, formula_col="Chemical")

    # only keep features in the previously featurised dataset for classification
    cl_features_to_keep = X_cl.columns
    cl_valid_features = [col for col in cl_features_to_keep if col in candidates_feat.columns]
    candidates_feat_cl = candidates_feat[cl_valid_features]

    # reordering so that candidate features are the same order as during fit
    candidates_feat_cl = candidates_feat_cl[X_cl.columns]

    # keeping features in previously featurised dataset for regression
    rg_features_to_keep = X_cl.columns
    rg_valid_features = [col for col in rg_features_to_keep if col in candidates_feat.columns]
    candidates_feat_rg = candidates_feat[rg_valid_features]

    # reordering to match original features during fit
    candidates_feat_rg = candidates_feat_rg[X_rg.columns]

    return candidates_feat_cl, candidates_feat_rg

def update_candidates(candidates):

    updated_candidates = candidates.copy()

    # featurise candidates for sklearn model compatibility
    candidates_feat_cl, candidates_feat_rg = feat_candidates(candidates)

    label_mapping = {0: "Successful", 1: "Unsuccessful"}

    cat_success = cat_success_model.predict(candidates_feat_cl)
    updated_candidates["Cat_Success"] = cat_success
    updated_candidates["Cat_Success"] = updated_candidates["Cat_Success"].map(label_mapping)

    successful_candidates = candidates_feat_rg[cat_success == 0] # only keep successful from rg featurised dataframe

    successful_candidates = normalize(scaler.transform(successful_candidates))
    p2o3_values = p2o3ratio_model.predict(successful_candidates)

    updated_candidates["P2O3ratio"] = -1.0
    successful_rows = (cat_success == 0)
    updated_candidates.loc[successful_rows, "P2O3ratio"] = p2o3_values.round(decimals=2)

    # create "Chemical" column for groups
    columns_to_keep = ["Na", "Cu", "Fe", "Mn", "Ni"]
    element_ratios = updated_candidates[columns_to_keep]

    updated_candidates["Chemical"] = element_ratios.apply(lambda row: "".join(
                f"{col}{val:.6g}" for col, val in row.items() if pd.notna(val)
            ) + "O2", axis=1)
    
    return updated_candidates

# INITIALISING GP MODEL ===================================================

# Kernel aka covar_module is exchangeable, all Scaled ARD
def initialize_GPmodel(train_X, train_y, covar_type=None, noise=None): 
    """Initialize a GP model where
    trian_X.shape is [N, d]
    train_y.shape is [N, 1]
    Return the fitted GP model with selected kernel and noise level.
    """

    train_X_dim = train_X.shape[-1]
    train_y_dim = train_y.shape[-1]

    ard_num_dims=train_X.shape[-1]

    # Make fresh kernel dependent on type
    if covar_type == "RBF":
        covar_module = ScaleKernel(RBFKernel(ard_num_dims=ard_num_dims))
    elif covar_type == "Matern52":
        covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=ard_num_dims))
    elif covar_type == "Matern32":
        covar_module = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=ard_num_dims))
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

# ===================== A FEW EXTRAS TO HELP ================

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

# another function for RMSE, tensor compatibility -------------------
def rmse(y_test: torch.Tensor,
       y_pred: torch.Tensor
       ):
    rmse = torch.sqrt(torch.mean((y_test - y_pred) ** 2))
    return rmse

# Split into grouped folds
def sklearn_gkf_split(X, y, groups, n_splits, random_state=42):
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()

    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return gkf.split(X, y, groups)

# function to calculate the proportion of P2, O3 and mixed values in each fold
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
            pred_X_test = gp_fold.posterior(X_test) # returns the posterior distribution sampled from X_test
            
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

            total_y_actual.extend(y_test.tolist())
            total_y_pred.extend(y_pred.tolist())

            # obtain scalar mll per fold
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

    # Obtain scalar mll
    with torch.no_grad():
        output_MVN = gp(inputs)
        scalar_mll = mll(output_MVN, outputs).sum()
    
    return mll, gp, scalar_mll.item()

# ACTIVE LEARNING TO RETURN CANDIDATE COMPOSITIONS ======================

# INITIALISE ACQUISITION FUNCTION ----------------------------------------

# Initialise and optimise all acquisition functions and return candidate values for each one
def initialize_and_optimize_acq_function(
        fitted_model,
        current_X,
        acqf,
        num_restarts: int=10, # these happen in parallel
        raw_samples: int=128,
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
        torch.ones(4, dtype=current_X.dtype), # coefficients are all 1. in this case
        1.0 # RHS of equation: fractional compositions must add to 1
    )]
    
    inequality_constraints = [(
        torch.tensor([1]), # feature index for Na
        torch.tensor([1.], dtype=current_X.dtype), # coefficient of 1.
        0.5 # set Na >= 0.5
    )]

    if acqf == "qNIPV":
        qmc_sampler = SobolQMCNormalSampler(sample_shape=torch.Size([dims]), seed=42)

        with torch.no_grad():
            pred_current_X = fitted_model.posterior(current_X)

        # Obtains samples from the GP posterior, using the MC generated random sequences
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
        # construct aqcf to reduce integrated posterior variance
        print("Obtaining qUCB candidates...")
        qUCB = qUpperConfidenceBound(model=fitted_model, beta=1000) # can try larger
        # optimise the acqf and return the next samples
        next_samples, acq_value = optimize_acqf(
        qUCB,
        bounds=bounds,
        q=candidates,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        equality_constraints=equality_constraints,
        inequality_constraints=inequality_constraints)

    elif acqf == "qPSD":
        # construct aqcf to reduce integrated posterior variance
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

# FUNCTION TO UPDATE DATASET =========================================

# both df and updated_candidates should be dataframes
def update_dataset(df, updated_candidates):
    updated_dataset = pd.concat((df, updated_candidates), axis=0, ignore_index=True)
    return updated_dataset


# ==================================
# BATCH SIZE TESTING
# ==================================


# Settings for initial dataset sampling type and size
SAMPLING_TYPE = "Sobol"
INITIAL_DATASET = 100

# Retrieve initial dataset
initial_df = pd.read_csv(f"active_learning_model/initial_dataset_size/{SAMPLING_TYPE}/{SAMPLING_TYPE}_initial_dataset_{INITIAL_DATASET}.csv")

# Batch sizes to test
batch_sizes = [
    2, 
    5, 
    10, 
    20,
    50
    ]

# SETTINGS
DATASET_LIMIT = 301
COVAR_TYPE = "RBF"
ACQF = "qNIPV"

for batch_size in batch_sizes:

    # Specific batch size
    BATCH_SIZE = batch_size
    # number of iterations
    ITERS = (DATASET_LIMIT - INITIAL_DATASET) // BATCH_SIZE


    # RUN AL MODEL =========================
    gp_r2 = []
    gp_rmse = []
    gp_mll = []

    all_candidates = pd.DataFrame()

    # Reset data to initial dataset
    df = initial_df

    print(f"Evaluating AL model for batch size {BATCH_SIZE}")
    # start timer
    start = time.perf_counter()

    for i in range(ITERS):
        print(f"{ACQF} Batch {i}")

        X, y , groups = rearrage_dataset(df)

        # TRAIN AND REFIT GP MODEL W/ ALL DATA ========================

        # evaluate gp with cv for r2
        gp_results, plot_y_actual, plot_y_pred = run_cv_gp(X, y, groups, covar_type=COVAR_TYPE)
        gp_r2.append(gp_results["test_avg_r2"])
        gp_rmse.append(gp_results["test_avg_rmse"])

        # iniitalise new gp model and fit and obtain scalar mll
        mll_all, gp_all, scalar_mll = final_fit(X, y, covar_type=COVAR_TYPE)
        gp_mll.append(scalar_mll)

        # OPTIMISE ACQUISITION FUNCTION + PRODUCE CANDIDATES =====================

        # Produce candidates for experiment/simulation
        candidates, _ = initialize_and_optimize_acq_function(
            fitted_model=gp_all, current_X=X, acqf=ACQF, candidates=BATCH_SIZE
            )
        print("Successfully optimised acqf...")

        # format candidates for predictions
        df_candidates = candidates_tensor_to_expr_pd(candidates)

        # simluate values
        df_updated_candidates = update_candidates(df_candidates)

        # store results and add a column for batch number
        all_candidates = pd.concat([all_candidates, df_updated_candidates.assign(batch=i)], axis=0, ignore_index=True)

        print("UPDATED CANDIDATES")
        print(df_updated_candidates)

        # update dataset
        df = update_dataset(df, df_updated_candidates)

        print(f"FINISHED ITERATION {i}")

    # end timer
    end = time.perf_counter()
    minutes, seconds = divmod(end - start, 60)
    print(f"Run time: {int(minutes)}m {seconds:2f}s")

    # calculating data for number of datapoints
    max_val = INITIAL_DATASET + (BATCH_SIZE * ITERS)
    no_samples = list(range(INITIAL_DATASET, max_val, BATCH_SIZE))

    # SAVE RESULTS --------------------------------------------

    results = pd.DataFrame({"no_samples": no_samples,
                        "test_avg_r2": gp_r2,
                        "test_avg_rmse": gp_rmse,
                        "scalar_mll": gp_mll,
                        })

    results.to_csv(f"active_learning_model/batch_size_tests/Data{INITIAL_DATASET}/batch_size_{BATCH_SIZE}_scores.csv", index=False)

    # SAVE CANDIDATES -------------------------------------------
    all_candidates.to_csv(f"active_learning_model/batch_size_tests/Data{INITIAL_DATASET}/batch_size_{BATCH_SIZE}_candidates.csv.csv", index=False)

