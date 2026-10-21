# Finalised

# generates synthetic data for:
# 1. manually sampled dataset (369)
# 2. a sobol dataset (400)
# 3. a lhs dataset (400)
# 4. a random dataset (400)

# Simulates values for experimentally sampled compositions

import pandas as pd
import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples

from smt.sampling_methods import LHS
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import sys
sys.path.append("msc_project")
from ApprOxiMate.approx.approximate import ApprOXimate
from ApprOxiMate.approx.feature_engineering import MaterialFeatureExtractor
from mendeleev.fetch import fetch_table

# Make samples nicely formatted and rounded values for experimental feasibility
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

# GENERATE SYSTEMATICALLY SAMPLED DATA ======================================================

### LHS ----------------------------------
# Set bounds for each feature
# Feature order is: Time, Na, Cu, Fe, Mn (Ni computed later)
bounds = np.array([[3., 12.],
                   [0.5, 1.0],
                   [0., 1.],
                   [0., 1.],
                   [0., 1.],])

sampler = LHS(xlimits=bounds)

lhs_samples = pd.DataFrame()

while True:

    # Generate LHS samples
    X = torch.from_numpy(sampler(100)).float()

    # Keep only feasible compositions
    feasible = (X[:, 2] + X[:, 3] + X[:, 4] <= 1.0)

    X = X[feasible]

    # Add Ni column back in
    Ni = 1.0 - (X[:,2] + X[:,3] + X[:,4])
    X = torch.cat([X, Ni.unsqueeze(1)], dim=1)

    lhs_df = pd.DataFrame(
        X.numpy(),
        columns=["Time", "Na", "Cu", "Fe", "Mn", "Ni"]
    )

    lhs_df = make_experimental(lhs_df)

    # Add to previous samples
    lhs_samples = pd.concat([lhs_samples, lhs_df], ignore_index=True)

    # remove duplicates
    lhs_samples = lhs_samples.drop_duplicates()

    print(len(lhs_samples))

    if len(lhs_samples) >= 400:
        lhs_dataset = lhs_samples.iloc[:400].copy()
        break

# SOBOL ----------------------------------------------
# Reset bounds to be compatible with sobol sampler
bounds = torch.tensor([
    [3.,0.5,0.,0.,0.],
    [12.,1.,1.,1.,1.]
])

sobol_samples = pd.DataFrame()

while True:
    X = draw_sobol_samples(
        bounds=bounds,
        n=250,
        q=1
    ).squeeze(1)

    feasible = (X[:,2] + X[:,3] + X[:,4] <= 1.0)

    X = X[feasible]

    # Add Ni column
    Ni = 1.0 - (X[:,2] + X[:,3] + X[:,4])
    X = torch.cat([X, Ni.unsqueeze(1)], dim=1)

    sobol_df = pd.DataFrame(
        X.numpy(),
        columns=["Time", "Na", "Cu", "Fe", "Mn", "Ni"]
    )

    sobol_df = make_experimental(sobol_df)

    # Combine selected samples into a dataframe
    sobol_samples = pd.concat([sobol_samples, sobol_df], ignore_index=True)

    # Remove duplicates among newly generated samples too (only removes from same df)
    sobol_samples = sobol_samples.drop_duplicates()

    print(len(sobol_samples))

    if len(sobol_samples) >= 400:
        sobol_dataset = sobol_samples.iloc[:400].copy()
        break

# RANDOM --------------------------

# Set bounds for each feature
# Note feature order is: Time, Na, Cu, Fe, Mn (Ni is removed and will be computed later)
bounds = torch.tensor([
    [3.,0.5,0.,0.,0.],
    [12.,1.,1.,1.,1.]
])

rand_samples = pd.DataFrame()

while True:

    # Need a lot of samples to obtain feasible compositions, 200 --> 30 and 500 --> 90
    n = 1000

    # Random samples with shape (n, 6)
    X = torch.rand(n, 5)

    # Scale from [0, 1]^5 to feature ranges defined above
    X = bounds[0] + (bounds[1] - bounds[0]) * X

    # Add TM constraint
    feasible = (X[:,2] + X[:,3] + X[:,4] <= 1.0)
    X = X[feasible]

    # Add Ni column
    Ni = 1.0 - (X[:,2] + X[:,3] + X[:,4])
    X = torch.cat([X, Ni.unsqueeze(1)], dim=1)

    rand_df = pd.DataFrame(
        X.numpy(),
        columns=["Time", "Na", "Cu", "Fe", "Mn", "Ni"]
    )

    rand_df = make_experimental(rand_df)

    # Combine selected samples into a dataframe
    rand_samples = pd.concat([rand_samples, rand_df], ignore_index=True)

    # Remove duplicates among newly generated samples too (only removes from same df)
    rand_samples = rand_samples.drop_duplicates()

    print(len(rand_samples))

    if len(rand_samples) >= 400:
        rand_dataset = rand_samples.iloc[:400].copy()
        break

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

    # create "Chemical" column for groups
    columns_to_keep = ["Na", "Cu", "Fe", "Mn", "Ni"]
    element_ratios = candidates[columns_to_keep]

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

# UPDATE DATASETS WITH Cat_Success and P2O3 ratios =====================================

# Load manually sampled experimental dataset
experimental_df = pd.read_csv("active_learning_model/original_dataset/df_exp_semi_refined.csv")

cols_to_keep = ["Time", "Na", "Cu", "Fe", "Mn", "Ni"]
experimental_df = experimental_df[cols_to_keep]

# GENERATING DATA
experimental_df = update_candidates(experimental_df)
sobol_dataset = update_candidates(sobol_dataset)
lhs_dataset = update_candidates(lhs_dataset)
rand_dataset = update_candidates(rand_dataset)

# SAVE DATASETS
experimental_df.to_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_expr.csv",
    index=False
)
print("Saved expr dataset.")

sobol_dataset.to_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_sobol.csv",
                     index=False)
print("Saved sobol dataset.")

lhs_dataset.to_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_lhs.csv",
                   index=False)
print("Saved lhs dataset.")

rand_dataset.to_csv("active_learning_model/active_learning_model_synth/systematic_random_sampling_synth/synth_dataset_random.csv",
                   index=False)
print("Saved random dataset.")



