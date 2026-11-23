import json
import os

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# Author's code from original repository: multimodal_emotion_recogntion.ipynb
 

def discretize_labels(targets_df):
    """
    Apply optimal binning to emotion labels.
    
    Args:
        targets_df (DataFrame): DataFrame with continuous emotion labels.
        
    Returns:
        DataFrame: DataFrame with discretized emotion labels.
    """
    bin_thresholds = {
        "preceived_arousal": (4.8, 6.0),
        "preceived_valance": (3.8, 5.5),
        "felt_arousal": (4.6, 6.0),
        "felt_valance": (4.3, 5.2)
    }

    # Discretize each target column based on threshold ranges
    for col, (low_thr, high_thr) in bin_thresholds.items():
        targets_df[col] = targets_df[col].apply(lambda x: 0 if x <= low_thr else (1 if x <= high_thr else 2))

    return targets_df

def train_autogluon_models(features_df, targets_df):
    """
    Train AutoGluon models for each target emotion and compute performance metrics.
    
    Args:
        features_df (DataFrame): Input features.
        targets_df (DataFrame): Target labels.
        
    Returns:
        tuple: (Dictionary of models, cross-validation results, leaderboards)
    """
    models, cv_results, leaderboards = {}, {}, {}

    # Iterate through each target column and train a model
    for target_col in targets_df.columns:
        print(f"\nTraining AutoGluon model for {target_col}...")

        y = targets_df[target_col]
        # Split the data into training and testing sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            features_df, y, test_size=0.2, stratify=y, random_state=123
        )

        # Combine training features with the target
        df_combined_train = X_train.copy()
        df_combined_train[target_col] = y_train

        # Train the AutoGluon predictor with best quality presets, excluding slow KNN model
        predictor = TabularPredictor(label=target_col, eval_metric="f1_macro").fit(
            df_combined_train,
            presets="best_quality",
            excluded_model_types=['KNN'],
            ag_args_fit={"num_cpus": os.cpu_count()}
        )

        # Retrieve the leaderboard and store the best model name
        leaderboard = predictor.leaderboard(silent=True)
        best_model_name = leaderboard.iloc[0]["model"]
        leaderboards[target_col] = best_model_name

        # Predict on the test set and compute F1 and accuracy metrics
        y_pred = predictor.predict(X_test)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        accuracy = accuracy_score(y_test, y_pred)

        cv_results[target_col] = {
            "F1 Low": f"{f1_per_class[0]:.4f}",
            "F1 Medium": f"{f1_per_class[1]:.4f}",
            "F1 High": f"{f1_per_class[2]:.4f}",
            "F1 Macro": f"{f1_macro:.4f}",
            "Accuracy": f"{accuracy:.4f}"
        }

        models[target_col] = predictor

    return models, cv_results, leaderboards

MODEL_NAME_MAPPING = {
    "WeightedEnsemble_L2": "ENS_WEIGHTED",
    "ExtraTreesEntr": "XT",
    "ExtraTreesGini": "XT",
    "LightGBM": "GBM",
    "LightGBMXT": "GBM",
    "NeuralNetTorch": "NN_TORCH",
    "NeuralNetFastAI": "FASTAI",
    "XGBoost": "XGB",
    "CatBoost": "CAT",
    "RandomForest": "RF",
    "ExtraTrees": "XT",
    "KNeighbors": "KNN",
    "LogisticRegression": "LR",
    "Transformer": "TRANSF",
    "AG_TEXT_NN": "AG_TEXT_NN",
    "AG_IMAGE_NN": "AG_IMAGE_NN",
    "AG_AUTOMM": "AG_AUTOMM",
    "FT_TRANSFORMER": "FT_TRANSFORMER",
    "TABPFN": "TABPFN",
    "TABPFNMIX": "TABPFNMIX",
    "FASTTEXT": "FASTTEXT",
    "ENS_WEIGHTED": "ENS_WEIGHTED",
    "SIMPLE_ENS_WEIGHTED": "SIMPLE_ENS_WEIGHTED",
    "IM_RULEFIT": "IM_RULEFIT",
    "IM_GREEDYTREE": "IM_GREEDYTREE",
    "IM_FIGS": "IM_FIGS",
    "IM_HSTREE": "IM_HSTREE",
    "IM_BOOSTEDRULES": "IM_BOOSTEDRULES",
    "VW": "VW",
    "DUMMY": "DUMMY"
}

def get_best_models(predictors):
    """
    Extract the best (non-ensemble) model for each emotion target from AutoGluon's leaderboard.
    
    Args:
        predictors (dict): Dictionary of trained AutoGluon predictors.
        
    Returns:
        dict: Mapping from target column to best model name.
    """
    best_models = {}
    for target_col, predictor in predictors.items():
        leaderboard = predictor.leaderboard(silent=True)
        # Filter out ensemble models and select the best non-ensemble model
        non_ensemble_models = leaderboard[~leaderboard["model"].str.contains("WeightedEnsemble")].copy()
        best_model_name = non_ensemble_models.iloc[0]["model"]
        best_models[target_col] = MODEL_NAME_MAPPING.get(best_model_name, best_model_name)
    return best_models

def retrain_best_model_kfold(features_df, targets_df, models, n_splits=5):
    """
    Retrain the best AutoGluon models using K-Fold Cross-Validation.
    
    Args:
        features_df (DataFrame): Input features.
        targets_df (DataFrame): Target labels.
        models (dict): Dictionary of trained models.
        n_splits (int): Number of cross-validation splits.
        
    Returns:
        dict: Cross-validation results for each target.
    """
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    cv_results = {}
    best_models = get_best_models(models)
    
    for target_col, best_model_name in best_models.items():
        print(f"\nPerforming {n_splits}-Fold Cross-Validation for {target_col} using {best_model_name}...")
        y = targets_df[target_col]
        f1_scores_per_fold = {"High": [], "Medium": [], "Low": [], "Macro Avg": []}
        accuracy_scores = []

        # Loop through each fold in cross-validation
        for train_idx, val_idx in kfold.split(features_df, y):
            X_train_fold, X_val_fold = features_df.iloc[train_idx], features_df.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Combine training features with target
            df_combined_train = X_train_fold.copy()
            df_combined_train[target_col] = y_train_fold

            # Train predictor using only the best model's hyperparameters
            predictor = TabularPredictor(label=target_col).fit(
                df_combined_train,
                hyperparameters={best_model_name: {}},
                keep_only_best=True,
                num_bag_folds=0,  # Disable bagging
                num_stack_levels=0,  # Disable stacking
                presets="good_quality",
                ag_args_fit={"num_cpus": os.cpu_count()},
                verbosity=0
            )

            # Predict on the validation fold and calculate performance metrics
            y_val_preds = predictor.predict(X_val_fold)
            f1_per_class = f1_score(y_val_fold, y_val_preds, average=None)
            macro_f1 = f1_score(y_val_fold, y_val_preds, average="macro")
            accuracy = accuracy_score(y_val_fold, y_val_preds)

            f1_scores_per_fold["High"].append(f1_per_class[2])
            f1_scores_per_fold["Medium"].append(f1_per_class[1])
            f1_scores_per_fold["Low"].append(f1_per_class[0])
            f1_scores_per_fold["Macro Avg"].append(macro_f1)
            accuracy_scores.append(accuracy)

        # Store averaged CV metrics for this target
        cv_results[target_col] = {
            "Model": best_model_name,
            "High": f"{np.mean(f1_scores_per_fold['High']):.4f} ± {np.std(f1_scores_per_fold['High']):.4f}",
            "Medium": f"{np.mean(f1_scores_per_fold['Medium']):.4f} ± {np.std(f1_scores_per_fold['Medium']):.4f}",
            "Low": f"{np.mean(f1_scores_per_fold['Low']):.4f} ± {np.std(f1_scores_per_fold['Low']):.4f}",
            "Macro Avg": f"{np.mean(f1_scores_per_fold['Macro Avg']):.4f} ± {np.std(f1_scores_per_fold['Macro Avg']):.4f}",
            "Accuracy": f"{np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}"
        }
    return cv_results


####################################################################################################################################
# Our additional code for training and evaluating models on the AU features dataset.
# The csv comes from reproduction/prepare_features.py, and the results are saved to /results/au_results.json.

CSV_PATH = "reproduction/results/au_features_complete_AU.csv"
OUT_DIR = "reproduction/results"
OUT_JSON = os.path.join(OUT_DIR, "au_results.json")
OUT_CSV = os.path.join(OUT_DIR, "au_results.csv")

# we rename the labels as they are misspelled in original
LABEL_RENAME = {
    "perceived_arousal": "preceived_arousal",
    "perceived_valence": "preceived_valance",
    "felt_arousal": "felt_arousal",
    "felt_valence": "felt_valance",
}

# NeuralNetFastAI only appears in these two targets in Table 6
TARGETS = ["preceived_valance", "felt_arousal"]

#Untoggle below if we choose to run all four targets, but the challenge brief only asks for these two.
#TARGETS = ["preceived_arousal", "preceived_valance", "felt_arousal", "felt_valence"]

#Read in
df = pd.read_csv(CSV_PATH)

au_cols = []
for column in df.columns:
    if column.startswith("AU"):
        au_cols.append(column)

features_df = df[au_cols]



rating_columns = ["perceived_arousal", "perceived_valence", "felt_arousal", "felt_valence"]
ratings_df = df[rating_columns]

# Rename columns to match
ratings_df = ratings_df.rename(columns=LABEL_RENAME)


classes_df = discretize_labels(ratings_df)

# Keep only our two targets
targets_df = classes_df[TARGETS]


# We only choose FASTAI, as that is the model of interest in Table 6. No need to run other
def get_best_models(predictors): 
    best_models = {}
    for target in predictors:
        best_models[target] = "FASTAI"
    return best_models


#Train the FASTAI model on the two targets 

no_search_results = {}
for target in TARGETS:
    no_search_results[target] = None

cv_results = retrain_best_model_kfold(features_df, targets_df, no_search_results, n_splits=5)


#Save results
print(json.dumps(cv_results, indent=2, ensure_ascii=False))

os.makedirs(OUT_DIR, exist_ok=True)

# as dict
with open(OUT_JSON, "w") as f:
    json.dump(cv_results, f, indent=2, ensure_ascii=False)

#as csv
results_table = pd.DataFrame(cv_results).T
results_table.index.name = "target"
results_table.to_csv(OUT_CSV)

print("Saved results to", OUT_JSON, "and", OUT_CSV)
