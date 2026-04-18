import pandas as pd
import numpy as np
import os
import logging
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ─────────────────────────────────────────────
# DYNAMIC BASE PATH  (works from anywhere)
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))    # → .../src/
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) # → .../YL-MLOOPS-PIPELINE/

# ─────────────────────────────────────────────
# LOGGING SETUP  (same as all pipeline files)
# ─────────────────────────────────────────────
log_dir = os.path.join(PROJECT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')      # ← same logger object
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_ingestion.log')  # ← same log file
file_handler  = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ─────────────────────────────────────────────
# STEP 1 : LOAD ENGINEERED DATA
# ─────────────────────────────────────────────
def load_engineered_data(feature_store_dir: str):
    """
    Load train_engineered.csv and test_engineered.csv
    produced by featureengineering.py from feature_store/ folder.
    """
    try:
        train_path = os.path.join(feature_store_dir, "train_engineered.csv")
        test_path  = os.path.join(feature_store_dir, "test_engineered.csv")

        # ── Guard: check both files exist ───────────────
        for path in [train_path, test_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"❌ Required file not found: {path}\n"
                    f"💡 Run featureengineering.py first to generate engineered data."
                )

        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        logger.info(f"✅ Engineered Train loaded: {train_df.shape} "
                    f"| Engineered Test loaded: {test_df.shape}")
        return train_df, test_df

    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading engineered data: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 2 : SPLIT FEATURES & TARGET
# ─────────────────────────────────────────────
def split_features_target(train_df: pd.DataFrame,
                           test_df: pd.DataFrame,
                           target_column: str):
    """
    Separate features (X) and target (y)
    for both train and test sets.
    """
    try:
        # ── Validate target column ───────────────────────
        for df, name in [(train_df, "Train"), (test_df, "Test")]:
            if target_column not in df.columns:
                raise ValueError(
                    f"❌ Target column '{target_column}' not found in {name} set.\n"
                    f"   Available columns: {df.columns.tolist()}"
                )

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]

        X_test  = test_df.drop(columns=[target_column])
        y_test  = test_df[target_column]

        logger.info(f"✅ X_train: {X_train.shape} | y_train: {y_train.shape}")
        logger.info(f"✅ X_test : {X_test.shape}  | y_test : {y_test.shape}")
        logger.info(f"📊 Target distribution (Train):\n"
                    f"{y_train.value_counts().to_string()}")

        return X_train, y_train, X_test, y_test

    except Exception as e:
        logger.error(f"❌ Error splitting features and target: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 3 : TRAIN RANDOM FOREST MODEL
# ─────────────────────────────────────────────
def train_random_forest(X_train: pd.DataFrame,
                         y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier on the training data.
    """
    try:
        logger.info("🌲 Initializing Random Forest Classifier...")

        model = RandomForestClassifier(
            n_estimators      = 200,     # number of trees
            max_depth         = 10,      # max depth of each tree
            min_samples_split = 5,       # min samples to split a node
            min_samples_leaf  = 2,       # min samples at a leaf node
            max_features      = 'sqrt',  # features to consider at each split
            class_weight      = 'balanced',  # handle class imbalance
            random_state      = 42,
            n_jobs            = -1       # use all CPU cores
        )

        logger.info(f"⚙️  Model parameters:")
        logger.info(f"    n_estimators      : 200")
        logger.info(f"    max_depth         : 10")
        logger.info(f"    min_samples_split : 5")
        logger.info(f"    min_samples_leaf  : 2")
        logger.info(f"    max_features      : sqrt")
        logger.info(f"    class_weight      : balanced")
        logger.info(f"    random_state      : 42")

        logger.info("🚀 Training started...")
        model.fit(X_train, y_train)
        logger.info("✅ Model training complete.")

        return model

    except Exception as e:
        logger.error(f"❌ Error during model training: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 4 : EVALUATE MODEL
# ─────────────────────────────────────────────
def evaluate_model(model: RandomForestClassifier,
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame,  y_test: pd.Series):
    """
    Evaluate the trained model on both train and test sets.
    Logs accuracy, precision, recall, F1, AUC-ROC,
    confusion matrix, and classification report.
    """
    try:
        logger.info("📊 Evaluating model...")

        for split_name, X, y in [("TRAIN", X_train, y_train),
                                   ("TEST",  X_test,  y_test)]:

            y_pred     = model.predict(X)
            y_pred_proba = model.predict_proba(X)[:, 1]

            accuracy   = accuracy_score(y, y_pred)
            precision  = precision_score(y, y_pred, zero_division=0)
            recall     = recall_score(y, y_pred, zero_division=0)
            f1         = f1_score(y, y_pred, zero_division=0)
            auc_roc    = roc_auc_score(y, y_pred_proba)
            cm         = confusion_matrix(y, y_pred)
            report     = classification_report(y, y_pred, zero_division=0)

            logger.info(f"")
            logger.info(f"─── {split_name} SET METRICS ───────────────────────")
            logger.info(f"  Accuracy  : {accuracy:.4f}  ({accuracy*100:.2f}%)")
            logger.info(f"  Precision : {precision:.4f}")
            logger.info(f"  Recall    : {recall:.4f}")
            logger.info(f"  F1 Score  : {f1:.4f}")
            logger.info(f"  AUC-ROC   : {auc_roc:.4f}")
            logger.info(f"  Confusion Matrix:\n{cm}")
            logger.info(f"  Classification Report:\n{report}")

        # ── Feature Importance ───────────────────────────
        feature_importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        logger.info(f"")
        logger.info(f"─── TOP 10 FEATURE IMPORTANCES ─────────────────────")
        for feat, score in feature_importance.head(10).items():
            logger.info(f"  {feat:<35} : {score:.4f}")

        return feature_importance

    except Exception as e:
        logger.error(f"❌ Error during model evaluation: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 5 : SAVE MODEL
# ─────────────────────────────────────────────
def save_model(model: RandomForestClassifier,
               feature_importance: pd.Series):
    """
    Save the trained model and feature importances
    to the models/ folder for future use.
    """
    try:
        models_dir = os.path.join(PROJECT_DIR, "models")
        os.makedirs(models_dir, exist_ok=True)

        # ── Save model ───────────────────────────────────
        model_path = os.path.join(models_dir, "random_forest_model.pkl")
        joblib.dump(model, model_path)
        logger.info(f"📂 Model saved to       : {model_path}")

        # ── Save feature importances ─────────────────────
        fi_path = os.path.join(models_dir, "feature_importances.csv")
        feature_importance.reset_index().rename(
            columns={'index': 'Feature', 0: 'Importance'}
        ).to_csv(fi_path, index=False)
        logger.info(f"📂 Feature importance saved to: {fi_path}")

        return model_path, fi_path

    except Exception as e:
        logger.error(f"❌ Error saving model: {e}")
        raise


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    """
    Model Training pipeline entry point.
    Reads from feature_store/, trains RandomForest,
    evaluates, and saves model to models/ folder.
    """
    TARGET            = "Machine_Failure"
    FEATURE_STORE_DIR = os.path.join(PROJECT_DIR, "feature_store")
    MODELS_DIR        = os.path.join(PROJECT_DIR, "models")

    logger.info(f"📁 Project root     : {PROJECT_DIR}")
    logger.info(f"📂 Feature store    : {FEATURE_STORE_DIR}")
    logger.info(f"📂 Models dir       : {MODELS_DIR}")

    try:
        # ── Step 1: Load Engineered Data ─────────────────
        logger.info("--- STEP 1: Loading Engineered Data ---")
        train_df, test_df = load_engineered_data(FEATURE_STORE_DIR)

        # ── Step 2: Split Features & Target ──────────────
        logger.info("--- STEP 2: Splitting Features & Target ---")
        X_train, y_train, X_test, y_test = split_features_target(
            train_df, test_df, TARGET
        )

        # ── Step 3: Train Random Forest ───────────────────
        logger.info("--- STEP 3: Training Random Forest Model ---")
        model = train_random_forest(X_train, y_train)

        # ── Step 4: Evaluate Model ────────────────────────
        logger.info("--- STEP 4: Evaluating Model ---")
        feature_importance = evaluate_model(
            model, X_train, y_train, X_test, y_test
        )

        # ── Step 5: Save Model ────────────────────────────
        logger.info("--- STEP 5: Saving Model ---")
        model_path, fi_path = save_model(model, feature_importance)

        logger.info(f"📂 Model saved to           : {model_path}")
        logger.info(f"📂 Feature importance saved : {fi_path}")

    except Exception as e:
        logger.error(f"❌ Model training pipeline failed: {e}")
        raise


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()