import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# ─────────────────────────────────────────────
# DYNAMIC BASE PATH  (works from anywhere)
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))    # → .../src/
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) # → .../YL-MLOOPS-PIPELINE/

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
log_dir = os.path.join(PROJECT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

# Prevent duplicate log handlers if module is reloaded
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_file_path = os.path.join(log_dir, 'data_ingestion.log')
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ─────────────────────────────────────────────
# STEP 1 : LOAD DATA
# ─────────────────────────────────────────────
def load_data(data_url: str) -> pd.DataFrame:
    """Load raw data from a path or URL."""
    try:
        logger.info(f"🚀 Attempting to load data from: {data_url}")
        data = pd.read_csv(data_url)
        logger.info(f"✅ Data loaded successfully. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        logger.error(f"❌ File not found: {data_url}")
        logger.error(f"💡 Make sure the file exists at the correct path.")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading data from {data_url}: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 2 : PREPROCESS DATA
# ─────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Clean and prepare the data:
      - Handle missing values
      - Encode categorical columns
      - Scale numerical columns (target column is never touched)
    """
    try:
        logger.info("🔧 Starting data preprocessing...")
        df = df.copy()  # never mutate the original

        # ── Validate target column exists ───────────────
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame. "
                             f"Available columns: {df.columns.tolist()}")

        # ── Missing Values ──────────────────────────────
        missing_before = df.isnull().sum().sum()
        logger.info(f"Missing values before cleaning: {missing_before}")

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in [np.float64, np.int64]:
                    df[col].fillna(df[col].median(), inplace=True)
                    logger.info(f"  Filled numeric column '{col}' with median.")
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
                    logger.info(f"  Filled categorical column '{col}' with mode.")

        logger.info(f"Missing values after cleaning: {df.isnull().sum().sum()}")

        # ── Encode Categorical Columns ──────────────────
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))
            logger.info(f"  Label encoded column: '{col}'")

        # ── Scale Numerical Columns ─────────────────────
        # NOTE: Only scale true numerical columns, not label-encoded categoricals
        encoded_cols = categorical_cols  # already encoded above
        numerical_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns.tolist()
            if col != target_column and col not in encoded_cols
        ]

        if numerical_cols:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            logger.info(f"  Scaled {len(numerical_cols)} numerical column(s): {numerical_cols}")
        else:
            logger.info("  No numerical columns to scale.")

        logger.info(f"✅ Preprocessing complete. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"❌ Error during preprocessing: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 3 : SPLIT & SAVE
# ─────────────────────────────────────────────
def split_and_save_data(df: pd.DataFrame, target_column: str):
    """Split data into train/test and save to artifacts/ folder."""
    try:
        artifacts_dir = os.path.join(PROJECT_DIR, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        logger.info(f"Splitting data with stratify on target: '{target_column}'")
        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df[target_column]
        )

        train_path = os.path.join(artifacts_dir, "train.csv")
        test_path  = os.path.join(artifacts_dir, "test.csv")

        train_set.to_csv(train_path, index=False)
        test_set.to_csv(test_path,  index=False)

        logger.info(f"📂 Train set saved to: {train_path}  | shape: {train_set.shape}")
        logger.info(f"📂 Test  set saved to: {test_path}   | shape: {test_set.shape}")

        return train_path, test_path

    except Exception as e:
        logger.error(f"❌ Error during data splitting/saving: {e}")
        raise


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    """
    Main pipeline entry point.
    Orchestrates data loading, preprocessing, and splitting.
    """
    # ── Config ──────────────────────────────────────
    FILE_PATH = os.path.join(PROJECT_DIR, "experiments", "spam.csv")
    TARGET    = "Machine_Failure"

    # ── Guard: check file exists BEFORE loading ──────
    if not os.path.exists(FILE_PATH):
        logger.error(f"❌ File not found: {FILE_PATH}")
        logger.error(f"💡 Make sure 'spam.csv' is inside the 'experiments/' folder.")
        return  # exit cleanly instead of crashing

    logger.info("=" * 60)
    logger.info("🚀 ML PIPELINE STARTED")
    logger.info("=" * 60)
    logger.info(f"📁 Project root  : {PROJECT_DIR}")
    logger.info(f"📄 Data file path: {FILE_PATH}")
    logger.info(f"🎯 Target column : {TARGET}")

    try:
        # ── Step 1: Load ─────────────────────────────
        logger.info("--- STEP 1: Loading Data ---")
        raw_df = load_data(FILE_PATH)

        # ── Step 2: Preprocess ────────────────────────
        logger.info("--- STEP 2: Preprocessing Data ---")
        processed_df = preprocess_data(raw_df, target_column=TARGET)

        # ── Step 3: Split & Save ──────────────────────
        logger.info("--- STEP 3: Splitting & Saving Data ---")
        train_path, test_path = split_and_save_data(processed_df, TARGET)

        logger.info("=" * 60)
        logger.info("✅ ML PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"   Train → {train_path}")
        logger.info(f"   Test  → {test_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()