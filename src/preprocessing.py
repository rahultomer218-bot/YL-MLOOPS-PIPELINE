import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
import logging

# ─────────────────────────────────────────────
# DYNAMIC BASE PATH  (works from anywhere)
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))    # → .../src/
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) # → .../YL-MLOOPS-PIPELINE/

# ─────────────────────────────────────────────
# LOGGING SETUP  (same as data_ingestion.py)
# ─────────────────────────────────────────────
log_dir = os.path.join(PROJECT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')   # ← same logger object
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_ingestion.log')  # ← same log file
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ─────────────────────────────────────────────
# STEP 1 : LOAD ARTIFACTS
# ─────────────────────────────────────────────
def load_artifacts(artifacts_dir: str):
    """
    Load train and test CSVs from the artifacts/ folder
    produced by data_ingestion.py.
    """
    try:
        train_path = os.path.join(artifacts_dir, "train.csv")
        test_path  = os.path.join(artifacts_dir, "test.csv")

        # ── Guard: check both files exist ───────────────
        for path in [train_path, test_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"❌ Required file not found: {path}\n"
                    f"💡 Run data_ingestion.py first to generate artifacts."
                )

        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        logger.info(f"✅ Train loaded: {train_df.shape} | Test loaded: {test_df.shape}")
        return train_df, test_df

    except FileNotFoundError:
        logger.error(f"❌ File not found. Run data_ingestion.py first.")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading artifacts: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 2 : HANDLE MISSING VALUES
# ─────────────────────────────────────────────
def handle_missing_values(df: pd.DataFrame, label: str = "DataFrame") -> pd.DataFrame:
    """
    Fill missing values:
      - Numerical columns → median
      - Categorical columns → mode
    """
    try:
        df = df.copy()
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
        return df

    except Exception as e:
        logger.error(f"❌ Error handling missing values: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 3 : REMOVE OUTLIERS
# ─────────────────────────────────────────────
def remove_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    target_column: str, contamination: float = 0.05):
    """
    Remove outliers using IsolationForest.
      - Fitted on train only (to avoid data leakage)
      - Applied to both train and test
      - Target column is excluded
    """
    try:
        logger.info(f"🔍 Detecting outliers with IsolationForest "
                    f"(contamination={contamination})...")

        feature_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns
                        if c != target_column]

        iso = IsolationForest(contamination=contamination, random_state=42)
        iso.fit(train_df[feature_cols])

        # ── Train ────────────────────────────────────────
        train_mask    = iso.predict(train_df[feature_cols]) == 1
        train_before  = len(train_df)
        train_df      = train_df[train_mask].reset_index(drop=True)
        logger.info(f"  Train: removed {train_before - len(train_df)} outlier rows "
                    f"({train_before} → {len(train_df)})")

        # ── Test ─────────────────────────────────────────
        test_mask     = iso.predict(test_df[feature_cols]) == 1
        test_before   = len(test_df)
        test_df       = test_df[test_mask].reset_index(drop=True)
        logger.info(f"  Test : removed {test_before - len(test_df)} outlier rows "
                    f"({test_before} → {len(test_df)})")

        logger.info(f"✅ Outlier removal complete.")
        return train_df, test_df

    except Exception as e:
        logger.error(f"❌ Error during outlier removal: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 4 : ENCODE CATEGORICAL COLUMNS
# ─────────────────────────────────────────────
def encode_categoricals(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        target_column: str):
    """
    Label encode categorical columns.
      - Fitted on train only (to avoid data leakage)
      - Applied to both train and test
      - Target column is excluded
    """
    try:
        categorical_cols = train_df.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()

        if target_column in categorical_cols:
            categorical_cols.remove(target_column)

        if not categorical_cols:
            logger.info("  No categorical columns to encode.")
            return train_df, test_df, []

        le = LabelEncoder()
        for col in categorical_cols:
            le.fit(train_df[col].astype(str))
            train_df[col] = le.transform(train_df[col].astype(str))

            # Handle unseen labels in test set safely
            test_df[col] = test_df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            test_df[col] = le.transform(test_df[col])
            logger.info(f"  Label encoded column: '{col}'")

        logger.info(f"  Scaled {len(categorical_cols)} categorical column(s).")
        return train_df, test_df, categorical_cols

    except Exception as e:
        logger.error(f"❌ Error during encoding: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 5 : FEATURE ENGINEERING
# ─────────────────────────────────────────────
def feature_engineering(train_df: pd.DataFrame,
                        test_df: pd.DataFrame,
                        target_column: str):
    """
    Create new meaningful features from existing ones.
    Based on the spam.csv columns:
      - Air_temperature_K, Process_temperature_K,
        Rotational_speed_rpm, Torque_Nm, Tool_wear_min
    """
    try:
        logger.info("⚙️  Starting feature engineering...")

        for df in [train_df, test_df]:

            # Temperature difference between process and air
            if 'Process_temperature_K' in df.columns and 'Air_temperature_K' in df.columns:
                df['Temp_Difference'] = (
                    df['Process_temperature_K'] - df['Air_temperature_K']
                )
                logger.info("  ✚ Feature created: 'Temp_Difference'")

            # Power = Torque × Rotational speed (physics-based)
            if 'Torque_Nm' in df.columns and 'Rotational_speed_rpm' in df.columns:
                df['Power_W'] = df['Torque_Nm'] * df['Rotational_speed_rpm']
                logger.info("  ✚ Feature created: 'Power_W'")

            # Tool wear risk = Tool_wear × Torque
            if 'Tool_wear_min' in df.columns and 'Torque_Nm' in df.columns:
                df['Tool_Wear_Risk'] = df['Tool_wear_min'] * df['Torque_Nm']
                logger.info("  ✚ Feature created: 'Tool_Wear_Risk'")

        logger.info("✅ Feature engineering complete.")
        return train_df, test_df

    except Exception as e:
        logger.error(f"❌ Error during feature engineering: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 6 : SCALE NUMERICAL COLUMNS
# ─────────────────────────────────────────────
def scale_numerical(train_df: pd.DataFrame, test_df: pd.DataFrame,
                    target_column: str, encoded_cols: list):
    """
    StandardScale numerical columns.
      - Fitted on train only (to avoid data leakage)
      - Applied to both train and test
      - Target column and encoded categoricals are excluded
    """
    try:
        numerical_cols = [
            col for col in train_df.select_dtypes(include=[np.number]).columns
            if col != target_column and col not in encoded_cols
        ]

        if not numerical_cols:
            logger.info("  No numerical columns to scale.")
            return train_df, test_df

        scaler = StandardScaler()
        train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
        test_df[numerical_cols]  = scaler.transform(test_df[numerical_cols])

        logger.info(f"  Scaled {len(numerical_cols)} numerical column(s): {numerical_cols}")
        logger.info("✅ Scaling complete.")
        return train_df, test_df

    except Exception as e:
        logger.error(f"❌ Error during scaling: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 7 : SAVE PROCESSED DATA
# ─────────────────────────────────────────────
def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Save final processed train/test to artifacts/ folder."""
    try:
        artifacts_dir = os.path.join(PROJECT_DIR, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        train_path = os.path.join(artifacts_dir, "train_processed.csv")
        test_path  = os.path.join(artifacts_dir, "test_processed.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        logger.info(f"📂 Train set saved to: {train_path}  | shape: {train_df.shape}")
        logger.info(f"📂 Test  set saved to: {test_path}   | shape: {test_df.shape}")

        return train_path, test_path

    except Exception as e:
        logger.error(f"❌ Error saving processed data: {e}")
        raise


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    """
    Preprocessing pipeline entry point.
    Reads from artifacts/, processes, and saves back to artifacts/.
    """
    TARGET        = "Machine_Failure"
    ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
    CONTAMINATION = 0.05   # 5% outlier threshold — adjust as needed

    logger.info(f"📁 Project root  : {PROJECT_DIR}")
    logger.info(f"📂 Artifacts dir : {ARTIFACTS_DIR}")

    try:
        # ── Step 1: Load ─────────────────────────────────
        logger.info("--- STEP 1: Loading Artifacts ---")
        train_df, test_df = load_artifacts(ARTIFACTS_DIR)

        # ── Step 2: Missing Values ────────────────────────
        logger.info("--- STEP 2: Handling Missing Values ---")
        train_df = handle_missing_values(train_df, label="Train")
        test_df  = handle_missing_values(test_df,  label="Test")

        # ── Step 3: Outlier Removal ───────────────────────
        logger.info("--- STEP 3: Removing Outliers ---")
        train_df, test_df = remove_outliers(
            train_df, test_df, TARGET, contamination=CONTAMINATION
        )

        # ── Step 4: Encode Categoricals ───────────────────
        logger.info("--- STEP 4: Encoding Categorical Columns ---")
        train_df, test_df, encoded_cols = encode_categoricals(
            train_df, test_df, TARGET
        )

        # ── Step 5: Feature Engineering ───────────────────
        logger.info("--- STEP 5: Feature Engineering ---")
        train_df, test_df = feature_engineering(train_df, test_df, TARGET)

        # ── Step 6: Scale Numerical ───────────────────────
        logger.info("--- STEP 6: Scaling Numerical Columns ---")
        train_df, test_df = scale_numerical(
            train_df, test_df, TARGET, encoded_cols
        )

        # ── Step 7: Save ──────────────────────────────────
        logger.info("--- STEP 7: Saving Processed Data ---")
        train_path, test_path = save_processed_data(train_df, test_df)

        logger.info(f"📂 Train set saved to: {train_path}")
        logger.info(f"📂 Test  set saved to: {test_path}")

    except Exception as e:
        logger.error(f"❌ Error during data splitting/saving: {e}")
        raise


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()