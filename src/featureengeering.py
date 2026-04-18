import pandas as pd
import numpy as np
import os
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
# STEP 1 : LOAD PROCESSED DATA
# ─────────────────────────────────────────────
def load_processed_data(artifacts_dir: str):
    """
    Load train_processed.csv and test_processed.csv
    produced by preprocessing.py from artifacts/ folder.
    """
    try:
        train_path = os.path.join(artifacts_dir, "train_processed.csv")
        test_path  = os.path.join(artifacts_dir, "test_processed.csv")

        # ── Guard: check both files exist ───────────────
        for path in [train_path, test_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"❌ Required file not found: {path}\n"
                    f"💡 Run preprocessing.py first to generate processed data."
                )

        train_df = pd.read_csv(train_path)
        test_df  = pd.read_csv(test_path)

        logger.info(f"✅ Processed Train loaded: {train_df.shape} "
                    f"| Processed Test loaded: {test_df.shape}")
        return train_df, test_df

    except FileNotFoundError as e:
        logger.error(f"❌ File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Error loading processed data: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 2 : TEMPERATURE FEATURES
# ─────────────────────────────────────────────
def add_temperature_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Temperature-based features:
      - Temp_Difference     : Process temp - Air temp
      - Temp_Ratio          : Process temp / Air temp
      - High_Temp_Flag      : 1 if Process temp > threshold
    """
    try:
        df = df.copy()

        if 'Process_temperature_K' in df.columns and 'Air_temperature_K' in df.columns:

            # Difference
            df['Temp_Difference'] = (
                df['Process_temperature_K'] - df['Air_temperature_K']
            )
            logger.info("  ✚ Feature created: 'Temp_Difference'")

            # Ratio (avoid divide by zero)
            df['Temp_Ratio'] = df['Process_temperature_K'] / (
                df['Air_temperature_K'].replace(0, np.nan)
            )
            df['Temp_Ratio'].fillna(0, inplace=True)
            logger.info("  ✚ Feature created: 'Temp_Ratio'")

            # High temperature flag (above mean)
            threshold = df['Process_temperature_K'].mean()
            df['High_Temp_Flag'] = (
                df['Process_temperature_K'] > threshold
            ).astype(int)
            logger.info(f"  ✚ Feature created: 'High_Temp_Flag' (threshold={threshold:.2f})")

        else:
            logger.info("  ⚠️  Temperature columns not found — skipping temperature features.")

        return df

    except Exception as e:
        logger.error(f"❌ Error in temperature features: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 3 : MECHANICAL / POWER FEATURES
# ─────────────────────────────────────────────
def add_mechanical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Physics-based mechanical features:
      - Power_W             : Torque × Rotational speed (Watts)
      - High_Power_Flag     : 1 if Power_W > mean
      - Tool_Wear_Risk      : Tool_wear × Torque
      - Torque_per_RPM      : Torque / RPM ratio
    """
    try:
        df = df.copy()

        # Power
        if 'Torque_Nm' in df.columns and 'Rotational_speed_rpm' in df.columns:
            df['Power_W'] = df['Torque_Nm'] * df['Rotational_speed_rpm']
            logger.info("  ✚ Feature created: 'Power_W'")

            # High power flag
            power_threshold = df['Power_W'].mean()
            df['High_Power_Flag'] = (df['Power_W'] > power_threshold).astype(int)
            logger.info(f"  ✚ Feature created: 'High_Power_Flag' "
                        f"(threshold={power_threshold:.2f})")

            # Torque per RPM ratio (avoid divide by zero)
            df['Torque_per_RPM'] = df['Torque_Nm'] / (
                df['Rotational_speed_rpm'].replace(0, np.nan)
            )
            df['Torque_per_RPM'].fillna(0, inplace=True)
            logger.info("  ✚ Feature created: 'Torque_per_RPM'")

        else:
            logger.info("  ⚠️  Torque/RPM columns not found — skipping mechanical features.")

        # Tool wear risk
        if 'Tool_wear_min' in df.columns and 'Torque_Nm' in df.columns:
            df['Tool_Wear_Risk'] = df['Tool_wear_min'] * df['Torque_Nm']
            logger.info("  ✚ Feature created: 'Tool_Wear_Risk'")

            # High wear flag
            wear_threshold = df['Tool_wear_min'].mean()
            df['High_Wear_Flag'] = (df['Tool_wear_min'] > wear_threshold).astype(int)
            logger.info(f"  ✚ Feature created: 'High_Wear_Flag' "
                        f"(threshold={wear_threshold:.2f})")

        else:
            logger.info("  ⚠️  Tool wear columns not found — skipping wear features.")

        return df

    except Exception as e:
        logger.error(f"❌ Error in mechanical features: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 4 : INTERACTION FEATURES
# ─────────────────────────────────────────────
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction features combining multiple columns:
      - Temp_Power_Index    : Temp_Difference × Power_W
      - Wear_Temp_Index     : Tool_Wear_Risk × Temp_Difference
      - Overall_Risk_Score  : sum of all flag columns
    """
    try:
        df = df.copy()

        # Temp × Power interaction
        if 'Temp_Difference' in df.columns and 'Power_W' in df.columns:
            df['Temp_Power_Index'] = df['Temp_Difference'] * df['Power_W']
            logger.info("  ✚ Feature created: 'Temp_Power_Index'")

        # Wear × Temp interaction
        if 'Tool_Wear_Risk' in df.columns and 'Temp_Difference' in df.columns:
            df['Wear_Temp_Index'] = df['Tool_Wear_Risk'] * df['Temp_Difference']
            logger.info("  ✚ Feature created: 'Wear_Temp_Index'")

        # Overall risk score — sum of all binary flag columns
        flag_cols = [col for col in df.columns if col.endswith('_Flag')]
        if flag_cols:
            df['Overall_Risk_Score'] = df[flag_cols].sum(axis=1)
            logger.info(f"  ✚ Feature created: 'Overall_Risk_Score' "
                        f"(from flags: {flag_cols})")

        return df

    except Exception as e:
        logger.error(f"❌ Error in interaction features: {e}")
        raise


# ─────────────────────────────────────────────
# STEP 5 : SAVE ENGINEERED DATA
# ─────────────────────────────────────────────
def save_engineered_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Save feature-engineered train/test to a separate
    feature_store/ folder for use by model training.
    """
    try:
        feature_store_dir = os.path.join(PROJECT_DIR, "feature_store")
        os.makedirs(feature_store_dir, exist_ok=True)

        train_path = os.path.join(feature_store_dir, "train_engineered.csv")
        test_path  = os.path.join(feature_store_dir, "test_engineered.csv")

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path,  index=False)

        logger.info(f"📂 Train set saved to: {train_path}  | shape: {train_df.shape}")
        logger.info(f"📂 Test  set saved to: {test_path}   | shape: {test_df.shape}")

        return train_path, test_path

    except Exception as e:
        logger.error(f"❌ Error saving engineered data: {e}")
        raise


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    """
    Feature Engineering pipeline entry point.
    Reads from artifacts/, engineers features,
    and saves to feature_store/ for model training.
    """
    TARGET        = "Machine_Failure"
    ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
    FEATURE_DIR   = os.path.join(PROJECT_DIR, "feature_store")

    logger.info(f"📁 Project root   : {PROJECT_DIR}")
    logger.info(f"📂 Artifacts dir  : {ARTIFACTS_DIR}")
    logger.info(f"📂 Feature store  : {FEATURE_DIR}")

    try:
        # ── Step 1: Load Processed Data ──────────────────
        logger.info("--- STEP 1: Loading Processed Data ---")
        train_df, test_df = load_processed_data(ARTIFACTS_DIR)

        # ── Step 2: Temperature Features ─────────────────
        logger.info("--- STEP 2: Adding Temperature Features ---")
        train_df = add_temperature_features(train_df)
        test_df  = add_temperature_features(test_df)

        # ── Step 3: Mechanical Features ───────────────────
        logger.info("--- STEP 3: Adding Mechanical Features ---")
        train_df = add_mechanical_features(train_df)
        test_df  = add_mechanical_features(test_df)

        # ── Step 4: Interaction Features ──────────────────
        logger.info("--- STEP 4: Adding Interaction Features ---")
        train_df = add_interaction_features(train_df)
        test_df  = add_interaction_features(test_df)

        logger.info(f"✅ Final Train shape: {train_df.shape}")
        logger.info(f"✅ Final Test  shape: {test_df.shape}")
        logger.info(f"📊 New features added: "
                    f"{[col for col in train_df.columns if col not in ['UDI', 'Product_ID', 'Type', 'Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min', TARGET]]}")

        # ── Step 5: Save to feature_store/ ───────────────
        logger.info("--- STEP 5: Saving Engineered Data ---")
        train_path, test_path = save_engineered_data(train_df, test_df)

        logger.info(f"📂 Train set saved to: {train_path}")
        logger.info(f"📂 Test  set saved to: {test_path}")

    except Exception as e:
        logger.error(f"❌ Feature engineering pipeline failed: {e}")
        raise


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()