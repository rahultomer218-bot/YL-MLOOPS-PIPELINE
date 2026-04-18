import pandas as pd
import os
import logging
import joblib
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
# प्रोजेक्ट पाथ सेटअप (MacBook Pro के लिए अनुकूलित)
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))    # src/ folder
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) # Root folder

# ─────────────────────────────────────────────
# LOGGING सेटअप
# ─────────────────────────────────────────────
log_dir = os.path.join(PROJECT_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'evaluation.log')
file_handler  = logging.FileHandler(log_file_path)
formatter     = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ─────────────────────────────────────────────
# STEP 1: लोड मॉडल और टेस्ट डेटा
# ─────────────────────────────────────────────
def load_resources():
    try:
        model_path = os.path.join(PROJECT_DIR, "models", "random_forest_model.pkl")
        test_path  = os.path.join(PROJECT_DIR, "feature_store", "test_engineered.csv")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ मॉडल फ़ाइल नहीं मिली: {model_path}")
        
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_path)
        
        logger.info("✅ मॉडल और टेस्ट डेटा सफलतापूर्वक लोड किया गया।")
        return model, test_df
    except Exception as e:
        logger.error(f"❌ लोड करने में त्रुटि: {e}")
        raise

# ─────────────────────────────────────────────
# STEP 2: इवैल्यूएशन फंक्शन
# ─────────────────────────────────────────────
def perform_evaluation(model, test_df, target_column="Machine_Failure"):
    try:
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        y_pred       = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # मैट्रिक्स कैलकुलेशन
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "AUC-ROC": roc_auc_score(y_test, y_pred_proba)
        }

        print("\n--- MODEL EVALUATION RESULTS ---")
        for k, v in metrics.items():
            print(f"{k:<10}: {v:.4f}")
            logger.info(f"{k}: {v:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
    except Exception as e:
        logger.error(f"❌ इवैल्यूएशन में एरर: {e}")
        raise

if __name__ == "__main__":
    model, test_data = load_resources()
    perform_evaluation(model, test_data)