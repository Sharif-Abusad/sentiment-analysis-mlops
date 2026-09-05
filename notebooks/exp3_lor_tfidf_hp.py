import os
import re
import string
import logging
import warnings

import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import mlflow
import mlflow.sklearn
import dagshub


# ==============================================================
# WARNINGS
# ==============================================================

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


# ==============================================================
# LOGGING CONFIGURATION
# ==============================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "exp3_lor_tfidf_hp.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ==============================================================
# CONFIGURATION
# ==============================================================

CONFIG = {
    "data_path": "notebooks/data.csv",
    "test_size": 0.2,

    "mlflow_tracking_uri": "https://dagshub.com/Sharif-Abusad/sentiment-analysis-mlops.mlflow",

    "dagshub_repo_owner": "Sharif-Abusad",
    "dagshub_repo_name": "sentiment-analysis-mlops",

    "experiment_name": "LoR Hyperparameter Tuning"
}


# ==============================================================
# SETUP MLFLOW & DAGSHUB
# ==============================================================

logger.info("Initializing MLflow and DagsHub...")

mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])

dagshub.init(
    repo_owner=CONFIG["dagshub_repo_owner"],
    repo_name=CONFIG["dagshub_repo_name"],
    mlflow=True
)

mlflow.set_experiment(CONFIG["experiment_name"])

logger.info("MLflow experiment initialized: %s", CONFIG["experiment_name"])


# ==============================================================
# TEXT PREPROCESSING
# ==============================================================

def lemmatization(text):
    """
    Lemmatize words in the input text.
    """

    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def remove_stop_words(text):
    """
    Remove English stop words.
    """

    stop_words = set(stopwords.words("english"))

    return " ".join([word for word in text.split() if word not in stop_words])


def removing_numbers(text):
    """
    Remove numeric characters.
    """

    return "".join([char for char in text if not char.isdigit()])


def lower_case(text):
    """
    Convert text to lowercase.
    """
    return text.lower()


def removing_punctuations(text):
    """
    Remove punctuation characters.
    """

    return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)


def removing_urls(text):
    """
    Remove URLs from text.
    """

    return re.sub(r"https?://\S+|www\.\S+", "", text)


def preprocess_text(df):
    """
    Apply all text preprocessing steps.
    """
    try:

        logger.info("Starting text normalization...")

        df["review"] = df["review"].apply(lower_case)

        df["review"] = df["review"].apply(remove_stop_words)

        df["review"] = df["review"].apply(removing_numbers)

        df["review"] = df["review"].apply(removing_punctuations)

        df["review"] = df["review"].apply(removing_urls)

        df["review"] = df["review"].apply(lemmatization)

        logger.info("Text normalization completed.")
        return df

    except Exception as e:
        logger.exception("Error during text normalization.")
        raise


# ==============================================================
# LOAD & PREPROCESS DATA
# ==============================================================

def load_and_prepare_data(file_path):
    """
    Load, preprocesses and vectorizes the dataset.
    """

    try:

        logger.info("Loading dataset from: %s", file_path)

        df = pd.read_csv(file_path)

        logger.info("Dataset loaded successfully. Shape: %s", df.shape)

        df = preprocess_text(df)

        # Keep only positive and negative sentiments
        df = df[df["sentiment"].isin(["positive", "negative"])]

        logger.info("Filtered dataset shape: %s", df.shape)

        # Convert sentiment labels
        df["sentiment"] = (
            df["sentiment"]
            .replace(
                {
                    "negative": 0,
                    "positive": 1
                }
            )
            .infer_objects(
                copy=False
            )
        )

        logger.info("Sentiment labels converted to 0/1.")

        logger.info("Class distribution:\n%s", df["sentiment"].value_counts())

        # Convert text data to TF-IDF vectors
        vectorizer = TfidfVectorizer()

        logger.info("Applying %s vectorizer...", vectorizer)

        X = vectorizer.fit_transform(df['review'])
        y = df['sentiment']

        logger.info("Feature extraction completed.")
        logger.info("Feature matrix shape: %s", X.shape)

        return X, y, vectorizer

    except Exception as e:
        logger.exception("Error loading data.")
        raise


# ===============================
# Train & Log Model
# ===============================
def train_and_log_model(X, y, vectorizer):
    """
    Trains a Logistic Regression model with GridSearch and logs results to MLflow.
    """

    logger.info("=" * 70)
    logger.info("Starting Logistic Regression GridSearch experiment.")
    logger.info("Input feature matrix shape: %s", X.shape)
    logger.info("Target variable size: %s", y.shape)

    # ==========================================
    # TRAIN TEST SPLIT
    # ==========================================

    logger.info("Splitting data into train and test sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    logger.info("Train/Test split completed.")
    logger.info("Training samples: %d", X_train.shape[0])
    logger.info("Testing samples: %d", X_test.shape[0])

    # ==========================================
    # GRID SEARCH PARAMETERS
    # ==========================================
    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }

    logger.info("GridSearch parameter grid: %s", param_grid)
    logger.info("Total hyperparameter combinations: %d", 6)

    # ==========================================
    # MLFLOW PARENT RUN
    # ==========================================
    logger.info("Starting MLflow parent run...")

    with mlflow.start_run():

        logger.info("MLflow parent run started successfully.")

        # ==========================================
        # GRID SEARCH
        # ==========================================
        logger.info("Starting GridSearchCV...")
        grid_search = GridSearchCV(
            LogisticRegression(),
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        logger.info("GridSearchCV completed successfully.")
        logger.info("Best parameters found: %s", grid_search.best_params_)
        logger.info("Best cross-validation F1 score: %.4f", grid_search.best_score_)

        # ==========================================
        # LOG ALL HYPERPARAMETER RUNS
        # ==========================================

        logger.info("Logging individual hyperparameter tuning runs...")

        for params, mean_score, std_score in zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
            grid_search.cv_results_["std_test_score"],
        ):

            logger.info("Training model with parameters: %s", params)

            with mlflow.start_run(
                run_name=f"LoR with params: {params}",
                nested=True
            ):

                logger.info("Started nested MLflow run for params: %s", params)

                model = LogisticRegression(**params)

                logger.info("Fitting Logistic Regression model...")

                model.fit(X_train, y_train)

                logger.info("Model training completed.")

                # ==========================================
                # PREDICTION
                # ==========================================

                logger.info("Generating predictions on test data...")

                y_pred = model.predict(X_test)

                logger.info("Prediction completed.")

                # ==========================================
                # EVALUATION
                # ==========================================

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall": recall_score(y_test, y_pred, zero_division=0),
                    "f1_score": f1_score(y_test, y_pred, zero_division=0),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }

                logger.info("Evaluation metrics calculated: %s", metrics)

                # ==========================================
                # LOG PARAMETERS
                # ==========================================
                logger.info("Logging hyperparameters to MLflow...")

                mlflow.log_params(params)

                logger.info("Hyperparameters logged successfully.")

                # ==========================================
                # LOG METRICS
                # ==========================================
                logger.info("Logging metrics to MLflow...")

                mlflow.log_metrics(metrics)

                logger.info("Metrics logged successfully.")

                logger.info("Nested run completed for params: %s", params)

                logger.info(
                    f"Params: {params} | "
                    f"Accuracy: {metrics['accuracy']:.4f} | "
                    f"F1: {metrics['f1_score']:.4f}"
                )

        # ==========================================
        # BEST MODEL
        # ==========================================
        logger.info("Retrieving best model from GridSearch...")

        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        logger.info("Best parameters: %s", best_params)
        logger.info("Best cross-validation F1 score: %.4f", best_f1)

        # ==========================================
        # LOG BEST MODEL PARAMETERS
        # ==========================================
        logger.info("Logging best model parameters to MLflow...")

        mlflow.log_params({
            f"best_{key}": value
            for key, value in best_params.items()
        })

        logger.info("Best model parameters logged.")

        # ==========================================
        # LOG BEST MODEL METRIC
        # ==========================================
        mlflow.log_metric("best_f1_score",best_f1)

        logger.info("Best F1 score logged to MLflow.")

        # ==========================================
        # LOG BEST MODEL
        # ==========================================
        logger.info("Logging best model to MLflow...")

        mlflow.sklearn.log_model(best_model,"model")

        logger.info("Best model logged successfully.")

        # ==========================================
        # FINAL RESULT
        # ==========================================

        logger.info(
            f"\nBest Params: {best_params} | "
            f"Best F1 Score: {best_f1:.4f}"
        )

        logger.info("=" * 70)
        logger.info("Logistic Regression GridSearch experiment completed successfully.")
        logger.info("=" * 70)


# =============================
# Main Execution
# =============================
if __name__ == "__main__":
    X, y, vectorizer = load_and_prepare_data("notebooks/data.csv")
    train_and_log_model(X, y, vectorizer)