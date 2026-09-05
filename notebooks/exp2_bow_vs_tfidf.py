import os
import re
import string
import logging
import warnings

import pandas as pd
import scipy.sparse

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier

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

LOG_FILE = os.path.join(LOG_DIR, "exp2_bow_vs_tfidf.log")

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

    "experiment_name": "Bow vs TfIdf"
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


def normalize_text(df):
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

def load_data(file_path):
    """
    Load and preprocess the dataset.
    """

    try:

        logger.info("Loading dataset from: %s", file_path)

        df = pd.read_csv(file_path)

        logger.info("Dataset loaded successfully. Shape: %s", df.shape)

        df = normalize_text(df)

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

        return df

    except Exception as e:
        logger.exception("Error loading data.")
        raise


# ==============================================================
# FEATURE ENGINEERING
# ==============================================================

VECTORIZERS = {
    "BoW": CountVectorizer(),
    "TF-IDF": TfidfVectorizer()
}


ALGORITHMS = {
    "LogisticRegression": LogisticRegression(),
    "MultinomialNB": MultinomialNB(),
    "XGBoost": XGBClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}


# ==============================================================
# MODEL PARAMETER LOGGING
# ==============================================================

def log_model_params(algo_name, model):
    """
    Log model hyperparameters to MLflow.
    """

    params_to_log = {}

    if algo_name == "LogisticRegression":
        params_to_log["C"] = model.C
        params_to_log["max_iter"] = model.max_iter

    elif algo_name == "MultinomialNB":
        params_to_log["alpha"] = model.alpha

    elif algo_name == "XGBoost":
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth

    elif algo_name == "RandomForest":
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth

    elif algo_name == "GradientBoosting":
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth

    mlflow.log_params(params_to_log)

    logger.info("Logged model parameters for %s: %s", algo_name, params_to_log)


# ==============================================================
# TRAIN & EVALUATE MODELS
# ==============================================================

def train_and_evaluate(df):

    logger.info("=" * 70)
    logger.info("Starting model training and evaluation.")
    logger.info("Total experiments: %d", len(ALGORITHMS) * len(VECTORIZERS))
    logger.info("=" * 70)

    # ----------------------------------------------------------
    # Parent MLflow Run
    # ----------------------------------------------------------

    with mlflow.start_run(run_name="All Experiments"):

        logger.info("MLflow parent run started.")

        # ------------------------------------------------------
        # Loop through algorithms
        # ------------------------------------------------------
        for algo_name, algorithm in ALGORITHMS.items():

            # --------------------------------------------------
            # Loop through vectorizers
            # --------------------------------------------------
            for vec_name, vectorizer in VECTORIZERS.items():

                run_name = (f"{algo_name} with {vec_name}")

                logger.info("-" * 70)
                logger.info("Starting experiment: %s", run_name)
                logger.info("-" * 70)

                # --------------------------------------------------
                # Child MLflow Run
                # --------------------------------------------------
                with mlflow.start_run(run_name=run_name,nested=True):

                    try:

                        # ==========================================
                        # FEATURE EXTRACTION
                        # ==========================================
                        logger.info("Applying %s vectorizer...", vec_name)

                        X = vectorizer.fit_transform(df["review"])
                        y = df["sentiment"]

                        logger.info("Feature extraction completed.")

                        logger.info("Feature matrix shape: %s", X.shape)

                        # ==========================================
                        # TRAIN TEST SPLIT
                        # ==========================================
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=CONFIG["test_size"], random_state=42
                        )

                        logger.info("Train/Test split completed.")
                        logger.info("Training samples: %d", X_train.shape[0])
                        logger.info("Testing samples: %d", X_test.shape[0])

                        # ==========================================
                        # LOG PREPROCESSING PARAMETERS
                        # ==========================================
                        mlflow.log_params({
                            "vectorizer": vec_name,
                            "algorithm": algo_name,
                            "test_size": CONFIG["test_size"],
                            "num_features": X.shape[1],
                            "train_samples": X_train.shape[0],
                            "test_samples": X_test.shape[0]
                        })

                        logger.info("Preprocessing parameters logged.")

                        # ==========================================
                        # HANDLE SPARSE DATA
                        # ==========================================

                        if (algo_name == "GradientBoosting" and scipy.sparse.issparse(X_train)):

                            logger.info("Converting sparse matrix to dense for GradientBoosting.")

                            X_train_model = X_train.toarray()
                            X_test_model = X_test.toarray()

                        else:
                            X_train_model = X_train
                            X_test_model = X_test

                        # ==========================================
                        # TRAIN MODEL
                        # ==========================================

                        logger.info("Training %s model...", algo_name)

                        model = algorithm
                        model.fit(X_train_model, y_train)

                        logger.info("%s model training completed.", algo_name)

                        # ==========================================
                        # LOG MODEL PARAMETERS
                        # ==========================================

                        log_model_params(algo_name, model)

                        # ==========================================
                        # PREDICTION
                        # ==========================================

                        logger.info("Generating predictions...")

                        y_pred = model.predict(X_test_model)

                        logger.info("Prediction completed.")

                        # ==========================================
                        # EVALUATION
                        # ==========================================

                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": precision_score(y_test, y_pred, zero_division=0),
                            "recall": recall_score(y_test, y_pred, zero_division=0),
                            "f1_score": f1_score(y_test, y_pred, zero_division=0)
                        }

                        # ==========================================
                        # LOG METRICS
                        # ==========================================
                        mlflow.log_metrics(metrics)

                        logger.info("Metrics logged to MLflow.")

                        # ==========================================
                        # LOG MODEL
                        # ==========================================
                        logger.info("Logging model to MLflow...")

                        input_example = X_test_model[:5]

                        mlflow.sklearn.log_model(model, "model", input_example=input_example)

                        logger.info("Model logged successfully.")

                        # ==========================================
                        # PRINT RESULTS
                        # ==========================================
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vec_name}")
                        print(f"Metrics: {metrics}")

                        logger.info("Experiment completed successfully: %s", run_name)
                        logger.info("Metrics: %s", metrics)

                    except Exception as e:
                        logger.exception("Experiment failed: %s", run_name)

                        mlflow.log_param("status", "failed")
                        mlflow.log_param("error", str(e))

                        print(f"Error in training {algo_name} with {vec_name}: {e}")

        logger.info("=" * 70)
        logger.info("All experiments completed.")
        logger.info("=" * 70)


# ==============================================================
# EXECUTION
# ==============================================================
if __name__ == "__main__":

    logger.info("=" * 70)
    logger.info("Starting Sentiment Analysis Experiment 2")
    logger.info("=" * 70)

    try:
        df = load_data(CONFIG["data_path"])

        train_and_evaluate(df)

        logger.info("Experiment completed successfully.")

    except Exception as e:
        logger.exception("Experiment failed.")
        raise