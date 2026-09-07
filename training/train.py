"""
Train a Titanic survival classifier.

What this script does, step by step:
1. Loads the classic Titanic passenger dataset (built into seaborn).
2. Cleans it up (fills in missing ages/embarkation ports).
3. Builds a scikit-learn Pipeline that encodes the categorical columns
   (sex, embarked) and feeds everything into a RandomForestClassifier.
4. Splits the data into train/test sets, trains the model, and evaluates it.
5. Saves the trained pipeline to model/titanic_model.joblib so the FastAPI
   backend can load it later.

Run it with:
    python train.py
"""

import pathlib

import joblib
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# The columns our model will use to make predictions.
FEATURE_COLUMNS = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
TARGET_COLUMN = "survived"

# Where the trained model gets saved. The FastAPI backend loads it from here.
MODEL_PATH = pathlib.Path(__file__).resolve().parent.parent / "app" / "backend" / "model" / "titanic_model.joblib"


def load_data() -> pd.DataFrame:
    """Load the classic Titanic dataset that ships with seaborn."""
    df = sns.load_dataset("titanic")
    return df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def build_pipeline() -> Pipeline:
    """
    Build a pipeline that handles missing values + categorical encoding,
    then trains a RandomForestClassifier.

    Using a Pipeline means we save ONE object that knows how to go straight
    from raw passenger details to a prediction - the backend doesn't need
    to duplicate any preprocessing logic.
    """
    numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked"]

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])


def main():
    print("Loading Titanic dataset...")
    df = load_data()
    print(f"Loaded {len(df)} passengers.")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("\nEvaluating on held-out test data...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"]))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
