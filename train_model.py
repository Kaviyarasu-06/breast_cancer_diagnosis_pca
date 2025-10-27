import os
import pickle
from joblib import dump as joblib_dump

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_data(path='wdbc.data'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}. Please place the original 'wdbc.data' in the working folder.")
    cols = [
        "ID", "Diagnosis",
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
        "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean",
        "fractal_dimension_mean",
        "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
        "compactness_se", "concavity_se", "concave_points_se", "symmetry_se",
        "fractal_dimension_se",
        "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
        "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst",
        "fractal_dimension_worst"
    ]
    df = pd.read_csv(path, header=None, names=cols)
    return df


def build_and_train(df, save_path='model_pipeline.pkl'):
    # Prepare data
    df = df.copy()
    df = df.drop(['ID'], axis=1)
    df['Diagnosis'] = df['Diagnosis'].map({'M': 1, 'B': 0})

    X = df.drop('Diagnosis', axis=1)
    y = df['Diagnosis']

    # Skewed features from the notebook (kept the same set)
    skewed_features = [
        'area_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean',
        'fractal_dimension_mean', 'radius_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
        'fractal_dimension_se', 'radius_worst', 'perimeter_worst', 'area_worst',
        'compactness_worst', 'concavity_worst', 'symmetry_worst', 'fractal_dimension_worst'
    ]

    # ColumnTransformer: apply Yeo-Johnson to skewed features, leave others as-is
    preprocessor = ColumnTransformer(
        transformers=[
            ("pt", PowerTransformer(method='yeo-johnson', standardize=False), skewed_features),
        ],
        remainder='passthrough',
        sparse_threshold=0
    )

    # Full pipeline: preprocessor -> scaler -> PCA -> SVC
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=6)),
        ('svc', SVC(kernel='rbf', probability=True, random_state=42))
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print('\nClassification Report:\n', classification_report(y_test, y_pred))
    print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))

    # Save pipeline + metadata
    model_artifact = {
        'pipeline': pipeline,
        'feature_names': X.columns.tolist(),
        'feature_means': X.mean().to_dict(),
        'target_mapping': {0: 'Benign', 1: 'Malignant'}
    }
    with open(save_path, 'wb') as f:
        pickle.dump(model_artifact, f)

    # Also save a joblib copy which is often more robust for sklearn objects on cloud platforms
    joblib_path = save_path.replace('.pkl', '.joblib')
    try:
        joblib_dump(model_artifact, joblib_path)
        print(f"Saved joblib model artifact to '{joblib_path}'")
    except Exception as e:
        print(f"Could not save joblib artifact: {e}")

    print(f"Saved model pipeline and metadata to '{save_path}'")


if __name__ == '__main__':
    # When run directly, load data, train, and save model
    try:
        df = load_data('wdbc.data')
    except FileNotFoundError as e:
        print(e)
        print("If you have the data elsewhere, pass it into this script or place 'wdbc.data' in the same folder.")
        raise

    build_and_train(df, save_path='model_pipeline.pkl')
