import os
import pickle
from joblib import dump as joblib_dump

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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

    # Instead of using ColumnTransformer (which can cause cross-version pickle issues),
    # we fit the PowerTransformer, StandardScaler, PCA and SVC separately and save them
    # individually. This makes loading in different environments more robust.

    pt = PowerTransformer(method='yeo-johnson', standardize=False)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit PowerTransformer on skewed features using training data
    pt.fit(X_train[skewed_features])

    # Apply transformation to full training set
    X_train_trans = X_train.copy()
    X_train_trans[skewed_features] = pt.transform(X_train[skewed_features])

    # Fit scaler on transformed training set
    scaler = StandardScaler()
    scaler.fit(X_train_trans)

    # Scale training data
    X_train_scaled = scaler.transform(X_train_trans)

    # Fit PCA on scaled training data
    pca = PCA(n_components=6)
    pca.fit(X_train_scaled)

    X_train_pca = pca.transform(X_train_scaled)

    # Fit SVC
    svc = SVC(kernel='rbf', probability=True, random_state=42)
    svc.fit(X_train_pca, y_train)

    # For convenience, assemble a pipeline-like object for local use (but we won't pickle ColumnTransformer)
    try:
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline([
            ('pt_passthrough', pt),
            ('scaler', scaler),
            ('pca', pca),
            ('svc', svc)
        ])
    except Exception:
        pipeline = None

    # Prepare test set using the same manual transforms
    X_test_trans = X_test.copy()
    X_test_trans[skewed_features] = pt.transform(X_test_trans[skewed_features])
    X_test_scaled = scaler.transform(X_test_trans)
    X_test_pca = pca.transform(X_test_scaled)

    # Evaluate
    y_pred = svc.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print('\nClassification Report:\n', classification_report(y_test, y_pred))
    print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))

    # Save pipeline + metadata
    model_artifact = {
        # Individual components
        'pt': pt,
        'scaler': scaler,
        'pca': pca,
        'svc': svc,
        'pipeline': pipeline,  # may be None on some envs
        'skewed_features': skewed_features,
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
