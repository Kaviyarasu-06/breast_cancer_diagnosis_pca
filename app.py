import pickle
import pandas as pd
import streamlit as st


def load_model(path='model_pipeline.pkl'):
    """Try to load the saved model artifact. Attempt joblib first then pickle. Returns the artifact dict or raises.

    We avoid caching here to make error reporting simpler on deployment platforms.
    """
    # Import common ML packages early to help unpickling on some platforms
    try:
        import sklearn  # noqa: F401
        import numpy as np  # noqa: F401
    except Exception:
        # non-fatal here; we'll still try to load and let pickle/joblib raise helpful errors
        pass

    # Try joblib first (safe for sklearn objects), then fallback to pickle
    try:
        from joblib import load as joblib_load
    except Exception:
        joblib_load = None

    if joblib_load is not None:
        try:
            return joblib_load(path.replace('.pkl', '.joblib'))
        except FileNotFoundError:
            # joblib file not present — fall back to pickle
            pass
        except Exception:
            # if joblib exists but fails, continue to try pickle and show full error later
            pass

    # Fallback to pickle
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    st.title('Breast Cancer Diagnosis (SVC + PCA)')
    st.write('Provide feature values in the sidebar and click Predict.')

    try:
        art = load_model('model_pipeline.pkl')
    except FileNotFoundError:
        st.error("Model file 'model_pipeline.pkl' not found. Run 'python train_model.py' first to create it.")
        return
    except Exception as e:
        st.error("Failed to load model artifact. See exception below for details.")
        st.exception(e)
        return

    pipeline = art['pipeline']
    feature_names = art['feature_names']
    feature_means = art.get('feature_means', {})
    target_mapping = art.get('target_mapping', {0: 'Benign', 1: 'Malignant'})

    st.sidebar.header('Input features')
    inputs = {}
    for feat in feature_names:
        default = float(feature_means.get(feat, 0.0))
        # Using a wide range; user can type values or use arrows
        inputs[feat] = st.sidebar.number_input(feat, value=default, format="%.6f")

    if st.sidebar.button('Predict'):
        df_input = pd.DataFrame([inputs], columns=feature_names)
        pred = pipeline.predict(df_input)
        proba = pipeline.predict_proba(df_input) if hasattr(pipeline, 'predict_proba') else None

        label = target_mapping.get(int(pred[0]), str(pred[0]))
        st.subheader('Prediction')
        st.write(f'Diagnosis: **{label}**')
        if proba is not None:
            # Find probability for predicted class
            prob = proba[0][int(pred[0])]
            st.write(f'Probability: **{prob:.3f}**')

        st.write('Raw prediction array:', pred)

    st.sidebar.markdown('---')
    st.sidebar.write('Model file: model_pipeline.pkl')


if __name__ == '__main__':
    main()
