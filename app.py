import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_models():
    # Update paths if `models` is not in the same directory as `app.py`
    rf_model = joblib.load('./models/random_forest_model.pkl')
    gb_model = joblib.load('./models/gradient_boosting_model.pkl')
    return rf_model, gb_model

@st.cache_resource
def load_feature_names():
    # Update the path if needed
    with open("models/feature_names.txt", "r") as f:
        return [line.strip() for line in f]

def preprocess_input(data, feature_names):
    try:
        columns_to_split = ['R_TOTAL_STR.', 'B_TOTAL_STR.']
        for col in columns_to_split:
            data[[col + "_Attempted", col + "_Landed"]] = data[col].str.split(" of ", expand=True).astype(int)

        data.drop(columns=columns_to_split, inplace=True)
        data = data[feature_names]  # Select only feature columns
        return data
    except KeyError as e:
        st.error(f"Missing or misnamed columns in input data: {e}")
        return None

def main():
    st.title("UFC Fight Prediction")
    st.write("Upload fight data and get predictions.")
    
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.write(data)

        rf_model, gb_model = load_models()
        feature_names = load_feature_names()

        st.write("Preprocessing input data...")
        data = preprocess_input(data, feature_names)
        
        if data is not None:
            st.write("Making predictions...")
            rf_predictions = rf_model.predict(data)
            gb_predictions = gb_model.predict(data)
            
            st.write("Random Forest Predictions:")
            st.write(rf_predictions)
            
            st.write("Gradient Boosting Predictions:")
            st.write(gb_predictions)

if __name__ == "__main__":
    main()


