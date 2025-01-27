import streamlit as st
import pandas as pd
import joblib

# Function to load models
@st.cache_resource
def load_models():
    rf_model = joblib.load('./models/random_forest_model.pkl')
    gb_model = joblib.load('./models/gradient_boosting_model.pkl')
    return rf_model, gb_model

# Function to preprocess input data
def preprocess_input(data):
    st.write("Preprocessing input data...")
    columns_to_split = ['R_TOTAL_STR.', 'B_TOTAL_STR.', 'R_TD', 'B_TD', 'R_HEAD', 'B_HEAD',
                        'R_BODY', 'B_BODY', 'R_LEG', 'B_LEG', 'R_DISTANCE', 'B_DISTANCE',
                        'R_CLINCH', 'B_CLINCH', 'R_GROUND', 'B_GROUND']
    for col in columns_to_split:
        if col in data.columns:
            data[f'{col}_Landed'] = data[col].str.split(' of ').str[0].astype(float)
            data[f'{col}_Attempted'] = data[col].str.split(' of ').str[1].astype(float)
            data.drop(columns=[col], inplace=True)

    time_columns = ['last_round_time', 'R_CTRL', 'B_CTRL']
    for col in time_columns:
        if col in data.columns:
            data[col] = data[col].replace('--', None)  # Replace '--' with NaN
            data[f'{col}_seconds'] = data[col].apply(
                lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if isinstance(x, str) else None
            )
            mean_seconds = data[f'{col}_seconds'].mean()
            data[f'{col}_seconds'] = data[f'{col}_seconds'].fillna(mean_seconds)
            data.drop(columns=[col], inplace=True)

    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    st.write("Input data preprocessing complete.")
    return data

# Main Streamlit app
def main():
    st.title("UFC Fight Outcome Predictor")
    st.write("Upload a CSV file containing fight data to predict outcomes using trained models.")

    # File upload section
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        try:
            input_data = pd.read_csv(uploaded_file)
            st.write("Uploaded File Preview:")
            st.write(input_data.head())

            # Load models
            rf_model, gb_model = load_models()

            # Preprocess the data
            preprocessed_data = preprocess_input(input_data)
            input_data_numeric = preprocessed_data.select_dtypes(include=['float64', 'int64'])

            # Make predictions
            st.write("Making predictions...")
            rf_prediction = rf_model.predict(input_data_numeric)
            gb_prediction = gb_model.predict(input_data_numeric)

            # Add predictions to the input data
            input_data['Random Forest Prediction'] = rf_prediction
            input_data['Gradient Boosting Prediction'] = gb_prediction

            # Display predictions
            st.write("Predictions:")
            st.write(input_data)

            # Option to download predictions
            csv = input_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
