import pandas as pd
import joblib 


def load_models():
    print("Loading models...")
    rf_model = joblib.load('../models/random_forest_model.pkl')
    gb_model = joblib.load('../models/gradient_boosting_model.pkl')
    print("Models loaded successfully.")
    return rf_model, gb_model


def preprocess_input(data):
    print("Preprocessing input data...")

   
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
            print(f"Handling missing values in '{col}'...")
            data[col] = data[col].replace('--', None)
            data[f'{col}_seconds'] = data[col].apply(
                lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if isinstance(x, str) else None
            )
            mean_seconds = data[f'{col}_seconds'].mean()
            data[f'{col}_seconds'] = data[f'{col}_seconds'].fillna(mean_seconds)
            data.drop(columns=[col], inplace=True)

   
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    print("Input data preprocessing complete.")
    return data


def predict(models, input_data):
    print("Making predictions...")
    rf_model, gb_model = models
    rf_prediction = rf_model.predict(input_data)
    gb_prediction = gb_model.predict(input_data)
    return rf_prediction, gb_prediction


def main():
   
    models = load_models()


   
    with open("../models/feature_names.txt", "r") as f:
        feature_names = [line.strip() for line in f]

  
    input_data = pd.read_csv('../data/sample_fight_data.csv')  
    preprocessed_data = preprocess_input(input_data)

   
    input_data_numeric = preprocessed_data.select_dtypes(include=['float64', 'int64'])
    input_data_numeric = input_data_numeric.reindex(columns=feature_names, fill_value=0)

    rf_prediction, gb_prediction = predict(models, input_data_numeric)

 
    input_data['Random Forest Prediction'] = rf_prediction
    input_data['Gradient Boosting Prediction'] = gb_prediction


    output_path = '../output/predictions.csv'
    input_data.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

   
    print("\nSample Predictions:")
    print(input_data.head())

if __name__ == "__main__":
    main()


