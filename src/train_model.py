import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE  # For oversampling
import joblib  # For saving and loading models


# Save the feature names
def save_feature_names(feature_names, path="../models/feature_names.txt"):
    with open(path, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print(f"Feature names saved to {path}.")


# Load and preprocess the data
def load_and_preprocess_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    # Map 'Winner' column to binary values (0 for Red, 1 for Blue)
    print("Mapping 'Winner' column to binary values...")
    df['Winner'] = df.apply(lambda row: 0 if row['Winner'] == row['R_fighter'] else (1 if row['Winner'] == row['B_fighter'] else None), axis=1)

    # Handle any remaining invalid or missing Winner values
    if df['Winner'].isnull().any():
        print("Filling invalid or missing 'Winner' values with 0 (Red) as fallback.")
        df['Winner'].fillna(0, inplace=True)

    # Process columns with "x of y" format
    columns_to_split = ['R_TOTAL_STR.', 'B_TOTAL_STR.', 'R_TD', 'B_TD', 'R_HEAD', 'B_HEAD',
                        'R_BODY', 'B_BODY', 'R_LEG', 'B_LEG', 'R_DISTANCE', 'B_DISTANCE',
                        'R_CLINCH', 'B_CLINCH', 'R_GROUND', 'B_GROUND']
    for col in columns_to_split:
        print(f"Processing column: {col}.")
        df[f'{col}_Landed'] = df[col].str.split(' of ').str[0].astype(float)
        df[f'{col}_Attempted'] = df[col].str.split(' of ').str[1].astype(float)
        df.drop(columns=[col], inplace=True)

    # Replace '--' with the mean seconds
    time_columns = ['last_round_time', 'R_CTRL', 'B_CTRL']
    for col in time_columns:
        print(f"Handling missing values in '{col}'...")
        df[col] = df[col].replace('--', None)  # Replace '--' with NaN for processing
        df[f'{col}_seconds'] = df[col].apply(
            lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if isinstance(x, str) else None
        )
        # Fill missing values with the mean of valid seconds
        mean_seconds = df[f'{col}_seconds'].mean()
        df[f'{col}_seconds'] = df[f'{col}_seconds'].fillna(mean_seconds)
        print(f"Mean seconds for {col}: {mean_seconds:.2f}")
        df.drop(columns=[col], inplace=True)

    # Fill missing numerical values with column means
    print("Filling missing numerical values with column means...")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    print(f"Data preprocessing complete. Shape: {df.shape}")
    return df

# Hyperparameter tuning for models
def hyperparameter_tuning(X_train, y_train):
    print("Starting hyperparameter tuning...")

    # Random Forest Hyperparameter Grid
    rf_param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
    rf_search = RandomizedSearchCV(rf_model, rf_param_grid, cv=3, scoring='f1', n_iter=50, random_state=42, n_jobs=-1)
    rf_search.fit(X_train, y_train)

    print(f"Best Random Forest Params: {rf_search.best_params_}")
    print(f"Best Random Forest Score: {rf_search.best_score_}")

    # Gradient Boosting Hyperparameter Grid
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_search = RandomizedSearchCV(gb_model, gb_param_grid, cv=3, scoring='f1', n_iter=50, random_state=42, n_jobs=-1)
    gb_search.fit(X_train, y_train)

    print(f"Best Gradient Boosting Params: {gb_search.best_params_}")
    print(f"Best Gradient Boosting Score: {gb_search.best_score_}")

    return rf_search.best_estimator_, gb_search.best_estimator_

# Train and evaluate models
def train_and_evaluate_models(df):
    print("Splitting dataset into train and test...")

    # Drop non-numeric columns
    X = df.drop(columns=['Winner']).select_dtypes(include=['float64', 'int64'])
    y = df['Winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Apply SMOTE for oversampling
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Hyperparameter tuning
    rf_model, gb_model = hyperparameter_tuning(X_train, y_train)

    # Evaluate Random Forest
    print("\nEvaluating Random Forest...")
    rf_preds = rf_model.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}, F1-Score: {f1_score(y_test, rf_preds):.4f}")
    print(classification_report(y_test, rf_preds))

    # Save the Random Forest model
    joblib.dump(rf_model, '../models/random_forest_model.pkl')
    print("Random Forest model saved.")

    # Evaluate Gradient Boosting
    print("\nEvaluating Gradient Boosting...")
    gb_preds = gb_model.predict(X_test)
    print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_preds):.4f}, F1-Score: {f1_score(y_test, gb_preds):.4f}")
    print(classification_report(y_test, gb_preds))

    # Save the Gradient Boosting model
    joblib.dump(gb_model, '../models/gradient_boosting_model.pkl')
    print("Gradient Boosting model saved.")

    # Display feature importances
    print("\nFeature Importances (Random Forest):", rf_model.feature_importances_)
    print("Feature Importances (Gradient Boosting):", gb_model.feature_importances_)

    # Get the feature names from X_train
    feature_names = X_train.columns.tolist()
    save_feature_names(feature_names)


# Main script
def main():
    DATA_PATH = '../data/enhanced_fight_data.csv'
    df = load_and_preprocess_data(DATA_PATH)

    # Add the class distribution check here
    print("\nClass distribution in 'Winner':")
    print(df['Winner'].value_counts(normalize=True))
    print()

    train_and_evaluate_models(df)

if __name__ == "__main__":
    main()


