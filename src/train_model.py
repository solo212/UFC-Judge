import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from imblearn.over_sampling import SMOTE  
import joblib 



def save_feature_names(feature_names, path="../models/feature_names.txt"):
    with open(path, "w") as f:
        for name in feature_names:
            f.write(f"{name}\n")
    print(f"Feature names saved to {path}.")



def load_and_preprocess_data(filepath):
    print("Loading dataset...")
    df = pd.read_csv(filepath)


    print("Mapping 'Winner' column to binary values...")
    df['Winner'] = df.apply(lambda row: 0 if row['Winner'] == row['R_fighter'] else (1 if row['Winner'] == row['B_fighter'] else None), axis=1)

    
    if df['Winner'].isnull().any():
        print("Filling invalid or missing 'Winner' values with 0 (Red) as fallback.")
        df['Winner'].fillna(0, inplace=True)

  
    columns_to_split = ['R_TOTAL_STR.', 'B_TOTAL_STR.', 'R_TD', 'B_TD', 'R_HEAD', 'B_HEAD',
                        'R_BODY', 'B_BODY', 'R_LEG', 'B_LEG', 'R_DISTANCE', 'B_DISTANCE',
                        'R_CLINCH', 'B_CLINCH', 'R_GROUND', 'B_GROUND']
    for col in columns_to_split:
        print(f"Processing column: {col}.")
        df[f'{col}_Landed'] = df[col].str.split(' of ').str[0].astype(float)
        df[f'{col}_Attempted'] = df[col].str.split(' of ').str[1].astype(float)
        df.drop(columns=[col], inplace=True)

 
    time_columns = ['last_round_time', 'R_CTRL', 'B_CTRL']
    for col in time_columns:
        print(f"Handling missing values in '{col}'...")
        df[col] = df[col].replace('--', None)  
        df[f'{col}_seconds'] = df[col].apply(
            lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if isinstance(x, str) else None
        )
       
        mean_seconds = df[f'{col}_seconds'].mean()
        df[f'{col}_seconds'] = df[f'{col}_seconds'].fillna(mean_seconds)
        print(f"Mean seconds for {col}: {mean_seconds:.2f}")
        df.drop(columns=[col], inplace=True)

  
    print("Filling missing numerical values with column means...")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

    print(f"Data preprocessing complete. Shape: {df.shape}")
    return df


def hyperparameter_tuning(X_train, y_train):
    print("Starting hyperparameter tuning...")

 
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


def train_and_evaluate_models(df):
    print("Splitting dataset into train and test...")

  
    X = df.drop(columns=['Winner']).select_dtypes(include=['float64', 'int64'])
    y = df['Winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

  
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

 
    rf_model, gb_model = hyperparameter_tuning(X_train, y_train)

  
    print("\nEvaluating Random Forest...")
    rf_preds = rf_model.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f}, F1-Score: {f1_score(y_test, rf_preds):.4f}")
    print(classification_report(y_test, rf_preds))


    joblib.dump(rf_model, '../models/random_forest_model.pkl')
    print("Random Forest model saved.")

   
    print("\nEvaluating Gradient Boosting...")
    gb_preds = gb_model.predict(X_test)
    print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, gb_preds):.4f}, F1-Score: {f1_score(y_test, gb_preds):.4f}")
    print(classification_report(y_test, gb_preds))

 
    joblib.dump(gb_model, '../models/gradient_boosting_model.pkl')
    print("Gradient Boosting model saved.")

 
    print("\nFeature Importances (Random Forest):", rf_model.feature_importances_)
    print("Feature Importances (Gradient Boosting):", gb_model.feature_importances_)

   
    feature_names = X_train.columns.tolist()
    save_feature_names(feature_names)



def main():
    DATA_PATH = '../data/enhanced_fight_data.csv'
    df = load_and_preprocess_data(DATA_PATH)

 
    print("\nClass distribution in 'Winner':")
    print(df['Winner'].value_counts(normalize=True))
    print()

    train_and_evaluate_models(df)

if __name__ == "__main__":
    main()


