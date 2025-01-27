import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os


DATA_PATH = "../data/cleaned_fight_data.csv"

def load_data(data_path):
    """Load the cleaned dataset."""
    return pd.read_csv(data_path)

def preprocess_percentages(df):
    """Convert percentage columns to numeric, handling invalid values."""
    if 'R_TD_pct' in df.columns and 'B_TD_pct' in df.columns:
      
        df['R_TD_pct'] = pd.to_numeric(df['R_TD_pct'].str.rstrip('%'), errors='coerce') / 100
        df['B_TD_pct'] = pd.to_numeric(df['B_TD_pct'].str.rstrip('%'), errors='coerce') / 100
    return df

def summarize_data(df):
    """Generate basic summary statistics."""
    print("\n--- Dataset Summary ---")
    print(df.info())
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nDuplicates:\n", df.duplicated().sum())
    print("\nDescriptive Statistics:\n", df.describe())

def plot_class_balance(df):
    """Visualize the class distribution of the target variable."""
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x='Winner', palette='viridis')
    plt.title('Class Balance of Winner')
    plt.xlabel('Winner (Red or Blue)')
    plt.ylabel('Count')
    plt.show()

def visualize_correlation(df):
    """Generate a heatmap for feature correlation."""
    plt.figure(figsize=(12, 10))
   
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=False, cmap="coolwarm", cbar=True)
    plt.title('Feature Correlation Heatmap')
    plt.show()

def compare_features(df, features):
    """Compare multiple features between Red and Blue fighters."""
    for feature in features:
        r_feature = f'R_{feature}'
        b_feature = f'B_{feature}'
        if r_feature in df.columns and b_feature in df.columns:
            if pd.api.types.is_numeric_dtype(df[r_feature]) and pd.api.types.is_numeric_dtype(df[b_feature]):
                plt.figure(figsize=(8, 4))
                sns.kdeplot(df[r_feature], label=f'Red {feature}', fill=True, color='red')
                sns.kdeplot(df[b_feature], label=f'Blue {feature}', fill=True, color='blue')
                plt.title(f'Red vs Blue {feature}')
                plt.xlabel(feature)
                plt.legend()
                plt.show()
            else:
                print(f"Skipping non-numeric feature: {feature}")

def feature_engineering(df):
    """Create new features to enhance the dataset."""
    print("\n--- Feature Engineering ---")
   
    if 'R_Reach_cms' in df.columns and 'B_Reach_cms' in df.columns:
        df['Reach_Advantage'] = df['R_Reach_cms'] - df['B_Reach_cms']
    else:
        print("Skipping Reach Advantage: Missing 'R_Reach_cms' or 'B_Reach_cms'.")

    if 'R_Height_cms' in df.columns and 'B_Height_cms' in df.columns:
        df['Height_Advantage'] = df['R_Height_cms'] - df['B_Height_cms']
    else:
        print("Skipping Height Advantage: Missing 'R_Height_cms' or 'B_Height_cms'.")
    
  
    if 'R_age' in df.columns and 'B_age' in df.columns:
        df['Age_Difference'] = df['R_age'] - df['B_age']

    if 'R_SIG_STR_Landed' in df.columns and 'R_SIG_STR_Attempted' in df.columns:
        df['R_Strike_Efficiency'] = df['R_SIG_STR_Landed'] / (df['R_SIG_STR_Attempted'] + 1)
    if 'B_SIG_STR_Landed' in df.columns and 'B_SIG_STR_Attempted' in df.columns:
        df['B_Strike_Efficiency'] = df['B_SIG_STR_Landed'] / (df['B_SIG_STR_Attempted'] + 1)

    def parse_control_time(column):
        return column.str.split(':').apply(lambda x: int(x[0]) * 60 + int(x[1]) if isinstance(x, list) and len(x) == 2 else 0)

    if 'R_CTRL' in df.columns and 'B_CTRL' in df.columns:
        df['R_Control_Time'] = parse_control_time(df['R_CTRL'])
        df['B_Control_Time'] = parse_control_time(df['B_CTRL'])
        df['Control_Time_Difference'] = df['R_Control_Time'] - df['B_Control_Time']

    print("Feature engineering completed.")
    return df

if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"Data not found at {DATA_PATH}. Please check the file path.")
    else:
      
        df = load_data(DATA_PATH)
        df = preprocess_percentages(df)

      
        summarize_data(df)
        plot_class_balance(df)
        visualize_correlation(df)

    
        compare_features(df, ['Height_cms', 'Reach_cms', 'SIG_STR_pct', 'TD_pct'])

      
        df = feature_engineering(df)

      
        enhanced_data_path = "../data/enhanced_fight_data.csv"
        df.to_csv(enhanced_data_path, index=False)
        print(f"Enhanced dataset saved to {enhanced_data_path}")

