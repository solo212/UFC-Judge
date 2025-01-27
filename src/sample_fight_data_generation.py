import pandas as pd

# Use the correct relative paths
data = pd.read_csv('../data/enhanced_fight_data.csv')  # Correct relative path
sample_data = data.sample(10)  # Select 10 random rows
sample_data.to_csv('../data/sample_fight_data.csv', index=False)  # Save in the parent `data` directory
print("Sample input file created at '../data/sample_fight_data.csv'")
