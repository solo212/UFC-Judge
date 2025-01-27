import pandas as pd

data = pd.read_csv('../data/enhanced_fight_data.csv')  
sample_data = data.sample(10)  
sample_data.to_csv('../data/sample_fight_data.csv', index=False)  
print("Sample input file created at '../data/sample_fight_data.csv'")
