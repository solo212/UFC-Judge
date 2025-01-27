import os
import pandas as pd
import numpy as np


RAW_DATA_PATH = "../data/raw_total_fight_data.csv"
CLEANED_DATA_PATH = "../data/cleaned_fight_data.csv"

def clean_fight_data(input_path, output_path):
  
    raw_data = pd.read_csv(input_path, delimiter=';')

   
    raw_data["R_SIG_STR_pct"] = pd.to_numeric(raw_data["R_SIG_STR_pct"].str.rstrip('%'), errors='coerce') / 100
    raw_data["B_SIG_STR_pct"] = pd.to_numeric(raw_data["B_SIG_STR_pct"].str.rstrip('%'), errors='coerce') / 100

    
    def split_landed_attempted(column):
        landed = column.str.split(' of ').str[0].replace('---', np.nan).astype(float)
        attempted = column.str.split(' of ').str[1].replace('---', np.nan).astype(float)
        return landed, attempted

    raw_data["R_SIG_STR_Landed"], raw_data["R_SIG_STR_Attempted"] = split_landed_attempted(raw_data["R_SIG_STR."])
    raw_data["B_SIG_STR_Landed"], raw_data["B_SIG_STR_Attempted"] = split_landed_attempted(raw_data["B_SIG_STR."])

 
    raw_data["date"] = pd.to_datetime(raw_data["date"], errors='coerce')

   
    raw_data = raw_data.drop(["Format", "R_SIG_STR.", "B_SIG_STR."], axis=1)

   
    raw_data.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

if __name__ == "__main__":
  
    if not os.path.exists("../data"):
        print("Data directory does not exist. Please download the dataset first.")
    else:
        clean_fight_data(RAW_DATA_PATH, CLEANED_DATA_PATH)
