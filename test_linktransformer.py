from sentence_transformers import SentenceTransformer
import linktransformer as lt
import pandas as pd
# Example usage of lm_merge_df

data2 = {
    "CompanyName": ["TechCorp", "InfoTech Solutions", "GlobalSoft Inc", "DataTech Co", "SoftSys Ltd", "TechCorp"],
    "Industry": ["Technology", "Technology", "Software", "Data Analytics", "Software", "Technology"],
    "Founded_Year": [2005, 1998, 2010, 2012, 2003, 2005]
}

# Create a DataFrame from the data
df2 = pd.DataFrame(data2)

data1 = {
    "CompanyName": ["Tech Corporation", "InfoTech Soln", "GlobalSoft Incorporated", "DataTech Corporation", 
                    "SoftSys Limited", "TechCorp", "AlphaSoft Systems"],
    "Revenue (Millions USD)": [5000, 4500, 3000, 2500, 4000, 5500, 3800],
    "Num_Employees": [10000, 8500, 6000, 5000, 7500, 12000, 7000],
    "Country": ["USA", "Canada", "India", "Germany", "UK", "USA", "Spain"]
}

# Create a DataFrame from the data
df1 = pd.DataFrame(data1)

df_lm_matched = lt.merge(df1, df2, merge_type='1:m', on="CompanyName", model="all-MiniLM-L6-v2",
left_on=None, right_on=None)

print(df_lm_matched)