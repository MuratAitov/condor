from keys import *
import boto3
import pandas as pd
import torch

def sliding_window(tensor, window_size=96, step_size=1):
    for i in range(0, tensor.size(0) - window_size + 1, step_size):
        yield tensor[i:i + window_size]
        
def check_and_report_nan(df):
    missing = df.isna().sum()
    
    if missing.any():
        print("Columns with NaN values:")
        print(missing[missing > 0])
        
        print("\nExamples of rows with NaN values:")
        nan_rows = df[df.isna().any(axis=1)]
        print(nan_rows)
        for col in missing[missing > 0].index:
            print(f"\nNaN found in column '{col}':")
            nan_in_col = df[df[col].isna()]
            for idx, row in nan_in_col.iterrows():
                print(f"Time: {row['Timestamp']}, Row index: {idx}")
        
        df.dropna(inplace=True)
        print("\nRows with NaN values have been removed.")
    else:
        print("No NaN values found.")




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

bucket_name = 'condor-data'
file1 = 'COM_all_data_2024-07-09.csv'
file2 = 'indicators_all_data_2024-07-09.csv'

def download_file(bucket, key):
    s3.download_file(bucket, key, key)

download_file(bucket_name, file1)
download_file(bucket_name, file2)

def prepare_com_data(filename):
    df = pd.read_csv(filename)
    df.rename(columns={df.columns[0]: 'Timestamp'}, inplace=True)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.floor('T')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df

def prepare_indicators_data(filename):
    df = pd.read_csv(filename)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.floor('T')
    df = df.sort_values('Timestamp').reset_index(drop=True)
    return df

com_df = prepare_com_data(file1)
indicators_df = prepare_indicators_data(file2)
merged_df = pd.merge(com_df, indicators_df, on='Timestamp', how='inner')

if 'ALI=F' in merged_df.columns:
    merged_df.drop('ALI=F', axis=1, inplace=True)

missing = merged_df.isna().sum()
if missing.any():
    check_and_report_nan(merged_df)
    
    # merged_df.dropna(inplace=True)
    # print("\nString with NaN have been deleted.")

if 'spx_close' not in merged_df.columns:
    print("Error: 'spx_close' column not found.")
    exit()


merged_df = merged_df[['spx_close'] + [col for col in merged_df.columns if col not in ['Timestamp', 'spx_close']]]

data_tensor = torch.tensor(merged_df.values, dtype=torch.float32).to(device)
window_generator = sliding_window(data_tensor)
spx_tensor = data_tensor[:, 0]  
spx_window_generator = sliding_window(spx_tensor.unsqueeze(1))

first_window = next(window_generator)
first_spx_window = next(spx_window_generator)

print("Shape of window:", first_window.shape)
print("Shape of SPX values in same window:", first_window[0].shape)
print("The first 96 SPX values:", first_spx_window.shape)       

print(spx_tensor.shape)