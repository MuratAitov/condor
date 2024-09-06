from keys import *  
import boto3
import pandas as pd
import torch
def sliding_window(tensor, window_size=96, step_size=1):
    for i in range(0, tensor.size(0) - window_size + 1, step_size):
        yield tensor[i:i + window_size]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)
response = s3.list_objects_v2(Bucket='condor-data')
available_files = [item['Key'] for item in response.get('Contents', []) if item['Key'].startswith('COM')]
if available_files:
    selected_file = available_files[0]
    s3.download_file('condor-data', selected_file, selected_file)

    data = pd.read_csv(selected_file)
    data['timestamp'] = pd.to_datetime(data.iloc[:, 0])
    data = data.drop(columns=[data.columns[0]]).sort_values(by='timestamp').reset_index(drop=True)
    data = data.drop('ALI=F', axis=1)

    data_tensor = torch.tensor(data.drop(columns=['timestamp']).values, dtype=torch.float32).to(device)
    first_window = next(sliding_window(data_tensor))
    print(first_window )