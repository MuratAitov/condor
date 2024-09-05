from keys import *  # Import AWS credentials
import boto3
import pandas as pd
import torch

# Function to create sliding window from tensor
def sliding_window(tensor, window_size=96, step_size=1):
    length = tensor.size(0)
    for i in range(0, length - window_size + 1, step_size):
        yield tensor[i:i + window_size]

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Connect to S3 using AWS credentials
s3 = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name
)

bucket_name = 'condor-data'
response = s3.list_objects_v2(Bucket=bucket_name)

# Check if there are files in the S3 bucket
if 'Contents' in response:
    print("Files in bucket:")
    available_files = [item['Key'] for item in response['Contents']]
    for i, file in enumerate(available_files):
        print(f"{i}: {file}")  

    file_index = int(input(f"Select a file by its index (0-{len(available_files) - 1}): "))
    selected_file = available_files[file_index]
    print(f"Selected file: {selected_file}")
else:
    print("No files found in the bucket.")
    exit()

# Download the selected file from S3
s3.download_file(bucket_name, selected_file, selected_file)




# Convert the first column to datetime and remove the original timestamp column
data = pd.read_csv(selected_file)
timestamp_col = data.columns[0]
data['timestamp'] = pd.to_datetime(data[timestamp_col])
data = data.drop(columns=[timestamp_col])
data = data.sort_values(by='timestamp')
data = data.reset_index(drop=True)

# Drop a column 'ALI=F'
nan_columns = data.columns[data.isna().any()].tolist()
print(f"Columns with NaN: {nan_columns}")
data = data.drop('ALI=F', axis=1)

# Convert the DataFrame to a tensor 
data_tensor = torch.tensor(data.drop(columns=['timestamp']).values, dtype=torch.float32).to(device)
print(f"Data tensor shape: {data_tensor.shape}")

# Generate the first sliding window using the sliding_window function
generator = sliding_window(data_tensor)
first_window = next(generator)
print(f"First window shape: {first_window.shape}")

# Сheck if the tensor is on CUDAЖ
if first_window.is_cuda:
    print("Tensor is on the GPU")
else:
    print("Tensor is on the CPU")
    