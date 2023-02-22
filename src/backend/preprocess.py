import sys
import os

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
import json

# Load in the data from interim folder
train = load_data('data/interim/train.csv', index_col=0)

target = train['class_num']
train = train.drop(['class', 'class_num'], axis=1)

# Separate columns into numerical and categorical
cat_cols, num_cols = separate_categorical_numerical(train)

# Re-order df with cat_cols and num_cols for easier column tracking
train = pd.concat([train[cat_cols], train[num_cols]], axis=1)

column_list = train.columns.tolist()

with open('references/columns.txt', 'w') as file:
    for column in column_list:
        file.write(str(column) + '\n')

# Encode categorical values based on type of encoding
label_cols = ['protocol_type']
binary_cols = ['service', 'flag']

train_label, dict_label = encode_df(train[label_cols], 'label', train=True)
train_binary, dict_binary = encode_df(train[binary_cols], 'binary', train=True)

# Merge the three dictionaries of encoded mappings and store into json

mappings = {}

for d in (dict_label, dict_binary):
    mappings.update(d)

with open("references/encoding_mappings.json", 'w') as f:
    json.dump(mappings, f)

# Normalize numerical values by minmax scaling because the distributions
# are skewed and save scaler parameters for deployment
train_num = normalize_dataframe(train[num_cols], train[num_cols], train=True, save_scaler=True, save_directory="references/minmax_scaler_params.npy", scaler='minmax')

# Concatenate the encoded dataframes with the normalized and scaled dataframe
train_processed = pd.concat([train_label, train_binary, train_num], axis=1).reset_index(drop=True)

train_processed = pd.concat([train_processed, target], axis=1)

train_processed.to_csv("data/processed/train.csv")