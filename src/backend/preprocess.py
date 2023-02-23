import sys
import os

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import *
import json
import argparse
import h2o

def main():

    # Initiate H2O cluster
    h2o.init()

    if args.train: 
        # Load in the data from input path
        train = load_data(args.input_path)
        train = remove_id_column(train)

        target = train[args.target_column]

        train = train.drop(args.target_column, axis=1)

        # Separate columns into numerical and categorical
        cat_cols, num_cols = separate_categorical_numerical(train)

        # Re-order df with cat_cols and num_cols for easier column tracking
        train = pd.concat([train[cat_cols], train[num_cols]], axis=1)

        column_list = train.columns.tolist()

        with open('references/columns.txt', 'w') as file:
            for column in column_list:
                file.write(str(column) + '\n')

        # Encode categorical values based on type of encoding
        label_cols = args.label_encoding
        binary_cols = args.binary_encoding

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
        train_processed = remove_id_column(train_processed)

        train_processed.to_csv(args.output_path)

        # Save processed data into h2o frame to save dtypes for checking during prediction
        train_processed_h2o = h2o.H2OFrame(train_processed)

        with open('references/train_processed_column_types.json', 'w') as f:
            json.dump(train_processed_h2o.types, f)
    
    else: 
        # Load in the data from input path
        test = load_data(args.input_path)
        test = remove_id_column(test)

        # Separate columns into numerical and categorical
        cat_cols, num_cols = separate_categorical_numerical(test)

        # Re-order df with cat_cols and num_cols for easier column tracking
        test = pd.concat([test[cat_cols], test[num_cols]], axis=1)

        # Encode categorical values based on type of encoding
        label_cols = args.label_encoding
        binary_cols = args.binary_encoding

        test_label, _ = encode_df(test[label_cols], 'label', train=False)
        test_binary, _  = encode_df(test[binary_cols], 'binary', train=False)

        # Normalize numerical values by minmax scaling because the distributions
        # are skewed and save scaler parameters for deployment
        test_num = normalize_dataframe(test[num_cols], test[num_cols], train=False, save_scaler=True, save_directory="references/minmax_scaler_params.npy", scaler='minmax')

        # Concatenate the encoded dataframes with the normalized and scaled dataframe
        test_processed = pd.concat([test_label, test_binary, test_num], axis=1).reset_index(drop=True)

        test_processed = remove_id_column(test_processed)

        test_processed.to_csv(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="path to input file")
    parser.add_argument("--output_path", help="path to output file")
    parser.add_argument("--target_column", type=str, help="Name of target column")
    parser.add_argument("--label_encoding", nargs='+', help="List of columns for label encoding")
    parser.add_argument("--binary_encoding", nargs='+', help="List of columns for binary encoding")
    parser.add_argument("--train", action='store_true', help="whether to process a train set with label or test set without")
    parser.add_argument("--no-train", dest='train', action='store_false', help="set to disable train")
    args = parser.parse_args()

    main()