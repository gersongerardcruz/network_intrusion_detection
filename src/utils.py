import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
import category_encoders as ce

def load_data(file_path, index_col=None, file_type='csv'):
    """
    Loads data into a pandas DataFrame.

    Parameters:
    file_path (str): The file path to the data.
    index_col (int, optional): The index to be used. Defaults to None. 
    file_type (str): The type of file to be loaded. 
                     Default is 'csv'. 
                     Other options include 'excel', 'json', 'hdf', 'parquet', etc.

    Returns:
    pandas DataFrame: The loaded data.

    """
    # Load the data into a pandas DataFrame
    if file_type == 'csv':
        df = pd.read_csv(file_path, index_col=index_col)
    elif file_type == 'excel':
        df = pd.read_excel(file_path, index_col=index_col)
    elif file_type == 'json':
        df = pd.read_json(file_path, index_col=index_col)
    elif file_type == 'hdf':
        df = pd.read_hdf(file_path, index_col=index_col)
    elif file_type == 'parquet':
        df = pd.read_parquet(file_path, index_col=index_col)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    return df

def compute_basic_stats(df):
    """
    This function takes a pandas dataframe column and returns the following statistics:
    1. Total missing values
    2. Percentage of missing values with respect to the entire dataframe
    3. Unique values present in the column
    4. Basic statistics (mean, median, mode, standard deviation) of the column
    5. 25th and 75th percentile of the column
    
    The function will handle both numerical and categorical columns, providing different statistics for each type.
    
    Parameters:
    col (pandas series): a single column from a pandas dataframe
    
    Returns:
    dict: a dictionary containing the statistics for the input column
    """
 
    stats = {}
    for col in df.columns:
        # If the column is numerical, calculate and return the statistics
        if df[col].dtype in ['float64', 'int64']:
            missing_values = df[col].isnull().sum()
            percent_missing = missing_values / len(df) * 100
            uniques = df[col].nunique()
            mean = df[col].mean()
            median = df[col].median()
            mode = df[col].mode().values[0]
            min = df[col].min()
            percentiles = df[col].quantile([0.25, 0.75])
            max = df[col].max()
            std = df[col].std()
            stats[col]={
                'missing_values': missing_values,
                'percent_missing': percent_missing,
                'uniques': uniques,
                'mean': mean,
                'median': median,
                'mode': mode,
                'min': min,
                '25th_percentile': percentiles[0.25],
                '75th_percentile': percentiles[0.75],
                'max': max,
                'std': std,
            }
        
        # If the column is categorical, calculate and return the mode
        else:
            missing_values = df[col].isnull().sum()
            percent_missing = missing_values / len(df) * 100
            uniques = df[col].nunique()
            mode = df[col].mode().values[0]
            stats[col]={
                'missing_values': missing_values,
                'percent_missing': percent_missing,
                'uniques': uniques,
                'mode': mode
            }
    
    return pd.DataFrame(stats)


def inspect_column(df, col_name):
    """
    Inspect a single column in a pandas DataFrame.
    
    Parameters:
    df (pandas DataFrame): The DataFrame to inspect.
    col_name (str): The name of the column to inspect.
    
    Returns:
    None
    """
    # Calculate the number of missing values
    missing_values = df[col_name].isna().sum()
    print(f"Number of missing values in {col_name}: {missing_values}")
    
    # Print the unique values in the column
    unique_values = df[col_name].nunique()
    print(f"Number of unique values in {col_name}: {unique_values}")
    print(f"Unique values in {col_name}: {df[col_name].unique()}")
    
    # Print basic statistics of the column
    print(f"Basic statistics of {col_name}:")
    print(df[col_name].describe())
    
    # Check if the column is quantitative or not
    if df[col_name].dtype in [np.int64, np.float64]:
        # Check for outliers using the IQR method
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = df[(df[col_name] < lower_bound) | (df[col_name] > upper_bound)]
        print(f"Number of outliers in {col_name}: {len(outliers)}")
        print(f"Outliers in {col_name}: {outliers}")
    else:
        # Print the mode of the column
        mode = df[col_name].mode().values[0]
        print(f"Mode of {col_name}: {mode}")

def separate_categorical_numerical(df):
    """
    Separate the categorical and numerical columns of a pandas DataFrame.
    
    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    
    Returns:
    tuple: A tuple containing two pandas DataFrames, one for the categorical columns and one for the numerical columns.
    """
    # Select the columns with data type 'object', which are assumed to be categorical
    categorical_cols = df.select_dtypes(include=['object']).columns
    # Select the columns with data types other than 'object', which are assumed to be numerical
    numerical_cols = df.select_dtypes(exclude=['object']).columns
    # Return two separate DataFrames, one for the categorical columns and one for the numerical columns

    return categorical_cols, numerical_cols

def plot_distributions(df, segment_col=None, figsize=(20, 20)):
    """
    This function plots the distributions of all numerical columns in a dataframe grouped by the segment column.
    It creates subplots with two graphs per row.
    
    Parameters:
        df (pandas dataframe): The dataframe to plot.
        segment_col (optional, str): The name of the column to group the distributions by. Defaults to None. 
        figsize (tuple, optional): The size of the plot. Default is (20, 10).
    
    Returns:
        None
    """

    num_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]
    n_cols = 2
    n_rows = math.ceil(len(num_cols) / n_cols)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.tight_layout(pad=3.0)
    
    if segment_col is not None:
        for i, num_col in enumerate(num_cols):
            print(f"{num_col}")
            df[num_col].hist(by=df[segment_col])
            plt.tight_layout()
            plt.show()
    
    else:
        for i, num_col in enumerate(num_cols):
            r = i // n_cols
            c = i % n_cols
            ax[r, c].hist(df[num_col])
            ax[r, c].set_title(num_col)

def get_correlation(df, figsize=(20, 20)):
    """
    This function takes a dataframe returns the correlation heatmap of that dataframe.
    
    Parameters:
    df (pandas.DataFrame): The input dataframe.
    figsize (tuple): The size of the correlation heatmap
    
    Returns:
    None
    """
    
    # Calculate the correlation between the input column and all other columns
    corr = df.corr()
    
    # Plot the correlation matrix as a heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()


def normalize_dataframe(df, train_df, train: bool, save_scaler: bool, save_directory, scaler='standard'):
    """
    Normalize all columns in a pandas dataframe using either StandardScaler or MinMaxScaler.
    
    Parameters:
    df (pandas dataframe): The data to be normalized.
    train_df (pandas dataframe): The training data for fitting the scaler.
    train (bool): Whether the data is from the training set or not. If True, the data is from the training set.
    save_scaler (bool): Whether to save the scaler parameters. 
    scaler (str, optional): The type of scaler to use. Must be either 'standard' or 'minmax'. Default is 'standard'.
    
    Returns:
    pandas.DataFrame, pandas.DataFrame: The normalized training data and test data as dataframes.
    """

    if scaler == 'standard':
        # Use StandardScaler to normalize the data
        scaler = StandardScaler()
    elif scaler == 'minmax':
        # Use MinMaxScaler to normalize the data
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Invalid scaler type: {scaler}. Must be either 'standard' or 'minmax'.")
    
    # Fit the scaler to the training data
    scaler.fit(train_df)

    # Save the scaler parameters to file
    if save_scaler == True:
        if scaler == 'standard':
            np.save(save_directory, [scaler.mean_, scaler.var_])
        else:
            np.save(save_directory, [scaler.min_, scaler.scale_])

    if train: 
        # Transform the training data
        normalized_train_data = scaler.transform(df)
        
        # Return the normalized data as a dataframe
        normalized_df_train = pd.DataFrame(normalized_train_data, columns=df.columns)

        return normalized_df_train
    
    else:
        # Transform the test data using the same parameters
        normalized_test_data = scaler.transform(df)
        
        # Return the normalized test data as a dataframe
        normalized_df_test = pd.DataFrame(normalized_test_data, columns=df.columns)

        return normalized_df_test


def encode_df(df, encoder_type, train: bool):
    """
    Perform encoding on a Pandas dataframe.
    
    Parameters:
    df (pandas.DataFrame): The dataframe to be encoded.
    encoder_type (str): The type of encoding to perform. Must be one of 'onehot', 'ordinal', or 'label'.
    train (bool): Whether the data is from the training set or not. If True, the data is from the training set.
    
    Returns:
    pandas.DataFrame: The encoded dataframe.
    dict: A dictionary containing a mapping of the original value and its encoded value for each column. Values encoded
    as strings for better json interaction. 
    """

    columns = df.columns
    print(columns)
    df_encoded = df.copy()
    
    encoders = {}
    mappings = {}

    for column in columns:
            if encoder_type == 'onehot':
                encoder = OneHotEncoder(handle_unknown='ignore')
                if train:
                    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoder.fit_transform(df_encoded[[column]]).toarray(), columns=encoder.get_feature_names_out([column]))], axis=1)
                else:
                    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoder.fit_transform(df_encoded[[column]]).toarray(), columns=encoder.get_feature_names_out([column]))], axis=1)
                df_encoded = df_encoded.drop(column, axis=1)

                # Encode the data and save the mapping to a dictionary
                mappings[column] = dict(zip(range(len(encoder.categories_[0])), encoder.categories_[0]))

                print(f"Successfully performed {encoder_type} encoding for {column} column!")

            elif encoder_type == 'ordinal':
                encoder = OrdinalEncoder()
                if train:
                    encoders[column] = encoder
                    df_encoded[column] = encoder.fit_transform(df_encoded[[column]])
                else:
                    encoders[column] = encoder
                    df_encoded[column] = encoder.fit_transform(df_encoded[[column]])

                # Encode the data and save the mapping to a dictionary
                mappings[column] = {label: str(idx) for idx, label in enumerate(encoder.categories_[0])}

                print(f"Successfully performed {encoder_type} encoding for {column} column!")

            elif encoder_type == 'label':
                encoder = LabelEncoder()
                if train:
                    encoders[column] = encoder
                    df_encoded[column] = encoder.fit_transform(df_encoded[column])
                else:
                    encoders[column] = encoder
                    df_encoded[column] = encoder.fit_transform(df_encoded[column])
                
                # Encode the data and save the mapping to a dictionary
                mappings[column] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                mappings[column] = {k: str(v) for k, v in mappings[column].items()}

                print(f"Successfully performed {encoder_type} encoding for {column} column!")

            elif encoder_type == 'binary':
                encoder = ce.BinaryEncoder()
                if train:
                    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoder.fit_transform(df_encoded[[column]]), columns=encoder.get_feature_names_out())], axis=1)
                else:
                    df_encoded = pd.concat([df_encoded, pd.DataFrame(encoder.fit_transform(df_encoded[[column]]), columns=encoder.get_feature_names_out())], axis=1)
                df_encoded = df_encoded.drop(column, axis=1)

                # Encode the data and save the mapping to a dictionary
                mappings[column] = dict(zip(range(len(encoder.get_feature_names_out())), encoder.get_feature_names_out()))

                print(f"Successfully performed {encoder_type} encoding for {column} column!")

            else:
                raise ValueError("Encoder type must be one of 'onehot', 'ordinal', 'label', or 'binary'.")

    return df_encoded, mappings

def remove_id_column(df):
    """
    Removes the ID column from a pandas dataframe.
    Takes into account different spellings of "ID" and "Unnamed: 0".

    Parameters:
        df (pandas.DataFrame): The input dataframe.

    Returns:
        pandas.DataFrame: The dataframe with the ID column removed.
    """
    id_columns = ['ID', 'Id', 'id', 'Unnamed: 0']
    for col in id_columns:
        if col in df.columns:
            return df.drop(columns=[col])
    return df

def match_test_set_types(h2o_test_set, json_path: str):
    # Load the JSON file
    with open(json_path) as f:
        column_types = json.load(f)

    # Match the column types of the test set to the specified types in the JSON file
    for column_name, column_type in column_types.items():
        if column_name in h2o_test_set.columns:
            if column_types[column_name] != h2o_test_set.types[column_name]:
                if column_type == 'int':
                    h2o_test_set[column_name] = h2o_test_set[column_name].asfactor()
                elif column_type == 'real':
                    h2o_test_set[column_name] = h2o_test_set[column_name].asnumeric()
                elif column_type == 'enum':
                    h2o_test_set[column_name] = h2o_test_set[column_name].asfactor()
                elif column_type == 'str':
                    h2o_test_set[column_name] = h2o_test_set[column_name].ascharacter()

    return h2o_test_set