import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """ Load messages and category data from CSV files, merge data and return in
        a dataframe

    Args:
    messages_filepath: str. Filepath to messages CSV file.
    categories_filepath: str. Filepath to categories CSV file.

    Returns:
    df: DataFrame. A DataFrame of merged message and category data
    """
    # Load messages and categories datasets from CSV files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge messages and categories datasets on the common id field
    df = pd.merge(messages, categories, left_on='id', right_on='id', how='outer')

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Clean DataFrame df and return in a format ready to be inserted to database

    Args:
    df: DataFrame.  A DataFrame of message and category data.

    Returns:
    df: DataFrame. A DataFrame of cleaned and de-duplicated data.
    """

    # Split values in categories column and expand strings into DataFrame columns
    categories = df.categories.str.split(';', expand=True)

    # Select first row of categories DataFrame and assign subset of string as
    # column names for categories DataFrame
    row = categories.iloc[0,:]
    categories.columns = row.map(lambda x: x[:-2])

    # Iterate through categories columns and convert category values from
    # strings to integers 0 and 1
    for col in categories.columns:
        # Set each value to be the last character of the string
        categories[col] = categories[col].str[-1]

        # Convert column from string to numeric
        categories[col] = pd.to_numeric(categories[col])

    # Replace 2s in the 'related' column with 1s
    categories.loc[categories.related==2, 'related'] = 1

    # Drop categories column from df and replace with categories DataFrame
    df.drop(columns=['categories'], axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    return df


def save_data(df: pd.DataFrame, database_filename: str):
    """ Save clean DataFrame df to a SQLite database

    Args:
    df: DataFrame. DataFrame of cleaned data to be saved to a Database.
    database_filename: str. File path for the SQLite Database

    Returns:
    None
    """

    # Create engine instance
    engine = create_engine(f'sqlite:///{database_filename}')

    # Define table name and save df to SQLite database
    db_table_name = 'messages'
    df.to_sql(db_table_name, engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
