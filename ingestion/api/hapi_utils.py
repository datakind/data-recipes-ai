import sys

import pandas as pd


def filter_hapi_df(df, admin0_code_field):
    """
    Filter a pandas DataFrame by removing columns where all values are null and removing rows where any value is null.
    Hack to get around the fact HDX mixes total values in with disaggregated values in the API

    Args:
        df (pandas.DataFrame): The DataFrame to be filtered.
        admin0_code_field (str): The name of the column containing the admin0 code.

    Returns:
        pandas.DataFrame: The filtered DataFrame.
    """
    df_orig = df.copy()

    if df.shape[0] == 0:
        return df_orig

    dfs = []
    if admin0_code_field in df.columns:
        for country in df[admin0_code_field].unique():
            df2 = df.copy()
            df2 = df2[df2[admin0_code_field] == country]

            # Remove any columns where all null
            df2 = df2.dropna(axis=1, how="all")

            # Remove any rows where one of the values is null
            df2 = df2.dropna(axis=0, how="any")

            dfs.append(df.iloc[df2.index])

        df = pd.concat(dfs)

    return df


def post_process_data(df, standard_names):
    """
    Post-processes the data by filtering and renaming columns.

    Args:
        df (pandas.DataFrame): The DataFrame to be post-processed.

    Returns:
        pandas.DataFrame: The post-processed DataFrame.
    """
    # aggregate and disaggregated data in the same tables, where the hierarchy differs by country
    df = filter_hapi_df(df, standard_names["admin0_code_field"])

    # Add a flag to indicate latest dataset by HDX ID, useful for LLM queries
    if "dataset_hdx_stub" in df.columns and "reference_period_start" in df.columns:
        df["latest"] = 0
        df["reference_period_start"] = pd.to_datetime(df["reference_period_start"])
        df["latest"] = df.groupby("dataset_hdx_stub")[
            "reference_period_start"
        ].transform(lambda x: x == x.max())

    return df
