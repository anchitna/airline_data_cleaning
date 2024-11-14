from typing import List

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_dataframes_on_common_columns(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    how: str = 'inner'
) -> pd.DataFrame:
    """
    Merge two pandas DataFrames on their common columns.

    This function identifies the common columns between two DataFrames and merges them using the specified join type.

    Args:
        df1 (pd.DataFrame): The first DataFrame to merge.
        df2 (pd.DataFrame): The second DataFrame to merge.
        how (str, optional): Type of merge to be performed. Options include 'left', 'right', 'outer', and 'inner'. Defaults to 'inner'.

    Returns:
        pd.DataFrame: The merged DataFrame.

    Raises:
        ValueError: If no common columns are found between the DataFrames.
        ValueError: If the merge operation fails due to incompatible DataFrames or invalid parameters.
    """
    common_columns = list(set(df1.columns) & set(df2.columns))
    logger.debug(f"Common columns identified for merging: {common_columns}")

    if not common_columns:
        logger.error("No common columns found to merge the DataFrames.")
        raise ValueError("No common columns found to merge the DataFrames.")

    logger.info(f"Common columns identified for merging: {common_columns}")

    try:
        merged_df = pd.merge(df1, df2, on=common_columns, how=how)
        logger.info(f"DataFrames merged successfully using '{how}' join.")
    except Exception as e:
        logger.error(f"Failed to merge DataFrames: {e}")
        raise ValueError(f"Failed to merge DataFrames: {e}") from e

    return merged_df


def chunk_dataframe(df: pd.DataFrame, chunk_size: int = 100) -> List[pd.DataFrame]:
    """
    Split a pandas DataFrame into smaller chunks.

    This function divides the original DataFrame into a list of smaller DataFrames, each containing a specified number of rows.

    Args:
        df (pd.DataFrame): The original DataFrame to be split.
        chunk_size (int, optional): Number of rows per chunk. Defaults to 100.

    Returns:
        List[pd.DataFrame]: A list of DataFrame chunks.

    Raises:
        ValueError: If `chunk_size` is not a positive integer.
    """
    if chunk_size <= 0:
        logger.error("chunk_size must be a positive integer.")
        raise ValueError("chunk_size must be a positive integer.")

    logger.info(f"Splitting DataFrame into chunks of size {chunk_size}.")

    chunks = [df[i:i + chunk_size].copy() for i in range(0, df.shape[0], chunk_size)]
    logger.info(f"DataFrame split into {len(chunks)} chunks.")

    return chunks