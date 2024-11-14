from typing import List
from langchain.agents import AgentType
from app.models.column_name_output import ColumnNamesOutput
from app.services.agent_service import get_agent
from app.services.handle_data_inconsistencies import handle_inconsistencies
from app.services.handle_missing_negative_values import handle_missing_values
from app.tools.column_name_correction import correct_columns_tool
from app.utils.dataframe_utils import merge_dataframes_on_common_columns
from app.utils.llm_helper import LLM

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CleanData:
    """
    A service class for cleaning flight booking data.

    This class handles the process of reading raw data, correcting column names,
    merging datasets, handling missing and inconsistent values, and creating a
    PandasAI agent for further data analysis.
    """

    @staticmethod
    def clean_data():
        """
        Clean flight booking data and initialize a PandasAI agent.

        The cleaning process includes:
            1. Reading raw CSV files for bookings and airline mappings.
            2. Correcting and humanizing column names using an LLM-powered agent.
            3. Merging the bookings and mappings dataframes on common columns.
            4. Handling missing values and data inconsistencies.
            5. Creating a PandasAI agent for interacting with the cleaned data.

        Returns:
            create_pandas_dataframe_agent: An initialized PandasAI agent trained on the cleaned booking details dataframe.

        Raises:
            FileNotFoundError: If any of the required CSV files are not found.
            AssertionError: If there is a mismatch in column counts after renaming.
            Exception: If there is an error during the merging or data cleaning process.
        """
        try:
            # Step 1: Read raw CSV files
            df_bookings = CleanData._read_csv('app/data/Flight Bookings.csv')
            df_mapping = CleanData._read_csv('app/data/Airline ID to Name.csv')

            # Step 2: Correct and humanize column names for bookings
            df_bookings = CleanData._correct_column_names(df_bookings, agent_name="Bookings")

            # Step 3: Correct and humanize column names for mappings
            df_mapping = CleanData._correct_column_names(df_mapping, agent_name="Mappings")

            # Step 4: Merge dataframes on common columns
            merged_df = CleanData._merge_dataframes(df_mapping, df_bookings)

            # Step 5: Handle missing values
            merged_df = handle_missing_values(merged_df)
            logger.info("Missing values handled successfully.")

            # Step 6: Handle data inconsistencies
            merged_df = handle_inconsistencies(merged_df)
            logger.info("Data inconsistencies handled successfully.")

            logger.info("\nData Cleaning Completed Successfully.")
            
            # Step 7: Create a PandasAI agent with the cleaned dataframe
            CleanData._save_cleaned_data(merged_df, 'app/data/Clean_Booking_Details.csv')
            logger.info("PandasAI agent created successfully.")

        except FileNotFoundError as fnf_error:
            logger.error(f"File not found: {fnf_error}")
            raise FileNotFoundError(f"File not found: {fnf_error}")
        except AssertionError as assert_error:
            logger.error(f"Assertion error: {assert_error}")
            raise AssertionError(f"Assertion error: {assert_error}")
        except Exception as e:
            logger.error(f"An error occurred during data cleaning: {e}")
            raise Exception(f"An error occurred during data cleaning: {e}")

    @staticmethod
    def _read_csv(file_path: str) -> pd.DataFrame:
        """
        Read a CSV file into a pandas DataFrame.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: The loaded DataFrame.

        Raises:
            FileNotFoundError: If the CSV file does not exist at the specified path.
            pd.errors.ParserError: If the CSV file cannot be parsed.
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Successfully read CSV file: {file_path}")
            return df
        except FileNotFoundError as e:
            logger.error(f"CSV file not found: {file_path}")
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {file_path}")
            raise Exception(f"Error parsing CSV file: {file_path}")

    @staticmethod
    def _correct_column_names(df: pd.DataFrame, agent_name: str) -> pd.DataFrame:
        """
        Correct and humanize column names of a DataFrame using an LLM-powered agent.

        Args:
            df (pd.DataFrame): The DataFrame whose column names need correction.
            agent_name (str): A descriptive name for logging purposes.

        Returns:
            pd.DataFrame: The DataFrame with corrected column names.

        Raises:
            AssertionError: If the number of columns changes after renaming.
        """
        current_columns: List[str] = df.columns.tolist()
        correct_columns_prompt = (
            "Please correct and humanize the following column names: " + ", ".join(current_columns)
        )
        tools = [correct_columns_tool]
        llm = LLM.get_llm()
        agent = get_agent(tools=tools, llm=llm)

        try:
            corrected_output: ColumnNamesOutput = agent.run(correct_columns_prompt)
            corrected_columns = corrected_output.corrected_columns
            df.columns = corrected_columns
            assert len(current_columns) == len(df.columns), (
                f"Column count mismatch after renaming in {agent_name} DataFrame."
            )
            logger.info(f"Column names corrected successfully for {agent_name} DataFrame.")
            return df
        except AssertionError as e:
            logger.error(e)
            raise AssertionError(e)
        except Exception as e:
            logger.error(f"Error correcting column names for {agent_name} DataFrame: {e}")
            raise Exception(f"Error correcting column names for {agent_name} DataFrame: {e}")

    @staticmethod
    def _merge_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        """
        Merge two DataFrames on their common columns.

        Args:
            df1 (pd.DataFrame): The first DataFrame.
            df2 (pd.DataFrame): The second DataFrame.

        Returns:
            pd.DataFrame: The merged DataFrame.

        Raises:
            Exception: If merging fails due to incompatible columns or other issues.
        """
        try:
            merged_df = merge_dataframes_on_common_columns(df1, df2)
            logger.info("DataFrames merged successfully.")
            return merged_df
        except Exception as e:
            logger.error(f"Error while merging DataFrames: {e}")
            raise Exception("Error while merging on common columns")

    @staticmethod
    def _save_cleaned_data(df: pd.DataFrame, file_path: str) -> None:
        """
        Save the cleaned DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): The cleaned DataFrame to be saved.
            file_path (str): The destination path for the CSV file.

        Raises:
            Exception: If there is an error during the file saving process.
        """
        try:
            df.to_csv(file_path, index=False)
            logger.info(f"Cleaned DataFrame saved successfully to '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to save cleaned DataFrame to '{file_path}': {e}")
            raise Exception(f"Failed to save cleaned DataFrame to '{file_path}': {e}")