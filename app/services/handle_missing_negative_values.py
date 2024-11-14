from typing import Any, Dict
from langchain.prompts import PromptTemplate
from app.utils.llm_helper import LLM
from app.models.inconsistencies_output import PandasCodeParser

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

analysis_prompt = PromptTemplate(
    input_variables=["columns", "data_sample"],
    template="""
    Given the following column names of a Pandas DataFrame:
    {columns}
    
    And the following sample data:
    {data_sample}
    
    Analyze potential data issues focusing solely on:
    1. Missing values in numerical columns.
    2. Negative values in numerical columns.
    
    Generate Python Pandas code to fix these issues. The code should handle:
    - Imputing or removing missing values in numerical columns.
    - Correcting or removing negative values in numerical columns where negative values are not logically valid.
    
    The Python code should define a function named `clean_flight_data` that takes a DataFrame as input and applies all necessary cleaning steps. 
    Only return the Python code with the function being called with a parameter `df` and have a code `cleaned_df = clean_flight_data(df)` at the end, nothing else.
    Don't give example usage and do all the imports correctly.
    """
)

chain = analysis_prompt | LLM.get_llm()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze and fix missing and negative values in numerical columns of a Pandas DataFrame.

    This function leverages a language model to generate Python code that identifies and
    resolves missing values and negative values in numerical columns. The generated code
    is then parsed and executed to produce a cleaned DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame after resolving missing and negative values.
                      If an error occurs during the cleaning process, the original DataFrame is returned.

    Raises:
        ValueError: If the generated code fails validation or execution.
        Exception: For any other unexpected errors during the cleaning process.
    """
    logger.info("Starting to handle missing and negative values in the DataFrame.")

    columns: list = list(df.columns)
    data_sample: Dict[str, Any] = df.head().to_dict()

    logger.debug(f"DataFrame columns: {columns}")
    logger.debug(f"DataFrame sample data: {data_sample}")

    try:
        logger.info("Invoking language model to generate cleaning code.")
        generated_code = chain.invoke({
            "columns": columns,
            "data_sample": data_sample
        })
        logger.debug(f"Generated code:\n{generated_code.content}")

        logger.info("Parsing and executing the generated cleaning code.")
        parsed_code = PandasCodeParser(code=generated_code.content)
        cleaned_df = parsed_code.execute(df)
        logger.info("Code executed successfully. DataFrame cleaned.")

        return cleaned_df

    except ValueError as ve:
        logger.error(f"Validation error during cleaning: {ve}")
        return df
    except Exception as e:
        logger.error(f"An unexpected error occurred during data cleaning: {e}")
        return df