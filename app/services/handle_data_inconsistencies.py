from typing import Dict, Any
from langchain.prompts import PromptTemplate
from app.utils.llm_helper import LLM
from app.models.inconsistencies_output import PandasCodeParser

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

analysis_prompt = PromptTemplate(
    input_variables=["columns", "data_sample"],
    template="""
    Given the following column names of a Pandas DataFrame:
    {columns}
    
    And the following sample data:
    {data_sample}
    
    Analyze potential data issues and inconsistencies in the dataset, focusing on:
    1. Missing values in numerical and date/time columns.
    2. Negative values in numerical columns where such values are not logically valid.
    3. Date and time inconsistencies, such as:
        - 'Arrival Date' being before 'Departure Date'.
        - 'Departure Time' being after 'Arrival Time' on the same day.
    4. Data type mismatches based on column names and sample data.
    5. Logical errors specific to the context of the dataset (e.g., invalid flight numbers).
    
    Generate Python Pandas code to fix these issues. The code should handle:
    - Imputing or removing missing values in numerical and date/time columns.
    - Correcting or removing negative values in numerical columns.
    - Ensuring that 'Arrival Date' is not before 'Departure Date'.
    - Ensuring that 'Departure Time' is not after 'Arrival Time' when dates are the same.
    - Correcting data type mismatches and any other logical inconsistencies based on the columns present.
    
    The Python code should:
    - Define a function named `clean_flight_data` that takes a DataFrame as input and applies all necessary cleaning steps.
    - Automatically identify the relevant columns based on their names and data types.
    - Handle various data issues dynamically without hardcoding column names, except for those explicitly mentioned.
    - Ensure that the cleaned DataFrame maintains data integrity and logical consistency.
    
    Only return the Python code with the function being defined and called with a parameter `df` have a have a code `cleaned_df = clean_flight_data(df)` at the end. Do not include any explanations or additional text.
    Don't give example usage and do all the imports correctly.
    """
)

# Create a chain by combining the prompt with the language model
chain = analysis_prompt | LLM.get_llm()

def handle_inconsistencies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze and fix data inconsistencies in a Pandas DataFrame.

    This function leverages a language model to generate Python code that identifies and
    resolves various data issues within the DataFrame, such as missing values,
    negative values, date/time inconsistencies, data type mismatches, and logical errors.
    The generated code is then parsed and executed to produce a cleaned DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to be cleaned.

    Returns:
        pd.DataFrame: The cleaned DataFrame after resolving inconsistencies.
                      If an error occurs during the cleaning process, the original DataFrame is returned.

    Raises:
        ValueError: If the generated code fails validation or execution.
    """
    logger.info("Starting to handle inconsistencies in the DataFrame.")
    
    # Extract column names and a sample of the data
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