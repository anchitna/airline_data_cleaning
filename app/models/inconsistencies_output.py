from pydantic import BaseModel, Field, field_validator
from typing import Any, Dict, Optional

import re
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PandasCodeParser(BaseModel):
    """
    A parser for executing dynamically generated Python Pandas code on a DataFrame.

    This class validates the provided code to ensure it contains function definitions
    and extracts the Python code block. It then executes the extracted code on the
    provided DataFrame to produce a cleaned DataFrame.

    Attributes:
        code (str): The Python code as a string, expected to contain function definitions
                    and be enclosed within a markdown code block.
    """

    code: str = Field(
        ...,
        title="Python Code",
        description=(
            "A string containing Python code with function definitions. The code should "
            "be enclosed within a markdown Python code block."
        ),
        example="""
        ```python
        def clean_flight_data(df):
            # Cleaning logic here
            cleaned_df = df.dropna()
            return cleaned_df

        cleaned_df = clean_flight_data(df)
        """)
        
    @field_validator('code')
    def validate_and_extract_functions(cls, value: str) -> str:
        """
        Validates that the provided code contains function definitions and extracts the Python code block.

        Args:
            value (str): The Python code as a string.

        Returns:
            str: The extracted Python code without markdown syntax.

        Raises:
            ValueError: If the code does not contain any function definitions or if the Python code block
                        is not properly formatted.
        """
        logger.debug("Validating the provided code for function definitions.")
        if 'def' not in value:
            logger.error("The code must contain function definitions.")
            raise ValueError("The code must contain function definitions.")

        logger.debug("Searching for Python code block in the provided code.")
        code_match = re.search(r'```python\n(.*?)```', value, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            logger.debug("Python code block extracted successfully.")
            return extracted_code
        else:
            logger.error("Python code block not found in the provided code.")
            raise ValueError("Code block not found in the message.")

    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes the validated Python code on the provided DataFrame to produce a cleaned DataFrame.

        The method uses Python's `exec` function to run the extracted code in a controlled
        local environment. It expects the executed code to define a `clean_flight_data` function
        and assign the cleaned DataFrame to a variable named `cleaned_df`.

        Args:
            df (pd.DataFrame): The input DataFrame to be cleaned.

        Returns:
            pd.DataFrame: The cleaned DataFrame after executing the provided code.
                        If execution fails, the original DataFrame is returned.

        Raises:
            ValueError: If the `cleaned_df` variable is not defined in the executed code.
            Exception: If any error occurs during the execution of the code.
        """
        logger.info("Starting execution of the generated Python code.")
        local_vars: Dict[str, Any] = {'df': df}
        exec_globals: Dict[str, Any] = {
            'pd': pd,
            'np': np
        }

        try:
            logger.debug("Executing the Python code.")
            exec(self.code, exec_globals, local_vars)
            cleaned_df: Optional[pd.DataFrame] = local_vars.get('cleaned_df')

            if cleaned_df is None:
                logger.error("The executed code did not define 'cleaned_df'.")
                raise ValueError("The executed code did not define 'cleaned_df'.")

            if not isinstance(cleaned_df, pd.DataFrame):
                logger.error("'cleaned_df' is not a pandas DataFrame.")
                raise ValueError("'cleaned_df' is not a pandas DataFrame.")

            logger.info("Code executed successfully. DataFrame cleaned.")
            return cleaned_df

        except ValueError as ve:
            logger.error(f"Validation error during code execution: {ve}")
            raise ve
        except Exception as e:
            logger.error(f"An error occurred during code execution: {e}")
            raise Exception(f"An error occurred during code execution: {e}") from e