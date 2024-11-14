from typing import Any
from langchain.agents import Tool
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from app.models.column_name_output import ColumnNamesOutput
from app.utils.llm_helper import LLM

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = PydanticOutputParser(pydantic_object=ColumnNamesOutput)

prompt = PromptTemplate(
    template=(
        "Please correct and humanize the following column names to make them more readable by:\n"
        "1. Make it more human readable.\n"
        "2. Separating words with underscores.\n"
        "3. Capitalizing the first letter of each word.\n\n"
        "For eg. airlie_id to Airline_ID, flght# to Flight_Number"
        "{format_instructions}\n"
        "{query}\n"
    ),
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


def correct_column_names(input_query: str) -> ColumnNamesOutput:
    """
    Correct and humanize column names using the language model.

    This function leverages a language model to process a given query containing
    column names, corrects and humanizes them, and returns the structured output
    with the corrected column names.

    Args:
        input_query (str): The prompt containing the current column names.

    Returns:
        ColumnNamesOutput: The structured output with corrected column names.

    Raises:
        ValueError: If there is an error parsing the model's output.
        Exception: If there is an unexpected error during the process.
    """
    logger.info("Starting column name correction process.")

    prompt_and_model = prompt | LLM.get_llm()

    try:
        logger.debug(f"Invoking language model with query: {input_query}")
        output = prompt_and_model.invoke({"query": input_query})
        logger.debug(f"Language model output: {output}")

        corrected_output = parser.invoke(output)
        logger.info("Column names corrected successfully.")

        return corrected_output

    except ValueError as ve:
        logger.error(f"Error parsing the model output: {ve}")
        raise ValueError(f"Error parsing the model output: {ve}") from ve
    except Exception as e:
        logger.error(f"An unexpected error occurred during column name correction: {e}")
        raise Exception(f"An unexpected error occurred: {e}") from e


correct_columns_tool = Tool(
    name="CorrectColumnNames",
    func=correct_column_names,
    description="Corrects and humanizes column names for a flight dataset.",
    return_direct=True
)