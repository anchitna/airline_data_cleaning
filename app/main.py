from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.services.data_cleaning_service import CleanData
from app.models.chat_model import UserChat
from pandasai import Agent
from app.utils.llm_helper import LLM

import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flight Booking Data Analysis API")

"""
    Initialize and train a PandasAI Agent..

    This function attempts to create and train a PandasAI `Agent` using the provided
    CSV data and language model (LLM). If an exception occurs during initialization
    or training, the function will retry the process up to `max_retries` times.

    Args:
        csv_path (str): 
            The file path to the CSV file containing booking details.
        llm: 
            The language model instance to be used by the Agent for processing queries.
        max_retries (int, optional): 
            The maximum number of retry attempts in case of failures. Defaults to 5.
        verbose (bool, optional): 
            Enables verbose logging within the Agent. Defaults to True.
        enable_cache (bool, optional): 
            Enables caching within the Agent to store intermediate results. Defaults to False.

    Creates:
        Agent: 
            An initialized and trained PandasAI `Agent` ready to handle user queries.
"""
    
retry_count = 0
while retry_count < 5:
    try:
        '''
        Performing cleaning on the data.
        '''
        CleanData.clean_data()
        llm = LLM.get_llm()
        booking_details = pd.read_csv("app/data/Clean Booking Details.csv")
        agent = Agent(booking_details, config={"llm": llm, "verbose": True, "enable_cache": False})
        agent.train(docs='''Try to find out the code for each user query and then try to execute it on the dataframe. 
                    Then answer the questions respectively.''')
        agent.train(docs='''Some questions can't be answered directly from the current columns. 
                    For those questions, generate the column according to the question 
                    given and then answer the question''')
        logging.info("PandasAI agent created and trained successfully.")
        retry_count = 5
    except Exception as e:
        logger.error("Retrying, error occured. ", e)
        retry_count += 1


@app.get("/", response_class=HTMLResponse)
async def read_index() -> HTMLResponse:
    """
    Serve the index HTML page.

    Reads and returns the content of `index.html` located in the `app/static` directory.

    Returns:
        HTMLResponse: The HTML content of the index page.

    Raises:
        HTTPException: If the index.html file is not found or cannot be read.
    """
    try:
        with open("app/static/index.html", "r", encoding="utf-8") as file:
            content = file.read()
            logger.info("Index page served successfully.")
            return HTMLResponse(content=content)
    except FileNotFoundError:
        logger.error("index.html file not found.")
        raise HTTPException(status_code=404, detail="Index page not found.")
    except Exception as e:
        logger.error(f"Error reading index.html: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

@app.post("/insights")
async def ask_query(query: UserChat) -> dict:
    """
    Handle user queries to fetch insights from the booking data.

    Processes the user's query by rephrasing it, passing it to the agent for analysis,
    and returning the fetched answer.

    Args:
        query (UserChat): The user query encapsulated in a UserChat model.

    Returns:
        dict: A dictionary containing the fetched answer or an error message.
    """
    try:
        logger.info(f"Received query: {query.query}")
        rephrased_query = agent.rephrase_query(query.query)
        logger.debug(f"Rephrased query: {rephrased_query}")
        answer = agent.chat(rephrased_query)
        logger.info("Query processed successfully.")
        return {"answer_fetched": answer}
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"answer_fetched": f"An error occurred while fetching the answer. Error: {e}"}