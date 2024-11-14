from typing import Any, List
from langchain.agents import AgentType, initialize_agent

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_agent(tools: List[Any], llm: Any) -> Any:
    """
    Initialize and return a Langchain agent with the specified tools and language model.

    This function sets up a Langchain agent using the provided tools and language model (LLM).
    The agent is configured to use the ZERO_SHOT_REACT_DESCRIPTION agent type with verbose output.

    Args:
        tools (List[Any]): A list of tools to be integrated with the agent. Each tool should
                           conform to the expected interface required by Langchain agents.
        llm (Any): The language model instance to be used by the agent. This should be an
                   instance compatible with Langchain's LLM interface.

    Returns:
        Any: An initialized Langchain agent configured with the provided tools and LLM.

    Example:
        ```python
        from langchain.llms import OpenAI
        from langchain.tools import SomeTool

        llm = OpenAI(api_key="your-api-key")
        tools = [SomeTool(), AnotherTool()]
        agent = get_agent(tools, llm)
        ```
    """
    logger.info("Initializing Langchain agent with provided tools and language model.")

    try:
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        logger.info("Langchain agent initialized successfully.")
        return agent
    except Exception as e:
        logger.error(f"Failed to initialize Langchain agent: {e}")
        raise RuntimeError(f"Failed to initialize Langchain agent: {e}")