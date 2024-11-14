from pydantic import BaseModel, Field

class UserChat(BaseModel):
    """
    Represents a user chat query for the Flight Booking Data Analysis API.

    Attributes:
        query (str): The user's natural language query seeking insights or analysis from the flight booking data.
    """
    query: str = Field(
        ...,
        title="User Query",
        description="A natural language question or request for insights related to flight booking data."
    )