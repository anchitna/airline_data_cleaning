from pydantic import BaseModel, Field, field_validator
from typing import List

class ColumnNamesOutput(BaseModel):
    """
    Represents the output containing corrected and humanized column names.

    Attributes:
        corrected_columns (List[str]): A list of corrected and humanized column names.
    """

    corrected_columns: List[str] = Field(
        ...,
        title="Corrected Columns",
        description="List of corrected and humanized column names.",
        example=["FlightID", "DepartureDate", "ArrivalDate", "PassengerCount"]
    )

    @field_validator('corrected_columns')
    def no_empty_columns(cls, value: List[str]) -> List[str]:
        """
        Validates that the `corrected_columns` list is not empty.

        Args:
            value (List[str]): The list of corrected column names.

        Returns:
            List[str]: The validated list of corrected column names.

        Raises:
            ValueError: If the `corrected_columns` list is empty.
        """
        if not value:
            raise ValueError("The corrected_columns list cannot be empty.")
        return value