import pandas as pd
import logging

class TestQuestions:
    def __init__(self, data_path: str):
        """
        Initializes the TestQuestions class with data loading and logger setup.

        Parameters:
        - data_path (str): Path to the CSV data file.
        """
        self.data_path = data_path
        self.logger = self.setup_logger()
        self.data = self.load_data()

    def setup_logger(self) -> logging.Logger:
        """
        Sets up the logger to output logs to the console.

        Returns:
        - logging.Logger: Configured logger object.
        """
        logger = logging.getLogger("TestQuestionsLogger")
        logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)

        if not logger.handlers:
            logger.addHandler(ch)

        logger.info("Logger initialized and set to output to console.")
        return logger

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the specified CSV file into a pandas DataFrame.

        Returns:
        - pd.DataFrame: Loaded data.
        """
        try:
            self.logger.info(f"Loading data from {self.data_path}.")
            data = pd.read_csv(self.data_path)
            self.logger.info("Data loaded successfully.")
            return data
        except FileNotFoundError:
            self.logger.error(f"File not found: {self.data_path}.")
            raise
        except pd.errors.ParserError:
            self.logger.error(f"Error parsing the file: {self.data_path}.")
            raise
        except Exception as e:
            self.logger.error(f"An unexpected error occurred: {e}.")
            raise

    def preprocess_data(self):
        """
        Preprocesses the data by converting date columns to datetime and extracting the departure month.
        """
        try:
            self.logger.info("Preprocessing data: Converting 'Departure_Date' to datetime.")
            self.data['Departure_Date'] = pd.to_datetime(self.data['Departure_Date'])
            self.logger.debug("'Departure_Date' conversion successful.")

            self.logger.info("Extracting 'Departure_Month' from 'Departure_Date'.")
            self.data['Departure_Month'] = self.data['Departure_Date'].dt.month_name()
            self.logger.debug("'Departure_Month' extraction successful.")
        except KeyError as e:
            self.logger.error(f"Missing expected column: {e}.")
            raise
        except Exception as e:
            self.logger.error(f"An error occurred during preprocessing: {e}.")
            raise

    def get_top_airline(self) -> tuple:
        """
        Determines which airline has the most flights listed.

        Returns:
        - tuple: (Airline Name, Number of Flights)
        """
        try:
            self.logger.info("Calculating the airline with the most flights.")
            airline_flight_counts = self.data['Airline_Name'].value_counts()
            self.logger.debug(f"Airline flight counts:\n{airline_flight_counts}")

            top_airline = airline_flight_counts.idxmax()
            top_airline_count = airline_flight_counts.max()

            self.logger.info(f"Top airline: {top_airline} with {top_airline_count} flights.")
            return top_airline, top_airline_count
        except KeyError:
            self.logger.error("'Airline_Name' column not found in data.")
            raise
        except Exception as e:
            self.logger.error(f"An error occurred while determining the top airline: {e}.")
            raise

    def get_top_month(self) -> tuple:
        """
        Identifies the month with the highest number of bookings.

        Returns:
        - tuple: (Month Name, Number of Bookings)
        """
        try:
            self.logger.info("Calculating the month with the highest number of bookings.")
            monthly_bookings = self.data['Departure_Month'].value_counts()
            self.logger.debug(f"Monthly bookings counts:\n{monthly_bookings}")

            top_month = monthly_bookings.idxmax()
            top_month_count = monthly_bookings.max()

            self.logger.info(f"Top month: {top_month} with {top_month_count} bookings.")
            return top_month, top_month_count
        except KeyError:
            self.logger.error("'Departure_Month' column not found in data.")
            raise
        except Exception as e:
            self.logger.error(f"An error occurred while determining the top month: {e}.")
            raise

    def test_queries(self):
        """
        Executes the test queries and prints the results.
        """
        try:
            self.logger.info("Starting test queries.")
            self.preprocess_data()

            # Query 1: Which airline has the most flights listed?
            top_airline, top_airline_count = self.get_top_airline()
            print(f"The airline with the most flights listed is {top_airline} with {top_airline_count} flights.")

            # Query 2: Month with the highest number of bookings.
            top_month, top_month_count = self.get_top_month()
            print(f"The month with the highest number of bookings is {top_month} with {top_month_count} bookings.")

            self.logger.info("Test queries executed successfully.")
        except Exception as e:
            self.logger.error(f"An error occurred during test queries execution: {e}.")
            print("An error occurred while executing the test queries. Please check the logs for more details.")
            
if __name__ == "__main__":
    # Path to your CSV data file
    data_file_path = "app/data/Clean_Booking_Details.csv"

    # Create an instance of TestQuestions
    test = TestQuestions(data_path=data_file_path)

    # Execute the test queries
    test.test_queries()
