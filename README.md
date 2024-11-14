# Installation Guide

Follow these steps to set up the project locally

1. Clone the Repository: git clone https://github.com/your-username/your-repo-name.git
2. cd your-repo-name
3. Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies.

# bash

python3 -m venv venv

Activate the virtual environment:

On Unix or MacOS:

source venv/bin/activate

On Windows:

venv\Scripts\activate

# Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

#  Quick Start

Start the FastAPI application using Uvicorn.

uvicorn main:app --reload
main: The Python file where your FastAPI app instance is located (e.g., main.py).
app: The FastAPI instance (e.g., app = FastAPI()).
--reload: Enables auto-reloading on code changes (useful during development).
After running the command, you should see output similar to:

INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [28724] using statreload
INFO:     Started server process [28726]
INFO:     Waiting for application startup.
INFO:     Application startup complete.

# Usage
Accessing the API
Once the server is running, you can interact with the API endpoints.

Base URL: http://127.0.0.1:8000
Interactive API Docs
FastAPI provides interactive documentation out of the box.