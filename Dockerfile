# Use an official lightweight Python 3.11 image as the foundation
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Install 'uv' (the extremely fast Python package manager)
RUN pip install uv

# Copy all of our local project files into the /app folder inside the container
COPY . /app

# Install all the project dependencies from pyproject.toml
RUN uv sync

# The command that tells the container how to start our Streamlit UI
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0"]
