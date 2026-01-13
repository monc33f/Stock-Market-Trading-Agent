# Use an official Python runtime as a parent image
FROM python:3.10

# Set environment variables
# PYTHONUNBUFFERED=1 ensures print statements appear immediately in logs
# MPLBACKEND=Agg prevents Matplotlib from trying to open a GUI window (which crashes Docker)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
# gcc/g++ are often needed to build RL libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create directory for results if it doesn't exist
RUN mkdir -p results

# Define the command to run your app
# Assuming your main entry point is src/run_experiment.py
CMD ["python", "src/run_experiment.py"]