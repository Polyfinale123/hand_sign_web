# Base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Create working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 5000

# Start the app
CMD ["python", "app.py"]
