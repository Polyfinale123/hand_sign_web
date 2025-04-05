# Use an official Python image as base
FROM python:3.11-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Render uses this)
EXPOSE 10000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Start the Flask app with gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:10000"]
