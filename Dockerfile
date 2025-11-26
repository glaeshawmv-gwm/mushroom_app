# Step 1: Use an official Python 3.11 slim image
FROM python:3.11-slim

# Step 2: Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Set working directory inside the container
WORKDIR /app

# Step 4: Copy project files into the container
COPY . /app

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Set environment variables
ENV FLASK_ENV=production
ENV TF_CPP_MIN_LOG_LEVEL=2
# Note: TF_CPP_MIN_LOG_LEVEL suppresses TensorFlow warnings

# Step 7: Expose the port
EXPOSE 8080

# Step 8: Run the app using Python
CMD ["python", "predict.py"]
