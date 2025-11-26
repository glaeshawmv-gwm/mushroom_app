# Step 1: Use an official Python 3.11 slim image

FROM python:3.11-slim

# Step 2: Install system dependencies required by OpenCV

RUN apt-get update && apt-get install -y 
libgl1 
libglib2.0-0 
&& rm -rf /var/lib/apt/lists/*

# Step 3: Set working directory inside the container

WORKDIR /app

# Step 4: Copy project files into the container

COPY . /app

# Step 5: Install Python dependencies

RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Set environment variables for production

ENV FLASK_ENV=production
ENV PORT=8080
ENV TF_CPP_MIN_LOG_LEVEL=2  # suppress TensorFlow warnings

# Step 7: Expose the port Railway expects

EXPOSE 8080

# Step 8: Run the app using gunicorn for production

# Replace 'predict:app' with your Flask app module and variable

CMD ["gunicorn", "-b", "0.0.0.0:8080", "predict:app"]
