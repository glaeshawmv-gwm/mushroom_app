# Step 1: Use official Python 3.11 slim
FROM python:3.11-slim

# Step 2: Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Set working directory
WORKDIR /app

# Step 4: Copy project files
COPY . /app

# Step 5: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Step 6: Environment variables
ENV FLASK_ENV=production
ENV TF_CPP_MIN_LOG_LEVEL=2

# Step 7: Expose port (Railway will set $PORT dynamically)
EXPOSE 8080

# Step 8: Run the app with gunicorn (shell form so $PORT expands)
CMD gunicorn -b 0.0.0.0:$PORT predict:app
