FROM tensorflow/tensorflow:2.16.1

WORKDIR /app

COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --upgrade pip \
    && pip install --no-cache-dir numpy==1.24.4 scikit-learn==1.3.0 imbalanced-learn==0.12.4 \
    && pip install --no-cache-dir -r requirements.txt --no-deps

EXPOSE 5000

CMD ["python", "app.py"]
