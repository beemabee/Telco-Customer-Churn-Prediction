# Use python base image 
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements.txt into the image
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --default-timeout=100

# Copy the rest of the application code
COPY . .

# Copy the model file
COPY model/catboost_model.cbm /app/model/catboost_model.cbm

# Expose port 
EXPOSE 5000

# Run the app
CMD [ "python", "src/predict.py" ]