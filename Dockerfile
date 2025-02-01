# Use an official lightweight Python image
FROM python:3.11.5

# Set the working directory inside the container
WORKDIR /app

# Copy only necessary files
COPY requirements.txt .
COPY model.pkl scaler.pkl label_encoders.pkl .  
COPY ./templates ./templates
COPY app.py .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 8080

# Run the Flask app
CMD ["python", "app.py"]
