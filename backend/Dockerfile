FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy backend files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask/Gunicorn port
EXPOSE 7860

# Launch Flask application using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:superkart_api"]