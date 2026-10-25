FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy frontend files into container
COPY . .

# Install frontend dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Launch Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]