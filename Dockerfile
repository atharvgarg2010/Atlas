FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# The start script will run both the Streamlit app and the background scheduler
COPY start.sh .
RUN chmod +x start.sh

CMD ["./start.sh"]
