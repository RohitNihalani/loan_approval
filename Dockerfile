FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything (backend folder, frontend folder, models, etc.)
COPY . .

# Fix Linux permissions for the script
RUN chmod +x start.sh

# Expose ports
EXPOSE 8501
EXPOSE 8000

CMD ["./start.sh"]


