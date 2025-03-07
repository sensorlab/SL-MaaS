# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt

ADD . /app


# For ARM architecture (e.g., Raspberry Pi): switch to the appropriate wheel file
RUN pip install ts2vg==1.2.3
# RUN pip install ts2vg-1.2.3-cp310-cp310-linux_aarch64.whl

EXPOSE 8000

# Run app.py when the container launches
CMD ["uvicorn", "api-endpoint:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--limit-concurrency", "8000", "--log-level", "error", "--backlog", "8000"]
