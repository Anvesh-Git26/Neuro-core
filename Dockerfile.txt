FROM python:3.9-slim

# Set the working directory to the root of the project
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code and data
COPY src/ ./src/
COPY data/ ./data/

# CRITICAL FIX: Set PYTHONPATH so Python knows where to find your modules
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Run training (using the robust script)
RUN python src/train_model.py

# Default command to run the robot
CMD ["python", "src/main_robot.py"]
