FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- NEW STEP: Download SpaCy Model ---
RUN python -m spacy download en_core_web_sm
# --------------------------------------

COPY src/ ./src/
COPY data/ ./data/

ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Pre-train the classifier so the container is ready to go
RUN python src/train_model.py

CMD ["python", "src/main_robot.py"]
