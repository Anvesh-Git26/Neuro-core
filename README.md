# Hybrid Neuro-Core: Latency-Optimized NLP Engine for Robotics

## 🚀 Project Overview
This project implements a dual-layer NLP architecture designed for companion robots . It solves the problem of distinguishing between **Hardware Commands** (e.g., "Dance") and **Open-Ended Questions** (e.g., "Why is the sky blue?") with low latency.

It uses a **DistilBERT** classifier for intent detection and a **FAISS-based RAG system** for retrieving knowledge from unstructured text.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, HuggingFace Transformers (DistilBERT)
* **Vector Search:** FAISS (Facebook AI Similarity Search), Sentence-Transformers
* **Deployment:** Docker, Python 3.9
* **Data Engineering:** Synthetic Data Generation via LLM Prompting

## ⚙️ Architecture
1.  **Input Layer:** User speech text is tokenized.
2.  **Intent Classification:** A fine-tuned DistilBERT model routes the query:
    * `ACTION` -> Triggers motor control functions.
    * `KNOWLEDGE` -> Triggers the RAG pipeline.
3.  **Knowledge Retrieval:** Semantic search over a vector database using Cosine Similarity.

## 🔧 How to Run
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Train the Model:**
    ```bash
    cd src
    python train_model.py
    ```
3.  **Run the Robot Brain:**
    ```bash
    python main_robot.py
    ```
