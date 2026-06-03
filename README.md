# Spider ML Induction Task 1 - Submission

A complete machine learning suite encompassing foundational model evaluation, advanced local optimization, and custom hybrid information retrieval systems.

## 📁 Repository Structure
* **base_task/**: Contains data science pipeline notebooks and prediction outputs.
* **applied_ml_domain/chatbot_code/**: An air-gapped, fully local Hybrid RAG Research Assistant (`all-MiniLM-L6-v2` + `BM25Okapi` + `Ollama/Gemma`).
* **bonus_task/**: High-level optimization code implementations and performance benchmarks.

## 🛠️ How to Boot the Local RAG Chatbot
1. Ensure Ollama is running in the background with the Gemma model: `ollama run gemma4:e4b`
2. Navigate to the codebase: `cd applied_ml_domain/chatbot_code`
3. Launch the web controller: `python app.py`
4. Access the platform in your browser at `http://localhost:5000`
