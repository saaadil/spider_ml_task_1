# Spider ML Induction: Task 1

This repository contains my complete submission for the Spider ML inductions. The workspace is split into three main deliverables: a baseline classifier, an autoencoder dimensionality experiment, and a fully local Multi-Query RAG web application.

## Directory Layout
* `/base_task/` - The Fashion-MNIST classification pipeline, saved weights, and evaluation metrics.
* `/bonus_task/` - Dimensionality reduction experiments using an MLP Autoencoder to find the intrinsic dataset bottleneck.
* `/applied_ml_domain/chatbot_code/` - A local, air-gapped Multi-Query RAG Research Assistant built with Flask, FAISS, and Ollama.

## Running the RAG Web App
The chatbot runs entirely locally. Ensure you have Ollama installed and running in the background before booting the server.

1. Start the local LLM: `ollama run gemma4:e4b`
2. Navigate to the backend directory: `cd applied_ml_domain/chatbot_code`
3. Launch the web controller: `python app.py`
4. Access the UI at: `http://localhost:5000`