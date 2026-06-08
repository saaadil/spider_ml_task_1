# imports
import os
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
from pathlib import Path
import logging

# get engine
from rag_engine import RAGEngine

# make app
app=Flask(__name__,template_folder='ui',static_folder='ui',static_url_path='')
CORS(app)

# start RAG
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Initializing RAG...")

# quick checks for required artifacts
rag = None
try:
    rag=RAGEngine()
    logger.info("Server ready.")
except Exception as e:
    logger.exception("Failed to initialize RAG engine")
    logger.error("RAG engine failed to initialize")

# load page
@app.route("/")
def index():
    return render_template("index.html")

# serve pictures
@app.route("/images/<path:filename>")
def serve_image(filename):
    base = os.path.normpath(os.path.join(os.path.dirname(__file__), "ui", "images"))
    safe_path = os.path.normpath(os.path.join(base, filename))
    if not safe_path.startswith(base):
        return jsonify({"error": "Invalid file path"}), 400
    if not os.path.exists(safe_path):
        return jsonify({"error": "File not found"}), 404
    return send_from_directory(base, filename)

# handle questions
@app.route("/api/query",methods=["POST"])
def query():
    # get data
    data=request.get_json()

    # check empty
    if not data or "question" not in data:
        return jsonify({"error": "Provide a 'question' field"}), 400

    # clean text
    question=data["question"].strip()

    # check blank
    if not question:
        return jsonify({"error": "Question is empty"}), 400

    # show query
    logger.info("Query: %s", question)

    # check engine ready
    if rag is None:
        return jsonify({"error": "RAG engine not ready"}), 503

    # fetch answer
    try:
        result=rag.query(question)
        return jsonify(result)
    except Exception:
        logger.exception("Error processing query")
        return jsonify({"error": "Internal server error"}), 500

# show stats
@app.route("/api/papers",methods=["GET"])
def list_papers():
    # ensure engine ready
    if rag is None or not hasattr(rag, "chunks"):
        return jsonify({"error": "RAG engine not ready"}), 503

    try:
        # count stuff
        papers=sorted({c["paper"] for c in rag.chunks})
        images=[c for c in rag.chunks if c.get("is_image")]
        
        # send stats
        return jsonify({
            "papers":       papers,
            "total_chunks": len(rag.chunks),
            "image_chunks": len(images)
        })
    except Exception:
        logger.exception("Error fetching papers")
        return jsonify({"error": "Failed to fetch papers"}), 500

# check health
@app.route("/api/health", methods=["GET"])
def health():
    if rag is None or not hasattr(rag, "index"):
        return jsonify({"status": "not ready"}), 503
    try:
        return jsonify({"status": "ok", "vectors": rag.index.ntotal})
    except Exception:
        logger.exception("Health check error")
        return jsonify({"status": "error"}), 500

# run server
if __name__ == "__main__":
    app.run(debug=True,port=5000)