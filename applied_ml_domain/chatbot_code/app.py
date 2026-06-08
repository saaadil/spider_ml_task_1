# imports
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path

# get engine
from rag_engine import RAGEngine

# make app
app = Flask(__name__,static_folder="ui")
CORS(app)

# start RAG
print("Initializing RAG...")
try:
    rag=RAGEngine()
    print("Server ready.\n")
except Exception as e:
    print(f"Failed to initialize RAG engine: {e}")
    raise

# load page
@app.route("/")
def index():
    base_dir=os.path.dirname(os.path.abspath(__file__))
    ui_dir=os.path.join(base_dir, "ui")
    return send_from_directory(ui_dir,"index.html")

# serve pictures
@app.route("/images/<path:filename>")
def serve_image(filename):
    return send_from_directory(os.path.join("ui","images"), filename)

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
    print(f"Query: {question}")

    # fetch answer
    try:
        result=rag.query(question)
        return jsonify(result)
    # catch error
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

# show stats
@app.route("/api/papers",methods=["GET"])
def list_papers():
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
    except Exception as e:
        print(f"Error fetching papers: {e}")
        return jsonify({"error": "Failed to fetch papers"}), 500

# check health
@app.route("/api/health", methods=["GET"])
def health():
    try:
        return jsonify({"status": "ok", "vectors": rag.index.ntotal})
    except Exception as e:
        print(f"Health check error: {e}")
        return jsonify({"status": "error"}), 500

# run server
if __name__ == "__main__":
    app.run(debug=True,port=5000)