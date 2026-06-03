# bring tools
import os
import pickle
import requests
import base64
import numpy as np
import fitz  
import faiss
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# paper links
PAPERS = {
    "Attention Is All You Need": "https://arxiv.org/pdf/1706.03762.pdf",
    "BERT": "https://arxiv.org/pdf/1810.04805.pdf",
    "GPT-3": "https://arxiv.org/pdf/2005.14165.pdf",
    "RAG": "https://arxiv.org/pdf/2005.11401.pdf",
    "Sentence-BERT": "https://arxiv.org/pdf/1908.10084.pdf",
    "LoRA": "https://arxiv.org/pdf/2106.09685.pdf",
    "Llama 2": "https://arxiv.org/pdf/2307.09288.pdf"
}

# folder paths
PAPERS_DIR = "papers"
DB_PATH = "embeddings/vector_db.pkl"
IMAGE_DIR = "ui/images"
SUPPORTED_IMG_EXTS = {"png","jpeg","jpg"}
OLLAMA_URL="http://localhost:11434/api/generate"

# get file
def download_paper(name,url):
    filename = os.path.join(PAPERS_DIR,name.replace(" ","_")+".pdf")
    if os.path.exists(filename): return filename
    print(f"  Downloading {name}...")
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"},timeout=60)
    if res.status_code == 200:
        with open(filename, "wb") as f: f.write(res.content)
        return filename
    return None

# look images
def extract_images_and_caption(filepath, paper_name):
    image_chunks = []
    doc = fitz.open(filepath)
    
    # loop pages
    for page_num in range(len(doc)):
        page = doc[page_num]
        images=page.get_images(full=True)
        
        # loop images
        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image =doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext= base_image["ext"].lower()
            
            # skip tiny
            if len(image_bytes) < 10000 or image_ext not in SUPPORTED_IMG_EXTS: 
                continue
            
            # save image
            img_filename = f"{paper_name.replace(' ', '_')}_p{page_num}_{img_idx}.{image_ext}"
            img_path = os.path.join(IMAGE_DIR, img_filename)
            with open(img_path, "wb") as f:
                f.write(image_bytes)
                
            print(f"  Captioning {img_filename} locally with gemma4:e4b...")
            b64_data = base64.b64encode(image_bytes).decode('utf-8')
            
            # ask
            try:
                payload = {
                    "model": "gemma4:e4b",
                    "prompt": "Describe this figure from a machine learning paper in extreme technical detail. Explain the architecture, data flow, or equations shown. Start with 'Figure showing...'",
                    "images": [b64_data],
                    "stream": False
                }
                response = requests.post(OLLAMA_URL, json=payload, timeout=None)
                caption = response.json().get("response", "").strip()
                
                # save caption
                if caption:
                    image_chunks.append({
                        "text": caption,
                        "paper": paper_name,
                        "image_path": f"images/{img_filename}",
                        "is_image": True
                    })
            # error block
            except Exception as e:
                print(f"  Ollama local captioning failed for {img_filename}: {e}")
                
    # shut file
    doc.close()
    return image_chunks

# read words
def extract_text_smart(filepath):
    doc = fitz.open(filepath)
    text = ""
    # loop text
    for page in doc:
        for b in page.get_text("blocks"):
            if b[6] == 0: text += b[4] + "\n"
    # shut file
    doc.close()
    return text

# main script
def main():
    # make folders
    os.makedirs(PAPERS_DIR, exist_ok=True)
    os.makedirs("embeddings", exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # text chopper
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    all_chunks = []
    chunk_id = 0

    # process loop
    for name, url in PAPERS.items():
        filepath = download_paper(name, url)
        if not filepath: continue

        # get text
        raw_text = extract_text_smart(filepath)
        for text_chunk in text_splitter.split_text(raw_text):
            all_chunks.append({"chunk_id": chunk_id, "text": text_chunk, "paper": name, "is_image": False})
            chunk_id += 1
            
        # get images
        #img_chunks = extract_images_and_caption(filepath, name)
        #for ic in img_chunks:
        #    ic["chunk_id"] = chunk_id
        #    all_chunks.append(ic)
        #    chunk_id += 1

        print(f"Processed {name} successfully.")

    # load model
    print("\nEmbedding all local chunks with all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([c["text"] for c in all_chunks], show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)

    # make index
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    # save math
    with open(DB_PATH, "wb") as f:
        pickle.dump({"faiss_index_bytes": faiss.serialize_index(index), "metadata": all_chunks}, f)

    print(f"\nSuccess. {index.ntotal} local hybrid vectors stored in {DB_PATH}")

# run execution
if __name__ == "__main__":
    main()