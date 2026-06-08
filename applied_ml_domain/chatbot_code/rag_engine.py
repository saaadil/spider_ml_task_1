# bring tools
import os
import pickle
import numpy as np
import faiss
import requests
import logging

logger = logging.getLogger(__name__)
from sentence_transformers import SentenceTransformer

# configuration values
TOP_K=5
DB_PATH="embeddings/vector_db.pkl"
OLLAMA_URL=os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
logger = logging.getLogger(__name__)
logger.info("Loading local embedding model...")
# engine class
class RAGEngine:
    def __init__(self):
        logger.info("Loading vector database from %s...", DB_PATH)
        logger.info("Loading local embedding model...")
        self.embed_model=SentenceTransformer("all-MiniLM-L6-v2")

        # open files
        logger.info("Loading vector database from %s...", DB_PATH)
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"{DB_PATH} not found. Run ingest.py first.")

        with open(DB_PATH,"rb") as f:
            db=pickle.load(f)

        # fix index
        self.index=faiss.deserialize_index(db["faiss_index_bytes"])
        self.chunks=db["metadata"]
        self.ollama_url=OLLAMA_URL

        # start searcher
        logger.info("RAG engine ready — %s vectors indexed.", self.index.ntotal)

    def generate_multi_queries(self,original_query):
        # make prompt
        prompt=f"""You are an AI assistant. The user wants to search a database of NLP research papers. 
Generate 2 alternative search queries that break down or rephrase the following question to maximize retrieval of relevant facts.
Return ONLY the queries, one per line. Do not number them. Do not include introductory text.
User Question: {original_query}"""

        # ask robot
        try:
            payload={
                "model": "gemma4:e4b",
                "prompt": prompt,
                "stream": False
            }
            response=requests.post(OLLAMA_URL, json=payload, timeout=None)
            raw_output=response.json().get("response", "").strip().split('\n')
            queries=[q.strip("- ") for q in raw_output if q.strip()]
        # backup check
        except Exception:
            queries=[]
            
        queries.append(original_query)
        seen = set()
        unique = []
        for q in queries:
            if q not in seen:
                seen.add(q)
                unique.append(q)
        return unique

    def retrieve(self,query_list, top_k=TOP_K):
        # empty bag
        all_results={}
        
        # search loops
        for q in query_list:
            q_emb=self.embed_model.encode([q])
            q_emb=np.array(q_emb).astype("float32")
            faiss.normalize_L2(q_emb)

            scores,indices=self.index.search(q_emb, top_k)

            # inspect match
            for score, idx in zip(scores[0], indices[0]):
                if idx==-1:
                    continue
                
                chunk =self.chunks[idx].copy()
                chunk_id =chunk["chunk_id"]
                
                # save best
                if chunk_id not in all_results or score > all_results[chunk_id]["score"]:
                    chunk["score"]=float(score)
                    all_results[chunk_id]=chunk

        # sort array
        sorted_results=sorted(list(all_results.values()), key=lambda x: x["score"], reverse=True)
        return sorted_results[:top_k * 2] 

    def build_context(self, chunks):
        # merge chunks
        parts=[]
        for c in chunks:
            if c.get("is_image"):
                parts.append(f"[Source: {c['paper']} — Figure]\n{c['text']}")
            else:
                parts.append(f"[Source: {c['paper']}]\n{c['text']}")
        return "\n\n---\n\n".join(parts)

    def generate_answer(self, query, chunks):
        # glue text
        context=self.build_context(chunks)

        prompt = f"""You are a research assistant. Answer the user's question 
based ONLY on the provided context from research papers.

If the context doesn't have enough info, say so clearly. Do not make things up.
Always mention which paper the information comes from.

Context:
{context}

Question: {query}

Answer:"""

        # query robot
        try:
            payload={
            "model": "gemma4:e4b", 
            "prompt": prompt, 
            "stream": False
        }
            response=requests.post(self.ollama_url, json=payload, timeout=None)
            answer=response.json().get("response", "Error generating response.")
        # backup text
        except Exception as e:
            answer=f"Failed to connect to local Ollama server: {e}"
        return answer

    def query(self, user_question):
        # expand text
        search_queries = self.generate_multi_queries(user_question)
        logger.info("Expanded Queries: %s", search_queries)
        
        # fetch matching
        retrieved=self.retrieve(search_queries)

        # verify existence
        if not retrieved:
            return {
                "answer": "No relevant information found in the indexed papers.",
                "sources": []
            }

        # fetch answer
        answer=self.generate_answer(user_question, retrieved)

        # setup sources
        sources=[]
        for c in retrieved:
            source_info={
                "paper": c["paper"],
                "score": round(c["score"], 3),
                "snippet": c["text"][:200] + "...",
                "is_image": c.get("is_image", False)
            }
            if source_info["is_image"] and "image_path" in c:
                source_info["image_path"]=c["image_path"]
            sources.append(source_info)

        return {"answer": answer,"sources": sources}