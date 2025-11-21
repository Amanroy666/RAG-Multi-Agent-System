"""Hybrid retrieval agent"""
import numpy as np
from typing import List, Dict

class RetrievalAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.vector_index = self._load_index()
    
    def _load_index(self):
        """Load FAISS index"""
        import faiss
        dimension = 1536
        index = faiss.IndexFlatL2(dimension)
        return index
    
    async def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Retrieve similar documents"""
        query_vector = np.array([query_embedding], dtype='float32')
        distances, indices = self.vector_index.search(query_vector, top_k)
        
        documents = []
        for dist, idx in zip(distances[0], indices[0]):
            documents.append({
                'id': int(idx),
                'score': float(1 / (1 + dist)),
                'content': f"Document {idx} content"
            })
        
        return documents
