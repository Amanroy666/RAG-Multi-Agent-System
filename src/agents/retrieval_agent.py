"""
Retrieval Agent - Handles hybrid search across vector and keyword indices
"""
from typing import List, Dict, Any
import numpy as np
from ..retrieval.vector_search import VectorSearch
from ..retrieval.keyword_search import KeywordSearch
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class RetrievalAgent:
    """Hybrid retrieval combining vector and keyword search"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.vector_search = VectorSearch(config['vector'])
        self.keyword_search = KeywordSearch(config['keyword'])
        self.alpha = config.get('hybrid_alpha', 0.7)  # Weight for vector search
    
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of relevant documents with scores
        """
        # Vector search
        vector_results = await self.vector_search.search(query, top_k=top_k * 2)
        
        # Keyword search
        keyword_results = await self.keyword_search.search(query, top_k=top_k * 2)
        
        # Hybrid fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results,
            k=60
        )
        
        return fused_results[:top_k]
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Fuse results using Reciprocal Rank Fusion
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            k: RRF constant
        
        Returns:
            Fused and ranked results
        """
        doc_scores = {}
        
        # Score from vector search
        for rank, doc in enumerate(vector_results, 1):
            doc_id = doc['id']
            score = self.alpha / (k + rank)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + score
            if doc_id not in doc_scores:
                doc_scores[doc_id] = {'doc': doc, 'score': 0}
            doc_scores[doc_id] = doc
        
        # Score from keyword search
        for rank, doc in enumerate(keyword_results, 1):
            doc_id = doc['id']
            score = (1 - self.alpha) / (k + rank)
            if doc_id in doc_scores:
                doc_scores[doc_id]['score'] += score
            else:
                doc['score'] = score
                doc_scores[doc_id] = doc
        
        # Sort by fused score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return sorted_docs
