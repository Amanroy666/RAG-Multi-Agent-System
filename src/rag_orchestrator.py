"""
RAG System Orchestrator - Coordinates multi-agent retrieval and generation
"""
from typing import List, Dict, Any
from .agents.retrieval_agent import RetrievalAgent
from .agents.reformulation_agent import ReformulationAgent
from .agents.synthesis_agent import SynthesisAgent
from .agents.attribution_agent import AttributionAgent
from .utils.logger import setup_logger

logger = setup_logger(__name__)

class RAGOrchestrator:
    """Orchestrates multi-agent RAG pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retrieval_agent = RetrievalAgent(config['retrieval'])
        self.reformulation_agent = ReformulationAgent(config['reformulation'])
        self.synthesis_agent = SynthesisAgent(config['synthesis'])
        self.attribution_agent = AttributionAgent(config['attribution'])
        logger.info("RAG Orchestrator initialized")
    
    async def query(self, user_query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Process user query through RAG pipeline
        
        Args:
            user_query: User's question
            top_k: Number of documents to retrieve
        
        Returns:
            Dict containing answer, sources, and confidence score
        """
        logger.info(f"Processing query: {user_query}")
        
        # Step 1: Query reformulation
        reformulated_queries = await self.reformulation_agent.reformulate(user_query)
        logger.debug(f"Reformulated queries: {reformulated_queries}")
        
        # Step 2: Multi-query retrieval
        all_documents = []
        for query in reformulated_queries:
            docs = await self.retrieval_agent.retrieve(query, top_k=top_k)
            all_documents.extend(docs)
        
        # Deduplicate and rerank
        unique_docs = self._deduplicate_documents(all_documents)
        ranked_docs = self._rerank_documents(unique_docs, user_query)[:top_k]
        
        logger.info(f"Retrieved {len(ranked_docs)} relevant documents")
        
        # Step 3: Answer synthesis
        answer = await self.synthesis_agent.synthesize(
            query=user_query,
            documents=ranked_docs
        )
        
        # Step 4: Source attribution
        attributed_answer = await self.attribution_agent.attribute(
            answer=answer,
            sources=ranked_docs
        )
        
        return {
            'answer': attributed_answer['answer'],
            'sources': attributed_answer['sources'],
            'confidence': attributed_answer['confidence'],
            'metadata': {
                'num_documents_retrieved': len(all_documents),
                'num_unique_documents': len(unique_docs),
                'reformulated_queries': reformulated_queries
            }
        }
    
    def _deduplicate_documents(self, documents: List[Dict]) -> List[Dict]:
        """Remove duplicate documents based on content hash"""
        seen = set()
        unique_docs = []
        
        for doc in documents:
            doc_hash = hash(doc['content'])
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _rerank_documents(self, documents: List[Dict], query: str) -> List[Dict]:
        """Rerank documents based on relevance score"""
        # Implement cross-encoder reranking here
        # For now, return documents sorted by score
        return sorted(documents, key=lambda x: x.get('score', 0), reverse=True)
