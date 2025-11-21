"""RAG pipeline orchestrator"""
from typing import List, Dict
from .agents.retrieval_agent import RetrievalAgent
from .agents.synthesis_agent import SynthesisAgent
from .embeddings.embedding_service import EmbeddingService

class RAGPipeline:
    def __init__(self, config: Dict):
        self.embedding_service = EmbeddingService(config['embeddings'])
        self.retrieval_agent = RetrievalAgent(config['retrieval'])
        self.synthesis_agent = SynthesisAgent(config['synthesis'])
    
    async def query(self, question: str, top_k: int = 5) -> Dict:
        """Process RAG query"""
        # Generate query embedding
        query_embedding = await self.embedding_service.embed(question)
        
        # Retrieve relevant documents
        documents = await self.retrieval_agent.retrieve(
            query_embedding, 
            top_k=top_k
        )
        
        # Generate answer
        answer = await self.synthesis_agent.generate(
            question=question,
            context=documents
        )
        
        return {
            'answer': answer['text'],
            'sources': documents,
            'confidence': answer['confidence']
        }
