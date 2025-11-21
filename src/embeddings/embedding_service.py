"""OpenAI embedding service"""
import openai
from typing import List
import numpy as np

class EmbeddingService:
    def __init__(self, config: Dict):
        self.model = config.get('model', 'text-embedding-ada-002')
        openai.api_key = config.get('api_key')
    
    async def embed(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        response = openai.Embedding.create(
            model=self.model,
            input=text
        )
        return np.array(response['data'][0]['embedding'])
    
    async def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for batch of texts"""
        response = openai.Embedding.create(
            model=self.model,
            input=texts
        )
        return [np.array(item['embedding']) for item in response['data']]
