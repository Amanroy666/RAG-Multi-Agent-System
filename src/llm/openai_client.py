"""OpenAI LLM client"""
import openai
from typing import Dict, List

class OpenAIClient:
    def __init__(self, api_key: str, model: str = "gpt-4"):
        openai.api_key = api_key
        self.model = model
    
    async def generate(self, prompt: str, max_tokens: int = 500) -> Dict:
        """Generate response from LLM"""
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        return {
            'text': response.choices[0].message.content,
            'tokens_used': response.usage.total_tokens
        }
