import os
from typing import Any, Optional, Tuple
import openai
from dotenv import load_dotenv


load_dotenv()

class OpenAIClient:
    def __init__(self, 
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4")
        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
            
        self.client = openai.OpenAI(**client_kwargs)
    
    def get_structured_response(self, 
                              task: str, 
                              system_prompt: str,
                              response_format: Any,
                              model: Optional[str] = None,
                              temperature: float = 0.0,
                              max_tokens: Optional[int] = None) -> Tuple[Any, bool]:
        model_name = model or self.model
        
        try:
            response = self.client.chat.completions.parse(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task}
                ],
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response, True
            
        except Exception as e:
            return {"error": f"{e}"}, False
    
def create_openai_client(local: bool = False) -> OpenAIClient:
    if local:
        base_url = os.getenv("LOCAL_OPENAI_BASE_URL", "http://localhost:8000/v1")
        api_key = os.getenv("LOCAL_OPENAI_API_KEY", "local-key")
        model = os.getenv("LOCAL_OPENAI_MODEL", "gpt-4")
    else:
        base_url = None
        api_key = None
        model = "gpt-4"
    
    return OpenAIClient(
        api_key=api_key,
        base_url=base_url,
        model=model
    )