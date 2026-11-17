"""OpenRouter native REST calls: embeddings and chat (no hidden local model)."""
from __future__ import annotations
import json
import time
import httpx
import numpy as np
from .config import Settings


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = httpx.Client(timeout=httpx.Timeout(90, connect=15))

    def _post(self, endpoint: str, payload: dict) -> dict:
        if not self.settings.api_key or self.settings.api_key == 'replace_with_your_key':
            raise OpenRouterError('Set OPENROUTER_API_KEY in .env before indexing or chatting')
        headers = {
            'Authorization': f'Bearer {self.settings.api_key}',
            'Content-Type': 'application/json',
            'X-OpenRouter-Title': 'Clinical RAG Assistant',
        }
        for attempt in range(3):
            try:
                response = self.client.post(f'{self.settings.base_url}/{endpoint}', headers=headers, json=payload)
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(0.8 * 2**attempt)
                    continue
                response.raise_for_status()
                return response.json()
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                if attempt == 2:
                    raise OpenRouterError(f'OpenRouter network error: {type(exc).__name__}') from exc
                time.sleep(0.8 * 2**attempt)
            except httpx.HTTPStatusError as exc:
                raise OpenRouterError(f'OpenRouter request failed: HTTP {exc.response.status_code}; check key, model, and credits') from exc
        raise OpenRouterError('OpenRouter temporarily unavailable')

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        arrays = []
        for start in range(0, len(texts), 24):
            batch = texts[start:start + 24]
            data = self._post('embeddings', {'model': self.settings.embed_model, 'input': batch})
            try:
                ordered = sorted(data['data'], key=lambda x: x.get('index', 0))
                vectors = np.asarray([x['embedding'] for x in ordered], dtype=np.float32)
            except (KeyError, TypeError, ValueError) as exc:
                raise OpenRouterError('Embedding response was invalid') from exc
            if len(vectors) != len(batch) or vectors.ndim != 2:
                raise OpenRouterError('Unexpected embedding batch size or dimension')
            arrays.append(vectors)
        return np.vstack(arrays)

    def complete(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 800) -> str:
        data = self._post('chat/completions', {
            'model': self.settings.chat_model,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'messages': messages,
        })
        try:
            content = data['choices'][0]['message']['content']
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                return '\n'.join(x.get('text', '') for x in content if isinstance(x, dict)).strip()
            raise ValueError('Unexpected message format')
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise OpenRouterError('OpenRouter generated an invalid completion') from exc

    def complete_stream(self, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 800):
        if not self.settings.api_key or self.settings.api_key == 'replace_with_your_key':
            raise OpenRouterError('Set OPENROUTER_API_KEY in .env before indexing or chatting')
        headers = {
            'Authorization': f'Bearer {self.settings.api_key}',
            'Content-Type': 'application/json',
            'X-OpenRouter-Title': 'Clinical RAG Assistant',
        }
        payload = {
            'model': self.settings.chat_model,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'messages': messages,
            'stream': True,
        }
        try:
            with self.client.stream('POST', f'{self.settings.base_url}/chat/completions', headers=headers, json=payload) as response:
                if response.status_code >= 400:
                    raise OpenRouterError(f'OpenRouter request failed: HTTP {response.status_code}; check key, model, and credits')
                for line in response.iter_lines():
                    if not line or not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk['choices'][0].get('delta') or {}
                        text = delta.get('content') or ''
                    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise OpenRouterError('OpenRouter stream was invalid') from exc
                    if text:
                        yield text
        except OpenRouterError:
            raise
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            raise OpenRouterError(f'OpenRouter network error: {type(exc).__name__}') from exc
