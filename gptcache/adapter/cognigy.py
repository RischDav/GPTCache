import requests
from gptcache.adapter.adapter import adapt
from gptcache.adapter.base import BaseCacheLLM
from gptcache.manager.scalar_data.base import Answer, DataType

class Cognigy(BaseCacheLLM):

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        url = "placeholder"
        
        payload = {
            "userId": llm_kwargs.get("userId"),
            "sessionId": llm_kwargs.get("sessionId"),
            "text": llm_kwargs.get("text"),
            "data": llm_kwargs.get("data", {})
        }
        
        # 1. VERSUCH
        response = requests.post(url, json=payload)
        response.raise_for_status()
        res_json = response.json()
        
        bot_text = res_json.get("text", "")
        status = res_json.get("data", {}).get("status")
        
        # Prüfen ob terminiert
        if "leeva" in bot_text.lower() or "repeat" in bot_text.lower() or status != "termination":
            print(f"[GPTCache-Adapter] Turn-1 Limitation erkannt: '{bot_text}'. Starte automatischen Retry...")
            
            # 2. VERSUCH
            response = requests.post(url, json=payload)
            response.raise_for_status()
            res_json = response.json()
            
        # Rückgabe der echten, finalen Antwort an den Cache
        return res_json

    @staticmethod
    def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs):
        """Cacht das Ergebnis nur, wenn es eine erfolgreiche finale Antwort ist."""
        text_to_cache = llm_data.get("text", "")
        status = llm_data.get("data", {}).get("status")
        
        if status == "termination" and text_to_cache:
            update_cache_func(Answer(text_to_cache, DataType.STR))
            
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs):
        if "text" in kwargs and "messages" not in kwargs:
            kwargs["messages"] = [{"content": kwargs.get("text")}]

        def cache_data_convert(cache_data):
            return {
                "text": cache_data,
                "data": {"status": "termination"},
                "outputStack": [{"text": cache_data, "data": {}, "source": "bot"}],
                "userId": kwargs.get("userId"),
                "sessionId": kwargs.get("sessionId")
            }

        return adapt(
            cls._llm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )