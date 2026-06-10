import requests
from gptcache.adapter.adapter import adapt
from gptcache.adapter.base import BaseCacheLLM

class Cognigy(BaseCacheLLM):

    @classmethod
    def _llm_handler(cls, *llm_args, **llm_kwargs):
        url = llm_kwargs.get("endpointUrl")
        if not url:
            raise ValueError("[GPTCache-Adapter] 'endpointUrl' wurde nicht im Request übergeben!")
        
        payload = {
            "userId": llm_kwargs.get("userId"),
            "sessionId": llm_kwargs.get("sessionId"),
            "text": llm_kwargs.get("text"),
            "data": llm_kwargs.get("data", {})
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        res_json = response.json()
        
        bot_text = res_json.get("text", "")
        res_data = res_json.get("data") or {}
        status = res_data.get("status")
        
        if "leeva" in bot_text.lower() or "repeat" in bot_text.lower():
            response = requests.post(url, json=payload)
            response.raise_for_status()
            res_json = response.json()
            
        return res_json

    @staticmethod
    def _update_cache_callback(llm_data, update_cache_func, *args, **kwargs):
        # --- SICHERHEIT: Lokaler Import direkt in der Methode ---
        import json 
        
        if not llm_data:
            return llm_data
            
        text_to_cache = llm_data.get("text", "")
        res_data = llm_data.get("data") or {}
        end_time = res_data.get("endTime", None)
        
        if text_to_cache:
            cache_payload = {
                "text": text_to_cache,
                "endTime": end_time
            }
            update_cache_func(json.dumps(cache_payload))
            
        return llm_data

    @classmethod
    def create(cls, *args, **kwargs):
        if "text" in kwargs and "messages" not in kwargs:
            kwargs["messages"] = [{"content": kwargs.get("text")}]

        def cache_data_convert(cache_data):
            # --- SICHERHEIT: Lokaler Import direkt im Konverter ---
            import json 
            
            try:
                parsed_data = json.loads(cache_data)
                plain_text = parsed_data.get("text", "")
                end_time = parsed_data.get("endTime", None)
            except Exception:
                plain_text = cache_data
                end_time = None

            return {
                "text": plain_text,
                "data": {
                    "status": "termination",
                    "endTime": end_time
                },
                "outputStack": [{"text": plain_text, "data": {}, "source": "cache"}],
                "userId": kwargs.get("userId"),
                "sessionId": kwargs.get("sessionId"),
                "gptcache": True
            }

        return adapt(
            cls._llm_handler,
            cache_data_convert,
            cls._update_cache_callback,
            *args,
            **kwargs,
        )