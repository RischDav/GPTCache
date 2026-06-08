import argparse
import os
import zipfile
from typing import Optional, Dict, Any

from gptcache import cache
from gptcache.adapter.api import (
    init_similar_cache,
    init_similar_cache_from_config,
)
from gptcache.utils import import_fastapi, import_pydantic
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

# Fix für ältere Transformers-Versionen
if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
    PreTrainedTokenizerBase.encode_plus = PreTrainedTokenizerBase.__call__

import_fastapi()
import_pydantic()

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel

# Importiere deinen angepassten SconBot-Adapter
from gptcache.adapter.cognigy import Cognigy

app = FastAPI(title="GPTCache SconBot Server")
cache_dir = ""
cache_file_key = ""
USE_CACHE = False  # Globale Variable für den Cache-Bypass


# Das saubere Datenmodell für die Schwarz-Gruppe
class SconRequest(BaseModel):
    userId: str
    sessionId: str
    text: str
    endpointUrl: str
    data: Optional[Dict[str, Any]] = {}


@app.get("/")
async def hello():
    return "hello gptcache scon-bot server"


# Der zentrale Webhook für Cognigy-Anfragen
@app.post("/scon-webhook")
async def scon_webhook(payload: SconRequest):
    global USE_CACHE
    try:
        # Wenn USE_CACHE False ist, soll der Cache übersprungen werden (cache_skip = True)
        cache_skip_flag = not USE_CACHE
        
        # Übergabe der Daten und des Cache-Bypass-Flags an den Cognigy-Adapter
        response_data = Cognigy.create(
            userId=payload.userId,
            sessionId=payload.sessionId,
            text=payload.text,
            prompt=payload.text, 
            endpointUrl=payload.endpointUrl,
            data=payload.data if payload.data else {},
            cache_skip=cache_skip_flag
        )
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Admin-Endpoint zum Download des Caches als ZIP
@app.get("/cache_file")
async def get_cache_file(key: str = "") -> FileResponse:
    global cache_dir, cache_file_key
    if cache_dir == "":
        raise HTTPException(status_code=403, detail="the cache_dir was not specified")
    if cache_file_key == "":
        raise HTTPException(status_code=403, detail="the cache file key was not specified")
    if cache_file_key != key:
        raise HTTPException(status_code=403, detail="the cache file key is wrong")
    
    zip_filename = cache_dir + ".zip"
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(cache_dir):
            for file in files:
                zipf.write(os.path.join(root, file))
    return FileResponse(zip_filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--host", default="localhost", help="the hostname to listen on")
    parser.add_argument("-p", "--port", type=int, default=8000, help="the port to listen on")
    parser.add_argument("-d", "--cache-dir", default="gptcache_data", help="the cache data dir")
    parser.add_argument("-k", "--cache-file-key", default="", help="the cache file key")
    parser.add_argument("-f", "--cache-config-file", default=None, help="the cache config file")
    
    # Ja/Nein-Flag für die Cache-Erlaubnis
    parser.add_argument(
        "-usecache",
        "--usecache",
        action="store_true",
        help="decide if the using of the cache is allowed",
    )

    args = parser.parse_args()
    global cache_dir, cache_file_key, USE_CACHE

    # CLI-Wert global sichern
    USE_CACHE = args.usecache
    print(f"Server gestartet. Caching aktiv: {USE_CACHE}")

    # Cache-Infrastruktur initialisieren
    if args.cache_config_file:
        init_conf = init_similar_cache_from_config(config_dir=args.cache_config_file)
        cache_dir = init_conf.get("storage_config", {}).get("data_dir", "")
    else:
        init_similar_cache(args.cache_dir)
        cache_dir = args.cache_dir
        
    cache_file_key = args.cache_file_key

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()