import json
import re
import time
import threading
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REQUEST_DELAY = 0.3
MAX_RETRIES = 5
HEADERS = {
    "User-Agent": "GraphChecking/1.0 (research project)"
}

_session = None
_session_lock = threading.Lock()


def _get_session():
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                retry_strategy = Retry(
                    total=3,
                    backoff_factor=0.3,
                    status_forcelist=[429, 500, 502, 503, 504],
                    allowed_methods=["HEAD", "GET", "OPTIONS"]
                )
                adapter = HTTPAdapter(
                    pool_connections=10,
                    pool_maxsize=10,
                    max_retries=retry_strategy
                )
                _session.mount('https://', adapter)
                _session.headers.update(HEADERS)
    return _session


def _api_request(url, params, headers=None):
    session = _get_session()
    for attempt in range(MAX_RETRIES):
        if attempt > 0:
            time.sleep(REQUEST_DELAY * attempt)
        try:
            resp = session.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


class MPCServer:
    def __init__(self):
        self._connected = False

    def connect(self):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def search(self, keyword, language="en", top_k=5):
        self._ensure_connected()
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": keyword,
            "srlimit": top_k,
            "converttitles": True,
        }
        data = _api_request(
            f"https://{language}.wikipedia.org/w/api.php",
            params=params,
        )
        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({
                "id": item["pageid"],
                "title": item["title"],
                "snippet": re.sub(r"<[^>]+>", "", item.get("snippet", "")),
            })
        return results

    def fetch(self, page_id, language="en"):
        self._ensure_connected()
        params = {
            "action": "parse",
            "format": "json",
            "pageid": page_id,
            "prop": "text",
        }
        data = _api_request(
            f"https://{language}.wikipedia.org/w/api.php",
            params=params,
        )
        return data.get("parse", {}).get("text", {}).get("*", "")

    def fetch_clean(self, page_id, language="en"):
        self._ensure_connected()
        params = {
            "action": "query",
            "format": "json",
            "pageids": page_id,
            "prop": "extracts",
            "explaintext": True,
            "exintro": False,
        }
        data = _api_request(
            f"https://{language}.wikipedia.org/w/api.php",
            params=params,
        )
        pages = data.get("query", {}).get("pages", {})
        page = pages.get(str(page_id), {})
        return page.get("extract", "")

    def _ensure_connected(self):
        if not self._connected:
            self.connect()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
