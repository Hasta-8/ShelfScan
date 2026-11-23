"""Wikimedia Commons downloads for the demo samples and the failure gallery."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "ShelfScan/0.3 (portfolio demo; research script)"}


def _get(url: str) -> bytes:
    for attempt in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read()
        except urllib.error.HTTPError as exc:  # Commons rate-limits bursts (429)
            if exc.code != 429 or attempt == 3:
                raise
            time.sleep(20 * (attempt + 1))
    raise RuntimeError("unreachable")


def thumbnail(title: str, width: int = 1920) -> bytes:
    """The file's Commons thumbnail at a standard width (1920 px; other widths get 429s)."""
    q = urllib.parse.urlencode(
        {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "imageinfo",
            "iiprop": "url",
            "iiurlwidth": width,
        }
    )
    page = next(
        iter(
            json.loads(_get("https://commons.wikimedia.org/w/api.php?" + q))["query"][
                "pages"
            ].values()
        )
    )
    return _get(page["imageinfo"][0]["thumburl"])
