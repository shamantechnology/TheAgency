"""
Searching with googleapi
"""
from typing import List

import os
import json
import time
from itertools import islice

from duckduckgo_search import DDGS

from ..forge_log import ForgeLogger
from .registry import ability

DUCKDUCKGO_MAX_ATTEMPTS = 3

logger = ForgeLogger(__name__)

@ability(
    name="web_search",
    description="Search the internet using DuckDuckGo",
    parameters=[
        {
            "name": "query",
            "description": "detailed search query",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def web_search(agent, task_id: str, query: str) -> str:
    try:
        search_results = []
        attempts = 0
        num_results = 8

        while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
            if not query:
                return json.dumps(search_results)

            results = DDGS().text(query)
            search_results = list(islice(results, num_results))

            if search_results:
                break

            time.sleep(1)
            attempts += 1

        results = json.dumps(search_results, ensure_ascii=False, indent=4)
        
        if isinstance(results, list):
            safe_message = json.dumps(
                [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
            )
        else:
            safe_message = results.encode("utf-8", "ignore").decode("utf-8")

        return safe_message
    except Exception as err:
        logger.error(f"google_search failed: {err}")
        raise err