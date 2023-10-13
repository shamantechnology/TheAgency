"""
Searching with googleapi
"""
from typing import List

import os
import json
import googleapiclient.discovery

from forge.sdk.memory.memstore_tools import add_ability_memory

from ..forge_log import ForgeLogger
from .registry import ability

logger = ForgeLogger(__name__)

@ability(
    name="google_search",
    description="Search the internet using Google",
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
async def google_search(agent, task_id: str, query: str) -> str:
    """
    Return list of snippets from google search
    """

    result =  "No results found"

    try:
        service = googleapiclient.discovery.build(
            "customsearch",
            "v1",
            developerKey=os.getenv("GOOGLE_API_KEY"))
                
        response = service.cse().list(
            q=query,
            cx=os.getenv("GOOGLE_CSE_ID")
        ).execute()

        resp_list = []
        for result in response["items"]:
            resp_list.append({
                "url": result["formattedUrl"],
                "snippet": result["snippet"]
            })

        try:
            result = json.dumps(resp_list)
        except json.JSONDecodeError as err:
            logger.error(f"json of result failed: {err}\n doing string")
            result = str(resp_list)
    except Exception as err:
        logger.error(f"google_search failed: {err}")
        raise err

    return result