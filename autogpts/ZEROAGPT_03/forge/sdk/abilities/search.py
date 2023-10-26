"""
Searching with googleapi
"""
from typing import List

import os
import json
import time
from itertools import islice
import requests
from bs4 import BeautifulSoup
import lxml

from duckduckgo_search import DDGS

# from ..memory.memstore_tools import add_search_memory
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
        num_results = 6

        while attempts < DUCKDUCKGO_MAX_ATTEMPTS:
            if not query:
                return json.dumps(search_results)

            results = DDGS().text(query)
            search_results = list(islice(results, num_results))

            if search_results:
                break

            time.sleep(1)
            attempts += 1

        cut_search_results = []
        for res in search_results:
            res["body"] = res["body"][:20]
            cut_search_results.append(res)

        results = json.dumps(cut_search_results, ensure_ascii=False)
        
        if isinstance(results, list):
            # cut down body to 10 words
            safe_message = json.dumps(
                [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
            )
        else:
            safe_message = results.encode("utf-8", "ignore").decode("utf-8")

        # save full list
        # add_search_memory(
        #     task_id,
        #     query,
        #     safe_message
        # )

        # return top 3 results to save tokens
        return safe_message

    except Exception as err:
        logger.error(f"google_search failed: {err}")
        raise err

# @ability(
#     name="web_search2",
#     description="Search the internet using Google",
#     parameters=[
#         {
#             "name": "query",
#             "description": "detailed search query",
#             "type": "string",
#             "required": True,
#         }
#     ],
#     output_type="str",
# )
# async def web_search2(agent, task_id: str, query: str) -> str:
#     params = {
#         "q": query,
#         "hl": "en",
#         "gl": "us",
#         "start": 0
#     }

#     headers = {
#         "User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
#     }

#     page_limit = 10
#     page_num = 0

#     data = []

#     while True:
#         page_num += 1
            
#         html = requests.get(
#             "https://www.google.com/search",
#             params=params,
#             headers=headers,
#             timeout=30
#         )

#         soup = BeautifulSoup(html.text, 'lxml')
        
#         for result in soup.select(".tF2Cxc"):
#             title = result.select_one(".DKV0Md").text
#             try:
#                 snippet = result.select_one(".lEBKkf span").text
#             except:
#                 snippet = None
            
#             links = result.select_one(".yuRUbf a")["href"]
        
#             data.append({
#                 "title": title,
#                 "snippet": snippet,
#                 "links": links
#             })

#         if page_num == page_limit:
#             break
#         if soup.select_one(".d6cvqb a[id=pnnext]"):
#             params["start"] += 10
#         else:
#             break
    
#     resp_json = json.dumps(data, ensure_ascii=False)
    
#     return resp_json