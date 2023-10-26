"""
Memstore methods used by tools
"""
import os
from typing import Any
from .chroma_memstore import ChromaMemStore
from .weaviate_memstore import WeaviateMemstore
from ..forge_log import ForgeLogger

logger = ForgeLogger(__name__)

def add_ability_memory(
    task_id: str, 
    output: str, 
    ability_name: str,
    memstore: Any
    ) -> None:
    """
    Add ability output to memory
    """
    logger.info(f"🧠 Adding ability {ability_name} memory for task {task_id}")
    try:
        if isinstance(memstore, ChromaMemStore):
            memstore.add(
                task_id=task_id,
                document=output,
                metadatas={
                    "function": ability_name,
                    "type": "ability"
                }
            )
        elif isinstance(memstore, WeaviateMemstore):
            memstore.add_data_obj("ability", {
                "function": ability_name,
                "content": output,
                "task_id": task_id
            })

    except Exception as err:
        logger.error(f"add_ability_memory failed: {err}")

def add_chat_memory(
    task_id: str, 
    chat_msg: dict,
    memstore: Any
) -> None:
    """
    Add chat entry to memory
    """
    logger.info(f"🧠 Adding chat memory for task {task_id}")
    try:
        if isinstance(memstore, ChromaMemStore):
            memstore.add(
                task_id=task_id,
                document=chat_msg["content"],
                metadatas={
                    "role": chat_msg["role"],
                    "type": "chat"
                }
            )
        elif isinstance(memstore, WeaviateMemstore):
            chat_msg["task_id"] = task_id
            memstore.add_data_obj("chat", chat_msg)

    except Exception as err:
        logger.error(f"add_chat_memory failed: {err}")

def add_website_memory(
    task_id: str,
    url: str,
    content: str,
    memstore: Any
) -> None:
    """
    Add website to memory
    """
    logger.info(f"🧠 Adding website memory {url} for task {task_id}")
    try:
        if isinstance(memstore, ChromaMemStore):
            memstore.add(
                task_id=task_id,
                document=content,
                metadatas={
                    "url": url,
                    "type": "website"
                }
            )
        elif isinstance(memstore, WeaviateMemstore):
            memstore.add_data_obj("website", {
                "url": url,
                "content": content,
                "task_id": task_id
            })
    except Exception as err:
        logger.error(f"add_website_memory failed: {err}")

def add_file_memory(
    task_id: str,
    file_name: str,
    content: str,
    memstore: Any
) -> None:
    """
    Add file to memory
    """
    logger.info(f"🧠 Adding file memory {file_name} for task {task_id}")
    try:
        if isinstance(memstore, ChromaMemStore):
            memstore.add(
                task_id=task_id,
                document=content,
                metadatas={
                    "filename": file_name,
                    "type": "file"
                }
            )
        elif isinstance(memstore, WeaviateMemstore):
            memstore.add_data_obj("file",{
                "filename": file_name,
                "content": content,
                "task_id": task_id
            })
    except Exception as err:
        logger.error(f"add_file_memory failed: {err}")

# def add_search_memory(
#     task_id: str,
#     query: str, 
#     search_results: str,
#     memstore: Any
# ) -> str:
#     """
#     Add search results to memory and return doc id
#     """
#     logger.info(f"🧠 Adding search results for task {task_id}")
#     try:
#         if isinstance(memstore, ChromaMemStore):
#             doc_id = memstore.add(
#                 task_id=task_id,
#                 document=search_results,
#                 metadatas={
#                     "query": query,
#                     "type": "search"
#                 }
#             )

#             return doc_id
#         elif isinstance(memstore, WeaviateMemstore):
#             uuid = memstore.add_data_obj("search results",{
#                 "query": query,
#                 "results": search_results,
#                 "task_id": task_id
#             })

#             return uuid
#     except Exception as err:
#         logger.error(f"add_search_memory failed: {err}")