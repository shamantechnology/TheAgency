"""
Memstore methods used by tools
"""
import os
from forge.sdk.memory.memstore import ChromaMemStore
from ..forge_log import ForgeLogger

logger = ForgeLogger(__name__)

def add_ability_memory(task_id: str, document: str, ability_name: str) -> None:
    """
    Add ability output to memory
    """
    logger.info(f"🧠 Adding ability {ability_name} memory for task {task_id}")
    try:
        chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}/chromadb/"
        memory = ChromaMemStore(chromadb_path)
        memory.add(
            task_id=task_id,
            document=document,
            metadatas={
                "function": ability_name,
                "type": "ability"
            }
        )
    except Exception as err:
        logger.error(f"add_ability_memory failed: {err}")

def add_chat_memory(task_id: str, chat_msg: dict) -> None:
    """
    Add chat entry to memory
    """
    logger.info(f"🧠 Adding chat memory for task {task_id}")
    try:
        chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}/chromadb/"
        memory = ChromaMemStore(chromadb_path)
        memory.add(
            task_id=task_id,
            document=chat_msg["content"],
            metadatas={
                "role": chat_msg["role"],
                "type": "chat"
            }
        )
    except Exception as err:
        logger.error(f"add_chat_memory failed: {err}")

def add_website_memory(task_id: str, url: str, content: str) -> None:
    """
    Add website to memory
    """
    logger.info(f"🧠 Adding website memory {url} for task {task_id}")
    try:
        chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}/chromadb/"
        memory = ChromaMemStore(chromadb_path)
        memory.add(
            task_id=task_id,
            document=content,
            metadatas={
                "url": url,
                "type": "website"
            }
        )
    except Exception as err:
        logger.error(f"add_chat_memory failed: {err}")

def add_file_memory(task_id: str, file_name: str, content: str) -> None:
    """
    Add file to memory
    """
    logger.info(f"🧠 Adding file memory {file_name} for task {task_id}")
    try:
        chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}/chromadb/"
        memory = ChromaMemStore(chromadb_path)
        memory.add(
            task_id=task_id,
            document=content,
            metadatas={
                "filename": file_name,
                "type": "file"
            }
        )
    except Exception as err:
        logger.error(f"add_chat_memory failed: {err}")

def add_search_memory(task_id: str, query: str, search_results: str) -> str:
    """
    Add search results to memory and return doc id
    """
    logger.info(f"🧠 Adding search results for task {task_id}")
    try:
        chromadb_path = f"{os.getenv('AGENT_WORKSPACE')}/{task_id}/chromadb/"
        memory = ChromaMemStore(chromadb_path)
        doc_id = memory.add(
            task_id=task_id,
            document=search_results,
            metadatas={
                "query": query,
                "type": "search"
            }
        )

        return doc_id
    except Exception as err:
        logger.error(f"add_search_memory failed: {err}")