"""
Memory tool for document search
"""
# from typing import List

# import os
from ..forge_log import ForgeLogger
from .registry import ability

from forge.sdk.memory.memstore import ChromaMemStore

from ..ai_memory import AIMemory

import requests
from bs4 import BeautifulSoup
from forge.sdk.memory.memstore_tools import (
    add_website_memory,
    add_file_memory
)

logger = ForgeLogger(__name__)

# change if you want more text or data sent to agent
MAX_OUT_SIZE = 150

@ability(
    name="add_to_memory",
    description="Add file or website to memory.",
    parameters=[
        {
            "name": "file_name",
            "description": "File name of file to add",
            "type": "string",
            "required": False,
        },
        {
            "name": "url",
            "description": "URL of website",
            "type": "string",
            "required": False,
        }
    ],
    output_type="str",
)
async def add_to_memory(
    agent,
    task_id: str,
    file_name: str = None,
    url: str = None
) -> str:
    
    if file_name:
        try:
            open_file = agent.workspace.read(task_id=task_id, path=file_name)
            open_file_str = open_file.decode()

            add_file_memory(
                task_id,
                file_name,
                open_file_str
            )
        
            return f"{file_name} added to memory"
        except Exception as err:
            logger.error(f"Adding {file_name} to memory failed: {err}")
            return f"Adding {file_name} to memory failed: {err}"
        
    elif url:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

            req = requests.get(
                url=url,
                headers=headers,
                timeout=5
            )

            html_soap = BeautifulSoup(req.text, "html.parser")
            html_text = html_soap.get_text()

            logger.info(f"Adding {url}\ncontent len {len(html_text)}")
            
            add_website_memory(
                task_id,
                url,
                html_text
            )
            
            return f"Added {url} to memory"
        except Exception as err:
            logger.error(f"add_website_to_memory failed: {err}")
            raise err
    
    return "No url or file_name arguments pass. Please specify one of the arguments."

@ability(
    name="read_from_memory",
    description="Return the contents of a file, chat, website, search results or anything stored in your memory",
    parameters=[
        {
            "name": "file_name",
            "description": "File name of file to add",
            "type": "string",
            "required": False,
        },
        {
            "name": "url",
            "description": "URL of website",
            "type": "string",
            "required": False,
        },
        {
            "name": "chat_role",
            "description": "Role you are searching for in chat history",
            "type": "string",
            "required": False,
        },
        {
            "name": "doc_id",
            "description": "doc_id for document in memory",
            "type": "string",
            "required": False
        },
        {
            "name": "qall",
            "description": "Search query for searching all of your memory",
            "type": "string",
            "required": False
        }

    ],
    output_type="str",
)
async def read_from_memory(
    agent,
    task_id: str,
    file_name: str = None,
    url: str = None,
    chat_role: str = None,
    doc_id: str = None,
    qall: str = None
) -> str:
    try:
        # find doc in chromadb
        cwd = agent.workspace.get_cwd_path(task_id)
        chroma_dir = f"{cwd}/chromadb/"

        memory = ChromaMemStore(chroma_dir)
        
        if file_name:
            memory_resp = memory.query(
                task_id=task_id,
                query="",
                filters={
                    "filename": file_name
                }
            )
        elif url:
            memory_resp = memory.query(
                task_id=task_id,
                query="",
                filters={
                    "url": url
                }
            )
        elif chat_role:
            memory_resp = memory.query(
                task_id=task_id,
                query="",
                filters={
                    "role": chat_role
                }
            )
        elif doc_id:
            memory_resp = memory.get(
                task_id=task_id,
                doc_ids=[doc_id]
            )
        elif qall:
            memory_resp = memory.query(
                task_id=task_id,
                query=qall
            )
        else:
            logger.error("No arguments found")
            mem_doc = "No arguments found. Please specify one of those arguments"
            return mem_doc

        # get the most relevant document and shrink to MAX_OUT_SIZE
        if len(memory_resp["documents"][0]) > 0:
            mem_doc = memory_resp["documents"][0][0]
            if(len(mem_doc) > MAX_OUT_SIZE):
                mem_doc = "This document is too long, use the ability 'mem_qna' to access it."
            else:
                mem_doc = memory_resp["documents"][0][0]
        else:
            # tell ai to use 'add_file_memory'
            mem_doc = "Nothing found in memory"
    except Exception as err:
        logger.error(f"read_from_memory failed: {err}")
        raise err
    
    return mem_doc

@ability(
    name="mem_qna",
    description="Ask a question about a file, chat, website or everything stored in memory",
    parameters=[
        {
            "name": "file_name",
            "description": "name of file",
            "type": "string",
            "required": False,
        },
        {
            "name": "chat_role",
            "description": "chat role - either 'user', 'system' or 'assistant'",
            "type": "string",
            "required": False,
        },
        {
            "name": "url",
            "description": "url of website",
            "type": "string",
            "required": False,
        },
        {
            "name": "doc_id",
            "description": "doc_id for document in memory",
            "type": "string",
            "required": False
        },
        {
            "name": "qall",
            "description": "Search query for searching all of your memory",
            "type": "string",
            "required": False
        },
        {
            "name": "query",
            "description": "question about memory",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def mem_qna(
    agent,
    task_id: str,
    query: str,
    file_name: str = None,
    chat_role: str = None,
    url: str = None,
    doc_id: str = None,
    qall: str = None
):
    mem_doc = "No documents found"
    try:
        if file_name:
            aimem = AIMemory(
                workspace=agent.workspace,
                task_id=task_id,
                query=query,
                file_name=file_name,
                doc_type="file",
                model="gpt-3.5-turbo-16k"
            )
        elif chat_role:
            aimem = AIMemory(
                workspace=agent.workspace,
                task_id=task_id,
                query=query,
                chat_role=chat_role,
                doc_type="chat",
                model="gpt-3.5-turbo-16k"
            )
        elif url:
            aimem = AIMemory(
                workspace=agent.workspace,
                task_id=task_id,
                query=query,
                url=url,
                doc_type="website",
                model="gpt-3.5-turbo-16k"
            )
        elif doc_id:
            aimem = AIMemory(
                workspace=agent.workspace,
                task_id=task_id,
                query=query,
                doc_id=doc_id,
                doc_type="doc_id",
                model="gpt-3.5-turbo-16k"
            )
        elif qall:
            aimem = AIMemory(
                workspace=agent.workspace,
                task_id=task_id,
                query=query,
                all_query=qall,
                doc_type="all",
                model="gpt-3.5-turbo-16k"
            )
        else:
            logger.error("No paramter to search by given.")
            mem_doc = "No paramter to search by given. Please provide 'file_name', 'chat_role', 'url', 'doc_id' or 'qall' parameter to search by."

        if aimem.get_doc():
            mem_doc = await aimem.query_doc_ai()
    except Exception as err:
        logger.error(f"mem_qna failed: {err}")
        raise err
    
    return mem_doc