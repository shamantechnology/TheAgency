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
    name="add_file_to_memory",
    description="Add content of file to your memory. " \
        "This should be ran before using 'read_file_from_memory' or 'mem_qna",
    parameters=[
        {
            "name": "file_name",
            "description": "File name of file to add",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def add_file_to_memory(agent, task_id: str, file_name: str) -> str:
    logger.info(f"🧠 Adding {file_name} to memory for task {task_id}")
    try:
        open_file = agent.workspace.read(task_id=task_id, path=file_name)
        open_file_str = open_file.decode()

        add_file_memory(
            task_id,
            file_name,
            open_file_str
        )
    except Exception as err:
        logger.error(f"add_file_memory failed: {err}")
        raise err
    
    return f"{file_name} added to memory"

@ability(
    name="add_website_to_memory",
    description="Get website and store content in your memory",
    parameters=[
        {
            "name": "url",
            "description": "Website's url",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def add_website_to_memory(agent, task_id: str, url: str) -> str:
    """
    add_website_to_memory

    takes a string URL and returns HTML and converts it to text
    stores converted text in vector database
    VSDB: chromadb
    """
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
        
    except Exception as err:
        logger.error(f"add_website_to_memory failed: {err}")
        raise err

    return f"Added {url} to memory"


@ability(
    name="read_file_from_memory",
    description="Read file stored in your memory",
    parameters=[
        {
            "name": "file_name",
            "description": "File name of file to add",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def read_file_from_memory(agent, task_id: str, file_name: str) -> str:
    try:
        # find doc in chromadb
        cwd = agent.workspace.get_cwd_path(task_id)
        chroma_dir = f"{cwd}/chromadb/"

        memory = ChromaMemStore(chroma_dir)
        memory_resp = memory.query(
            task_id=task_id,
            query="",
            filters={
                "filename": file_name
            }
        )

        # get the most relevant document and shrink to 50
        if len(memory_resp["documents"][0]) > 0:
            mem_doc = memory_resp["documents"][0][0]
            if(len(mem_doc) > MAX_OUT_SIZE):
                mem_doc = "This document is too long, use the ability 'mem_file_qna' to access it."
            else:
                mem_doc = memory_resp["documents"][0][0][:MAX_OUT_SIZE]
        else:
            # tell ai to use 'add_file_memory'
            mem_doc = "File not found in memory. Add the file with ability 'add_file_memory'"
    except Exception as err:
        logger.error(f"read_file_from_memory failed: {err}")
        raise err
    
    return mem_doc

@ability(
    name="website_from_memory",
    description="Search for a website in memory",
    parameters=[
        {
            "name": "url",
            "description": "URL of website",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def website_from_memory(agent, task_id: str, url: str) -> str:
    try:
        # find doc in chromadb
        cwd = agent.workspace.get_cwd_path(task_id)
        chroma_dir = f"{cwd}/chromadb/"

        memory = ChromaMemStore(chroma_dir)
        memory_resp = memory.query(
            task_id=task_id,
            query="",
            filters={
                "url": url
            }
        )

        # get the most relevant document and shrink to 50
        if len(memory_resp["documents"][0]) > 0:
            mem_doc = memory_resp["documents"][0][0]
            if(len(mem_doc) > MAX_OUT_SIZE):
                mem_doc = "This document is too long, use the ability 'mem_website_qna' to access it."
            else:
                mem_doc = memory_resp["documents"][0][0][:MAX_OUT_SIZE]
        else:
            # tell ai to use 'add_file_memory'
            mem_doc = "File not found in memory. Add the file with ability 'add_file_memory'"
    except Exception as err:
        logger.error(f"website_from_memory failed: {err}")
        raise err
    
    return mem_doc

ability(
    name="chat_from_memory",
    description="Search for chat history",
    parameters=[
        {
            "name": "query",
            "description": "Question used for chat search",
            "type": "string",
            "required": True,
        },
        {
            "name": "role",
            "description": "Role you are searching for in chat history",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def chat_from_memory(agent, task_id: str, query: str, role: str) -> str:
    try:
        # find doc in chromadb
        cwd = agent.workspace.get_cwd_path(task_id)
        chroma_dir = f"{cwd}/chromadb/"

        memory = ChromaMemStore(chroma_dir)
        memory_resp = memory.query(
            task_id=task_id,
            query=query,
            filters={
                "role": role
            }
        )

        # get the most relevant document and shrink to 50
        if len(memory_resp["documents"][0]) > 0:
            mem_doc = memory_resp["documents"][0][0]
            if(len(mem_doc) > MAX_OUT_SIZE):
                mem_doc = "This document is too long, use the ability 'mem_chat_qna' to access it."
            else:
                mem_doc = memory_resp["documents"][0][0][:MAX_OUT_SIZE]
        else:
            # tell ai to use 'add_file_memory'
            mem_doc = "File not found in memory. Add the file with ability 'add_file_memory'"
    except Exception as err:
        logger.error(f"chat_from_memory failed: {err}")
        raise err
    
    return mem_doc

# @ability(
#     name="mem_search",
#     description="query your memory for relevant stored documents",
#     parameters=[
#         {
#             "name": "query",
#             "description": "search query",
#             "type": "string",
#             "required": True,
#         }
#     ],
#     output_type="str",
# )
# async def mem_search(agent, task_id: str, query: str) -> str:
#     mem_doc = "No documents found"

#     try:
#         # find doc in chromadb
#         cwd = agent.workspace.get_cwd_path(task_id)
#         chroma_dir = f"{cwd}/chromadb/"

#         memory = ChromaMemStore(chroma_dir)
#         memory_resp = memory.query(
#             task_id=task_id,
#             query=query
#         )

#         # get the most relevant document and shrink
#         mem_doc = memory_resp["documents"][0][0]
#         if(len(mem_doc) > 187):
#             mem_doc = "This document is too long, use the ability 'mem_qna' to access it."
#         else:
#             mem_doc = memory_resp["documents"][0][0][:187]
#     except Exception as err:
#         logger.error(f"mem_search failed: {err}")
#         raise err
    
#     return mem_doc

@ability(
    name="mem_file_qna",
    description="Ask a question about a file stored in memory",
    parameters=[
        {
            "name": "file_name",
            "description": "name of file",
            "type": "string",
            "required": True,
        },
        {
            "name": "query",
            "description": "question about file",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def mem_file_qna(agent, task_id: str, file_name: str, query: str):
    mem_doc = "No documents found"
    try:
        aimem = AIMemory(
            workspace=agent.workspace,
            task_id=task_id,
            query=query,
            file_name=file_name,
            doc_type="file",
            model="gpt-3.5-turbo-16k"
        )

        aimem.get_doc()

        if aimem.relevant_doc:
            mem_doc = await aimem.query_doc_ai()
    except Exception as err:
        logger.error(f"mem_file_qna failed: {err}")
        raise err
    
    return mem_doc

@ability(
    name="mem_chat_qna",
    description="Ask a question about chat history per role stored in memory",
    parameters=[
        {
            "name": "chat_role",
            "description": "chat role - either 'user', 'system' or 'assistant'",
            "type": "string",
            "required": True,
        },
        {
            "name": "query",
            "description": "question about conversation",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def mem_chat_qna(agent, task_id: str, chat_role: str, query: str):
    mem_doc = "No documents found"
    try:
        aimem = AIMemory(
            workspace=agent.workspace,
            task_id=task_id,
            query=query,
            chat_role=chat_role,
            doc_type="chat",
            model="gpt-3.5-turbo-16k"
        )

        if aimem.get_doc():
            mem_doc = await aimem.query_doc_ai()
    except Exception as err:
        logger.error(f"mem_chat_qna failed: {err}")
        raise err
    
    return mem_doc

@ability(
    name="mem_website_qna",
    description="Ask a question about a website stored in memory",
    parameters=[
        {
            "name": "url",
            "description": "url of website",
            "type": "string",
            "required": True,
        },
        {
            "name": "query",
            "description": "question about website",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def mem_file_qna(agent, task_id: str, url: str, query: str):
    mem_doc = "No documents found"
    try:
        aimem = AIMemory(
            workspace=agent.workspace,
            task_id=task_id,
            query=query,
            url=url,
            doc_type="website",
            model="gpt-3.5-turbo-16k"
        )

        if aimem.get_doc():
            mem_doc = await aimem.query_doc_ai()
    except Exception as err:
        logger.error(f"mem_qna failed: {err}")
        raise err
    
    return mem_doc

@ability(
    name="mem_all_qna",
    description="Ask a question about everything in memory",
    parameters=[
        {
            "name": "query",
            "description": "question about website",
            "type": "string",
            "required": True,
        }
    ],
    output_type="str",
)
async def mem_file_qna(agent, task_id: str, query: str):
    mem_doc = "No documents found"
    try:
        aimem = AIMemory(
            workspace=agent.workspace,
            task_id=task_id,
            query=query,
            doc_type="all",
            model="gpt-3.5-turbo-16k"
        )

        if aimem.get_doc():
            mem_doc = await aimem.query_doc_ai()
    except Exception as err:
        logger.error(f"mem_qna failed: {err}")
        raise err
    
    return mem_doc
