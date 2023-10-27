"""
Memory tool for document search
"""
# from typing import List

import re

from ..forge_log import ForgeLogger
from .registry import ability
from forge.sdk.memory.chroma_memstore import ChromaMemStore
from forge.sdk.memory.weaviate_memstore import WeaviateMemstore
from forge.sdk.errors import *

from forge.sdk.memory.memstore_tools import (
    add_website_memory,
    add_file_memory
)

from ..ai_memory import AIMemory

from .web_content import (
    open_page_in_browser,
    scrape_text_with_selenium
)

from selenium.common.exceptions import WebDriverException

logger = ForgeLogger(__name__)

# change if you want more text or data sent to agent
MAX_OUT_SIZE = 150

class BrowsingError(CommandExecutionError):
    """An error occurred while trying to browse the page"""
def extract_text_from_website(url: str) -> str:
    """
    Extract using selenium and bs4
    """
    driver = None
    text = ""

    try:
        driver = open_page_in_browser(url)
        text = scrape_text_with_selenium(driver)
    except WebDriverException as e:
        # These errors are often quite long and include lots of context.
        # Just grab the first line.
        msg = e.msg.split("\n")[0]
        if "net::" in msg:
            raise BrowsingError(
                f"A networking error occurred while trying to load the page: "
                + re.sub(r"^unknown error: ", "", msg)
            )
        raise CommandExecutionError(msg)
    finally:
        if driver:
            driver.quit()
    
    return text

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
                open_file_str,
                agent.memstore
            )
        
            return f"{file_name} added to memory"
        except Exception as err:
            logger.error(f"Adding {file_name} to memory failed: {err}")
            return f"Adding {file_name} to memory failed: {err}"
        
    elif url:
        try:
            web_content = extract_text_from_website(url)

            add_website_memory(
                task_id,
                url,
                web_content,
                agent.memstore
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
        memory = agent.memstore
        if isinstance(memory, ChromaMemStore):
            # find doc in chromadb
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
        elif isinstance(memory, WeaviateMemstore):
            # find doc in weaviate
            if file_name:
                data_class = "file"
                query = file_name
            elif url:
                data_class = "website"
                query = url
            elif chat_role:
                data_class = "chat"
                query = chat_role
            elif doc_id or qall:
                pass
            else:
                logger.error("No arguments found")
                data_class = None
                mem_doc = "No arguments found. Please specify one of those arguments"

            if data_class:
                logger.info(f"find doc by data_class {data_class}")
                resp = memory.get_data_obj(
                    task_id,
                    data_class,
                    query
                )

                if len(resp) > 0:
                    if len(resp) > 1:
                        mem_doc = "Too many results. Use the ability 'mem_qna'"
                    elif len(resp[0]["content"]) > MAX_OUT_SIZE:
                        mem_doc = "This document is too long, use the ability 'mem_qna' to access it."
                    else:
                        mem_doc = resp[0]["content"]
                else:
                    mem_doc = "No documents found"
                    
            else:
                if doc_id:
                    found_id = False
                    for data_class in agent.memstore.data_class_names:
                        resp = memory.get_obj_by_id(
                            task_id,
                            data_class,
                            doc_id
                        )

                        if len(resp) > 0:
                            if len(resp) > 1:
                                mem_doc = "Too many results. Use the ability 'mem_qna'"
                            elif len(resp[0]["content"]) > MAX_OUT_SIZE:
                                mem_doc = "This document is too long, use the ability 'mem_qna' to access it."
                            else:
                                mem_doc = resp[0]["content"]

                            found_id = True
                            break   
                            

                    if not found_id:
                        mem_doc = "No documents found"

                elif qall:
                    found_something = False
                    for data_class in agent.memstore.data_class_names:
                        resp = memory.get_data_obj(
                            task_id,
                            data_class,
                            qall
                        )

                        if len(resp) > 0:
                            if len(resp) > 1:
                                mem_doc = "Too many results. Use the ability 'mem_qna'"
                            elif len(resp[0]["content"]) > MAX_OUT_SIZE:
                                mem_doc = "This document is too long, use the ability 'mem_qna' to access it."
                            else:
                                mem_doc = resp[0]["content"]
                            
                            found_something = True
                            break

                    if not found_something:
                        mem_doc = "No documents found"
        else:
            logger.error(f"No memstore found. Please supply a memstore to use.")
            raise AttributeError
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
    
    if file_name:
        doc_type = "file"
    elif chat_role:
        doc_type = "chat"
    elif url:
        doc_type = "website"
    elif doc_id:
        doc_type = "doc_id"
    elif qall:
        doc_type = "all"
    else:
        doc_type = None

    try:
        if doc_type:
            if isinstance(agent.memstore, ChromaMemStore):
            
                aimem = AIMemory(
                    workspace=agent.workspace,
                    task_id=task_id,
                    query=query,
                    memstore=agent.memstore,
                    all_query=qall,
                    doc_id=doc_id,
                    file_name=file_name,
                    chat_role=chat_role,
                    url=url,
                    doc_type=doc_type,
                    model="gpt-3.5-turbo-16k"
                )
            

                if aimem.get_doc():
                    mem_doc = await aimem.query_doc_ai()
                else:
                    logger.error("get_doc failed")
                    mem_doc = "No documents found"
            elif isinstance(agent.memstore, WeaviateMemstore):
                aimem = AIMemory(
                    workspace=agent.workspace,
                    task_id=task_id,
                    query=query,
                    memstore=agent.memstore,
                    memstore_classes=agent.memstore.data_class_names,
                    all_query=qall,
                    doc_id=doc_id,
                    file_name=file_name,
                    chat_role=chat_role,
                    url=url,
                    doc_type=doc_type,
                    model="gpt-3.5-turbo-16k"
                )


                mem_doc = await aimem.query_doc_ai()
                if mem_doc == "":
                    mem_doc = "No answer found"
        else:
            logger.error("No paramter to search by given.")
            mem_doc = "No paramter to search by given. Please provide 'file_name', 'chat_role', 'url', 'doc_id' or 'qall' parameter to search by."

    except Exception as err:
        logger.error(f"mem_qna failed: {err}")
        raise err
    
    return mem_doc