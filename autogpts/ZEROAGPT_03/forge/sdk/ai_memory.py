"""
Load documents in AI and for QnA
"""
import os
from typing import Anyfrom typing import Any
from .forge_log import ForgeLogger
from .memory.chroma_memstore import ChromaMemStore
from .memory.weaviate_memstore import WeaviateMemstore
from . import Workspace
from . import chat_completion_request

logger = ForgeLogger(__name__)

class AIMemory:
    """
    Takes in query, finds relevant document in memstore
    then creates a prompt to query the document with query also

    Still limited on long data
    """
    def __init__(
        self,
        workspace: Workspace,
        task_id: str,
        query: str,
        doc_type: str,
        memstore: Any,
        memstore_classes: list = None,
        file_name: str = None,
        chat_role: str = None,
        url: str = None,
        doc_id: str = None,
        all_query: str = None,
        model: str = os.getenv("OPENAI_MODEL")
    ):

        self.workspace = workspace
        self.task_id = task_id
        self.query = query
        self.model = model
        self.file_name = file_name
        self.chat_role = chat_role
        self.url = url
        self.doc_id = doc_id
        self.all_query = all_query
        self.memstore = memstore
        self.memstore_classes = memstore_classes

        if doc_type not in ["file", "chat", "website", "doc_id", "all"]:
            logger.error(f"{doc_type} not found in allowed types. Defaulting to 'file' type")
            self.doc_type = "file"
        else:
            self.doc_type = doc_type

        self.chat = []
        self.relevant_docs = []
        self.prompt = None

    def get_doc(self) -> None:
        """
        Get document from VecStor
        """
        try:
            if isinstance(self.memstore, ChromaMemStore):
                # find doc in chromadb
                if self.doc_type == "file":
                    memory_resp = self.memstore.query(
                        task_id=self.task_id,
                        query="",
                        filters={
                            "filename": self.file_name
                        }
                    )
                elif self.doc_type == "chat":
                    memory_resp = self.memstore.query(
                        task_id=self.task_id,
                        query="",
                        filters={
                            "role": self.chat_role
                        }
                    )
                elif self.doc_type == "website":
                    memory_resp = self.memstore.query(
                        task_id=self.task_id,
                        query="",
                        filters={
                            "url": self.url
                        }
                    )
                elif self.doc_type == "doc_id":
                    memory_resp = self.memstore.get(
                        task_id=self.task_id,
                        doc_ids=[self.doc_id]
                    )
                elif self.doc_type == "all":
                    memory_resp = self.memstore.query(
                        task_id=self.task_id,
                        query=self.all_query
                    )

                if len(memory_resp["documents"]) > 0:
                    logger.info(
                        f"Relevant docs found! Doc count: {len(memory_resp['documents'])}")
                    
                    # need to add in chucking up of large docs
                    for i in range(len(memory_resp['documents'])):
                        if len(memory_resp["documents"][i]) > 1:
                            self.relevant_docs.append(memory_resp["documents"][i][0])

                    if len(self.relevant_docs) == 0:
                        logger.info("No relevant docs found")
                        return False
                else:
                    logger.info("No relevant docs found")
                    return False
            else:
                logger.error(f"No supported memstore found for method 'get_doc'. Please supply a supported memstore")
                raise AttributeError
        except Exception as err:
            logger.error(f"get_doc failed: {err}")
            raise err

        return True
    
    async def query_doc_ai(self) -> str:
        """
        Uses doc found from VecStor and creates a QnA agent
        """
        if isinstance(self.memstore, ChromaMemStore):
            if self.relevant_docs:
                self.prompt = f"""
                You are Susan Anderson, a professional librarian. Your task is to answer questions using text from the pages of REFDOC. Please give a very short answer, less words the better, as you are talking with another bot that has a small token limit. Try removing any uncessary spacing and wording. For lists, give them in one line. 
                If the passage is irrelevant to the answer, you may ignore it.
                """

                self.chat.append({
                    "role": "system",
                    "content": self.prompt
                })

                # add documents to chat
                logger.info(f"Loading {len(self.relevant_docs)} docs into QnA chat")

                doc_page = 1
                for relevant_doc in self.relevant_docs:
                    self.chat.append({
                        "role": "system",
                        "content": f"REFDOC PAGE {doc_page}\n{relevant_doc}"
                    })

                    doc_page += 1

                self.chat.append({
                    "role": "user",
                    "content": f"{self.query}"
                })

                logger.info(f"Sending query to QnA Chat")

                try:
                    chat_completion_parms = {
                        "messages": self.chat,
                        "model": self.model,
                        "temperature": 0.7
                    }

                    response = await chat_completion_request(
                        **chat_completion_parms)
                    
                    resp_content = response["choices"][0]["message"]["content"]
                    resp_content = resp_content.replace("\n", " ")

                    return resp_content
                except Exception as err:
                    logger.error(f"chat completion failed: {err}")
                    return "chat completion failed, document might be too large"
            else:
                logger.error("no relevant_docs found")
                return "no relevant document found"
        elif isinstance(self.memstore, WeaviateMemstore):
            try:
                query_dict = {
                    "file": self.file_name,
                    "website": self.url,
                    "chat": self.chat_role
                }

                path_dict = {
                    "file": "filename",
                    "website": "url",
                    "chat": "role"
                }

                memory_resp = ""
                
                if self.doc_type != "doc_id" or self.doc_type != "all":
                    memory_resp = self.memstore.ask_data_obj(
                        task_id=self.task_id,
                        data_class=self.doc_type,
                        question=self.query,
                        with_path=path_dict[self.doc_type],
                        with_value=query_dict[self.doc_type]
                    )
                elif self.doc_type == "doc_id":
                    # have to search through all classes to find doc id
                    # when found then ask question with that class and break
                    for data_class in self.memstore_classes:
                        find_id = self.memstore.get_obj_by_id(
                            self.task_id,
                            data_class, 
                            self.doc_id
                        )
                        
                        if len(find_id) > 0:
                            memory_resp = self.memstore.ask_data_obj(
                                task_id=self.task_id,
                                data_class=data_class,
                                question=self.query,
                                uid=self.doc_id
                            )
                            break
                elif self.doc_type == "all":
                    for data_class in self.memstore_classes:
                        class_resp = self.memstore.ask_data_obj(
                            task_id=self.task_id,
                            data_class=data_class,
                            question=self.query
                        )

                        memory_resp += f"[{data_class}] {class_resp}\n"
            except Exception as err:
                logger.error(f"QnA failed: {err}")
                
            return memory_resp

                    



