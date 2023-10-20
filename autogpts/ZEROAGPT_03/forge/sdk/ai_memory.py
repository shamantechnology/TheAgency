"""
Load documents in AI and for QnA
"""
import os

from .forge_log import ForgeLogger
from .memory.memstore import ChromaMemStore
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
        file_name: str = None,
        chat_role: str = None,
        url: str = None,
        model: str = os.getenv("OPENAI_MODEL")):

        self.workspace = workspace
        self.task_id = task_id
        self.query = query
        self.model = model
        self.file_name = file_name
        self.chat_role = chat_role
        self.url = url

        if doc_type not in ["file", "chat", "website", "all"]:
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

            # find doc in chromadb
            cwd = self.workspace.get_cwd_path(self.task_id)
            chroma_dir = f"{cwd}/chromadb/"

            memory = ChromaMemStore(chroma_dir)

            if self.doc_type == "file":
                memory_resp = memory.query(
                    task_id=self.task_id,
                    query=self.query,
                    filters={
                        "filename": self.file_name
                    }
                )
            elif self.doc_type == "chat":
                 memory_resp = memory.query(
                    task_id=self.task_id,
                    query=self.query,
                    filters={
                        "role": self.chat_role
                    }
                )
            elif self.doc_type == "website":
                 memory_resp = memory.query(
                    task_id=self.task_id,
                    query=self.query,
                    filters={
                        "url": self.url
                    }
                )
            elif self.doc_type == "all":
                memory_resp = memory.query(
                    task_id=self.task_id,
                    query=self.query
                )

            if len(memory_resp["documents"][0]) > 0:
                logger.info(
                    f"Relevant docs found! Doc count: {len(memory_resp['documents'][0])}")
                
                # need to add in chucking up of large docs
                for i in range(len(memory_resp['documents'][0])):
                    self.relevant_docs.append(memory_resp["documents"][0][i])
            else:
                logger.info("No relevant docs found")
                return False
        except Exception as err:
            logger.error(f"get_doc failed: {err}")
            raise err

        return True
    
    async def query_doc_ai(self) -> str:
        """
        Uses doc found from VecStor and creates a QnA agent
        """

        if self.relevant_docs:
            self.prompt = f"""
            You are 'The Librarian' a bot that answers questions using text from the reference document included below. Please give short and concise answers as you are talking with another bot that is limited in space. Try removing any uncessary spacing and wording. For lists, give them in one line. 
            If the passage is irrelevant to the answer, you may ignore it.
            """

            self.chat.append({
                "role": "system",
                "content": self.prompt
            })

            # add documents to chat
            for relevant_doc in self.relevant_docs:
                self.chat.append({
                    "role": "system",
                    "content": f"{relevant_doc}"
                })

            self.chat.append({
                "role": "user",
                "content": f"{self.question}"
            })

            logger.info(f"Sending question to QnA Chat")

            try:
                chat_completion_parms = {
                    "messages": self.chat,
                    "model": self.model,
                    "temperature": 0.7
                }

                response = await chat_completion_request(
                    **chat_completion_parms)
                
                resp_content = response["choices"][0]["message"]["content"]

                logger.info(f"reponse: {resp_content}")

                return resp_content
            except Exception as err:
                logger.error(f"chat completion failed: {err}")
                return "chat completion failed, document might be too large"
        else:
            logger.error("no relevant_docs found")
            return "no relevant document found"


