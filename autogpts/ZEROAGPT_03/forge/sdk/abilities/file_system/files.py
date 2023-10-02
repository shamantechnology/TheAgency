from typing import List, Match
import re
import os

from forge.sdk.memory.memstore import ChromaMemStore

from ...forge_log import ForgeLogger
from ..registry import ability

logger = ForgeLogger(__name__)

@ability(
    name="list_files",
    description="List files in a directory",
    parameters=[
        {
            "name": "path",
            "description": "Path to the directory",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list[str]",
)
async def list_files(agent, task_id: str, path: str) -> List[str]:
    """
    List files in a workspace directory
    """
    only_files = []
    try:
        file_list = agent.workspace.list(task_id=task_id, path=path)
        for ffile in file_list:
            if ffile["filetype"] == "file":
                only_files.append(ffile["filename"])
    except Exception as err:
        logger.error(f"list_files failed: {err}")
        raise err
    
    if len(only_files) == 0:
        only_files.append('no files found')
        
    return only_files

# @ability(
#     name="write_source_code",
#     description="Write programming language source code to a file",
#     parameters=[
#         {
#             "name": "file_name",
#             "description": "Name of the file",
#             "type": "string",
#             "required": True,
#         },
#         {
#             "name": "code",
#             "description": "Code to write to the file",
#             "type": "string",
#             "required": True,
#         },
#     ],
#     output_type="None",
# )
# async def write_source_code(agent, task_id: str, file_name: str, code: str) -> None:
#     """
#     Write source code as string/text to a file
#     """

#     # clean extra escape slashes
#     code = code.replace('\\\\', '\\')
    
#     # clean \n being written as text and not a new line
#     code = code.replace('\\n', '\n')

#     agent.workspace.write_str(task_id=task_id, path=file_name, data=code)
    
#     await agent.db.create_artifact(
#         task_id=task_id,
#         file_name=file_name.split("/")[-1],
#         relative_path=file_name,
#         agent_created=True,
#     )

#     add_ability_memory(task_id, code, "write_source_code")


@ability(
    name="write_file",
    description="Write data to a file",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of the file",
            "type": "string",
            "required": True,
        },
        {
            "name": "data",
            "description": "Data to write to the file",
            "type": "bytes",
            "required": True,
        },
    ],
    output_type="None",
)
async def write_file(agent, task_id: str, file_path: str, data: bytes) -> None:
    """
    Write data to a file
    """
    if isinstance(data, str):
        data = data.encode()

    agent.workspace.write(task_id=task_id, path=file_path, data=data)
    
    await agent.db.create_artifact(
        task_id=task_id,
        file_name=file_path.split("/")[-1],
        relative_path=file_path,
        agent_created=True,
    )

    add_memory(task_id, str(data), "write_file")

@ability(
    name="read_file",
    description="Read data from a file",
    parameters=[
        {
            "name": "file_path",
            "description": "Path to the file including file name",
            "type": "string",
            "required": True,
        },
    ],
    output_type="bytes",
)
async def read_file(agent, task_id: str, file_path: str) -> bytes:
    """
    Read data from a file
    """
    return agent.workspace.read(task_id=task_id, path=file_path)

@ability(
    name="search_in_file",
    description="Search the contents of a file using regex",
    parameters=[
        {
            "name": "regex",
            "description": "Regular expression",
            "type": "string",
            "required": True
        },
        {
            "name": "file_name",
            "description": "Name of file",
            "type": "string",
            "required": True,
        }
    ],
    output_type="list"
)
async def search_file(agent, task_id: str, file_path: str, regex: str) -> List[Match]:
    """
    Search file using regex
    """
    open_file = agent.workspace.read(task_id=task_id, path=file_path)

    try:
        open_file = agent.workspace.read(task_id=task_id, path=file_name)
        search_rgx = re.findall(rf"{regex}", open_file.decode())
    except Exception as err:
        logger.error(f"search_file failed: {err}")
        raise err

    return search_rgx

# @ability(
#     name="get_cwd",
#     description="Get the current working directory",
#     parameters=[],
#     output_type="str"
# )
# async def get_cwd(agent, task_id) -> str:
#     return agent.workspace.get_cwd_path(task_id)

@ability(
    name="file_line_count",
    description="Returns the line count of a file. Useful to find the size of a file.",
    parameters=[
        {
            "name": "file_name",
            "description": "Name of file",
            "type": "string",
            "required": True,
        }
    ],
    output_type="int"
)
async def file_line_count(agent, task_id: str, file_name: str) -> int:
    line_count = 0
    try:
        open_file = agent.workspace.readlines(task_id=task_id, path=file_name)
        line_count = len(open_file)
    except Exception as err:
        logger.error(f"file_line_count failed: {err}")
        raise err
    
    return line_count