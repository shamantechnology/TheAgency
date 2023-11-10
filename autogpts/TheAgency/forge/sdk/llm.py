import typing
import os

import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential

from .forge_log import ForgeLogger
from litellm import completion, acompletion, AuthenticationError, InvalidRequestError

import tiktoken

LOG = ForgeLogger(__name__)

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def chat_completion_request(
    model, messages, **kwargs
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Generate a response to a list of messages using OpenAI's API"""
    try:
        kwargs["model"] = model
        kwargs["messages"] = messages

        resp = await acompletion(**kwargs)
        return resp
    except AuthenticationError as e:
        LOG.exception("Authentication Error")
    except InvalidRequestError as e:
        LOG.exception("Invalid Request Error")
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def create_embedding_request(
    messages, model="text-embedding-ada-002"
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Generate an embedding for a list of messages using OpenAI's API"""
    try:
        return await openai.Embedding.acreate(
            input=[f"{m['role']}: {m['content']}" for m in messages],
            engine=model,
        )
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def create_doc_embedding_request(
    text, model="text-embedding-ada-002"
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Generate an embedding for a list of messages using OpenAI's API"""
    try:
        return await openai.Embedding.acreate(
            input=[text],
            engine=model,
        )
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise

@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
async def transcribe_audio(
    audio_file: str,
) -> typing.Union[typing.Dict[str, typing.Any], Exception]:
    """Transcribe an audio file using OpenAI's API"""
    try:
        return await openai.Audio.transcribe(model="whisper-1", file=audio_file)
    except Exception as e:
        LOG.error("Unable to generate ChatCompletion response")
        LOG.error(f"Exception: {e}")
        raise

###
# token counting for chat
# from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
###
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
        
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. 
            See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    
    try:
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens
    except Exception as err:
        LOG.error(f"token calc for\n{messages}\nfailed: {err}")
        raise err


#####################
# OpenAI assistant #
####################

def get_assistant(assistant: str) -> any:
    """
    Returns the assistant specified
    """
    try:
        client = openai.OpenAI()
        assistant_obj = client.beta.assistants.retrieve(assistant)
    except Exception as err:
        LOG.error(f"Getting assistant {assistant} failed: {err}")
        raise

    return assistant_obj


def create_thread() -> str:
    """
    Creates an empty thread for messages with assistant
    Returns thread ID
    """
    try:
        client = openai.OpenAI()
        empty_thread = client.beta.threads.create()
    except Exception as err:
        LOG.error(f"Creating new thread failed: {err}")
        raise

    return empty_thread["id"]

