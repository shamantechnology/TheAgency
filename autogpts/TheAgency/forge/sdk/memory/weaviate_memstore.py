"""
Weaviate memstore
"""
import os
from typing import Any

import weaviate
from weaviate.embedded import EmbeddedOptions

from ..forge_log import ForgeLogger

logger = ForgeLogger(__name__)

class WeaviateMemstore():
    def __init__(
            self,
            data_classes: list = None,
            use_embedded: bool = False):
        
        # create client with openai key
        if use_embedded:
            logger.info("Connecting to weaviate embedded client")
            self.client = weaviate.Client(
                embedded_options=EmbeddedOptions(),
                additional_headers = {
                    "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                }
            )
        else:
            # this is for cloud hosted 
            auth_config = weaviate.AuthApiKey(api_key=os.getenv("WEAVIATE_API_KEY"))

            logger.info(f"Connecting to weaviate instance at {os.getenv('WEAVIATE_URL')}")
            try:
                self.client = weaviate.Client(
                    url=os.getenv("WEAVIATE_URL"),
                    auth_client_secret=auth_config,
                    additional_headers = {
                        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
                    }
                )
            except Exception as err:
                logger.info(f"Error starting client: {err}")
                raise err

        # create schema classes with openai vectorizer if not live
        try:
            if data_classes:
                self.data_class_names = []
                self.data_class_schemas = []
                for data_class in data_classes:
                    if not self.client.schema.exists(data_class):
                        self.data_class_names.append(data_class)

                        data_class_obj = {
                            "class": data_class,
                            "vectorizer": "text2vec-openai"
                        }

                        self.data_class_schemas.append(data_class_obj)
                        self.client.schema.create_class(data_class_obj)
        
            # create basic classes
            self.data_class_names = [
                "ability",
                "chat",
                "file",
                "website"
            ]

            self.data_class_schemas = [
                {
                    "class": "ability",
                    "vectorizer": "text2vec-openai",
                    "properties": [
                        {
                            "name": "function",
                            "dataType": ["text"],
                        },
                        {
                            "name": "content",
                            "dataType": ["text"],
                        },
                        {
                            "name": "task_id",
                            "dataType": ["text"],
                        }
                    ]
                },
                {

                    "class": "chat",
                    "vectorizer": "text2vec-openai",
                    "properties": [
                        {
                            "name": "role",
                            "dataType": ["text"],
                        },
                        {
                            "name": "content",
                            "dataType": ["text"],
                        },
                        {
                            "name": "task_id",
                            "dataType": ["text"],
                        }
                    ]
                },
                {

                    "class": "website",
                    "vectorizer": "text2vec-openai",
                    "properties": [
                        {
                            "name": "url",
                            "dataType": ["text"],
                        },
                        {
                            "name": "content",
                            "dataType": ["text"],
                        },
                        {
                            "name": "task_id",
                            "dataType": ["text"],
                        }
                    ]
                },
                {

                    "class": "file",
                    "vectorizer": "text2vec-openai",
                    "properties": [
                        {
                            "name": "filename",
                            "dataType": ["text"],
                        },
                        {
                            "name": "content",
                            "dataType": ["text"],
                        },
                        {
                            "name": "task_id",
                            "dataType": ["text"],
                        }
                    ]
                }
            
            ]

            for data_class_schema in self.data_class_schemas:
                print(f"checking class {data_class_schema['class']}")
                if not self.client.schema.exists(data_class_schema["class"]):
                    print(f"adding class {data_class_schema['class']}")
                    self.client.schema.create_class(data_class_schema)
        except Exception as err:
            logger.error(f"Schema creation failed: {err}")
            raise err

        
    def add_data_obj(self, data_class: str, data_obj: dict) -> str:
        """
        Add data obj to weaviate collection
        """
        if data_class in self.data_class_names:
            uuid = self.client.data_object.create(data_obj, data_class)
        else:
            logger.error(f"Data class {data_class} not found")
            raise AttributeError

        return uuid
    
    def get_data_obj(self, task_id: str, data_class: str, query: str) -> list:
        try:
            class_schema = self.client.schema.get(data_class)
        except Exception as err:
            logger.error(f"{data_class} schema was not found {err}")
            raise

        class_props = []
        for props in class_schema["properties"]:
            class_props.append(props["name"])

        try:
            resp = (self.client.query
                .get(data_class, class_props)
                .with_near_text({
                    "concepts": [query]
                })
                .with_where({
                    "path": ["task_id"],
                    "operator": "Equal",
                    "valueText": task_id
                })
                .do())
        except Exception as err:
            logger.error(f"query failed: {err}")
            raise

        # have to capitalize class name if not already
        # weaviate stores classes names with a capital
        return resp['data']['Get'][data_class.capitalize()]

    def get_obj_by_id(self, task_id: str, data_class: str, uid: str) -> list:
        class_schema = self.client.schema.get(data_class)

        class_props = []
        for props in class_schema["properties"]:
            class_props.append(props["name"])

        resp = (self.client.query
            .get(data_class, class_props)
            .with_near_object({
                "id": uid
            })
            .with_where({
                "path": ["task_id"],
                "operator": "Equal",
                "valueText": task_id
            })
            .do())

        return resp['data']['Get'][data_class.capitalize()]
    
    def ask_data_obj(
        self,
        task_id: str,
        data_class: str,
        question: str,
        with_path: str = None,
        with_value: Any = None,
        uid: str = None
    ) -> str:
        class_schema = self.client.schema.get(data_class)

        class_props = []
        for props in class_schema["properties"]:
            class_props.append(props["name"])

        # add answer format for graphql
        class_props.append(
            "_additional {answer {hasAnswer property result startPosition endPosition} }"
        )

        if uid:
            resp = (self.client.query
                .get(data_class, class_props)
                .with_near_object({
                    "id": uid
                })
                .with_where({
                    "path": ["task_id"],
                    "operator": "Equal",
                    "valueText": task_id
                })
                .with_ask({
                    "question": question
                })
                .do())
        elif with_path and with_value:
            resp = (self.client.query
                .get(data_class, class_props)
                .with_where({
                    "operator": "And",
                    "operands": [
                    {
                        "path": ["task_id"],
                        "operator": "Equal",
                        "valueText": task_id
                    },{
                        "path": [with_path],
                        "operator": "Equal",
                        "valueText": with_value
                    }]
                })
                .with_ask({
                    "question": question
                })
                .do())
        else:
            resp = (self.client.query
                .get(data_class, class_props)
                .with_where({
                    "path": ["task_id"],
                    "operator": "Equal",
                    "valueText": task_id
                })
                .with_ask({
                    "question": question
                })
                .do())
            
        print(resp['data']['Get'][data_class.capitalize()])
        answer = resp['data']['Get'][data_class.capitalize()][0]["_additional"]["answer"]

        if answer["hasAnswer"]:
            return answer["result"]
        else:
            return "No answer found"

