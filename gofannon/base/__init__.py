import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Callable
import json
import logging
from pathlib import Path
from ..config import ToolConfig

from typing import Any, Dict


try:
    from smolagents.tools import Tool as SmolTool
    from smolagents.tools import tool as smol_tool_decorator
    _HAS_SMOLAGENTS = True
except ImportError:
    _HAS_SMOLAGENTS = False

try:
    import boto3
    from botocore.exceptions import ClientError
    _HAS_BEDROCK = True
except ImportError:
    _HAS_BEDROCK = False

try:
    from langchain.tools import BaseTool as LangchainBaseTool
    from langchain.pydantic_v1 import BaseModel, Field
    from typing import Type, Optional
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

@dataclass
class ToolResult:
    success: bool
    output: Any
    error: str = None
    retryable: bool = False

class WorkflowContext:
    def __init__(self, firebase_config=None):
        self.data = {}
        self.execution_log = []
        self.firebase_config = firebase_config
        self.local_storage = Path.home() / ".llama" / "checkpoints"
        self.local_storage.mkdir(parents=True, exist_ok=True)


    def save_checkpoint(self, name="checkpoint"):
        if self.firebase_config:
            self._save_to_firebase(name)
        else:
            self._save_local(name)

    def _save_local(self, name):
        path = self.local_storage / f"{name}.json"
        with open(path, 'w') as f:
            json.dump({
                'data': self.data,
                'execution_log': self.execution_log
            }, f)

    def _save_to_firebase(self, name):
        from firebase_admin import firestore
        db = firestore.client()
        doc_ref = db.collection('checkpoints').document(name)
        doc_ref.set({
            'data': self.data,
            'execution_log': self.execution_log,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

    def log_execution(self, tool_name, duration, input_data, output_data):
        entry = {
            'tool': tool_name,
            'duration': duration,
            'input': input_data,
            'output': output_data
        }
        self.execution_log.append(entry)

class BaseTool(ABC):
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        self._load_config()
        self._configure(**kwargs)
        self.logger.debug("Initialized %s tool", self.__class__.__name__)
        self.name = kwargs.get('name', self.__class__.__name__.lower())
        self.description = kwargs.get('description', "No description provided")
        self._definition = None

    def _configure(self, **kwargs):
        """Set instance-specific configurations"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _load_config(self):
        """Auto-load config based on tool type"""
        if hasattr(self, 'API_SERVICE'):
            self.api_key = ToolConfig.get(f"{self.API_SERVICE}_api_key")

    @property
    def definition(self):
        """Return the tool's definition"""
        if self._definition is None:
            # Provide a default definition if not set
            self._definition = {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        return self._definition

    @definition.setter
    def definition(self, value):
        """Set the tool's definition"""
        self._definition = value

    @property
    def output_schema(self):
        return self.definition.get('function', {}).get('parameters', {})

    @abstractmethod
    def fn(self, *args, **kwargs):
        pass

    def execute(self, context: WorkflowContext, **kwargs) -> ToolResult:
        try:
            start_time = time.time()
            result = self.fn(**kwargs)
            duration = time.time() - start_time

            context.log_execution(
                tool_name=self.__class__.__name__,
                duration=duration,
                input_data=kwargs,
                output_data=result
            )

            return ToolResult(success=True, output=result)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                retryable=True
            )

    def import_from_smolagents(self, smol_tool: "SmolTool"):
        """
        Takes a smolagents Tool instance and adapts it into this Tool.
        """
        if not _HAS_SMOLAGENTS:
            raise RuntimeError(
                "smolagents is not installed or could not be imported. "
                "Install it or check your environment."
            )
        self.name = smol_tool.name[0]
        self.description = smol_tool.description #getattr(smol_tool, "description", "No description provided.")


        def adapted_fn(*args, **kwargs):
            return smol_tool.forward(*args, **kwargs)


        self.fn = adapted_fn

    def export_to_smolagents(self) -> "SmolTool":
        """
        Export this Tool as a smolagents Tool instance.
        This sets up a smolagents-style forward method that calls self.fn.
        """
        if not _HAS_SMOLAGENTS:
            raise RuntimeError(
                "smolagents is not installed or could not be imported. "
                "Install it or check your environment."
            )

            # Provide a standard forward function that calls self.fn
        def smol_forward(*args, **kwargs):
            return self.fn(*args, **kwargs)


        inputs_definition = {
            "example_arg": {
                "type": "string",
                "description": "Example argument recognized by this tool"
            }
        }
        output_type = "string"

        # Construct a new smolagents Tool with the minimal fields
        exported_tool = SmolTool()
        exported_tool.name = getattr(self, "name", "exported_base_tool")
        exported_tool.description = getattr(self, "description", "Exported from Tool")
        exported_tool.inputs = inputs_definition
        exported_tool.output_type = output_type
        exported_tool.forward = smol_forward
        exported_tool.is_initialized = True

        return exported_tool

    def import_from_langchain(self, langchain_tool: "LangchainBaseTool"):
        if not _HAS_LANGCHAIN:
            raise RuntimeError("langchain is not installed. Install with `pip install langchain-core`")

        self.name = getattr(langchain_tool, "name", "exported_langchain_tool")
        self.description = getattr(langchain_tool, "description", "No description provided.")

        maybe_args_schema = getattr(langchain_tool, "args_schema", None)
        if maybe_args_schema and hasattr(maybe_args_schema, "schema") and callable(maybe_args_schema.schema):
            args_schema = maybe_args_schema.schema()
        else:
            args_schema = {}

            # Store parameters to avoid modifying the definition property directly
        self._parameters = args_schema.get("properties", {})
        self._required = args_schema.get("required", [])

        # Adapt the LangChain tool's execution method
        def adapted_fn(*args, **kwargs):
            return langchain_tool._run(*args, **kwargs)

        self.fn = adapted_fn

    def export_to_langchain(self) -> "LangchainBaseTool":
        if not _HAS_LANGCHAIN:
            raise RuntimeError(
                "langchain is not installed. Install with `pip install langchain-core`"
            )

        from pydantic import create_model

        # Create type mapping from JSON schema types to Python types
        type_map = {
            "number": float,
            "string": str,
            "integer": int,
            "boolean": bool,
            "object": dict,
            "array": list
        }

        parameters = self.definition.get("function", {}).get("parameters", {})
        param_properties = parameters.get("properties", {})

        # Dynamically create ArgsSchema using pydantic.create_model
        fields = {}
        for param_name, param_def in param_properties.items():
            param_type = param_def.get("type", "string")
            description = param_def.get("description", "")
            fields[param_name] = (
                type_map.get(param_type, str),
                Field(..., description=description)
            )

        ArgsSchema = create_model('ArgsSchema', **fields)

        # Create tool subclass with our functionality
        class ExportedTool(LangchainBaseTool):
            name: str = self.definition.get("function", {}).get("name", "")
            description: str = self.definition.get("function", {}).get("description", "")
            args_schema: Type[BaseModel] = ArgsSchema
            fn: Callable = self.fn

            def _run(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

        # Instantiate and return the tool
        tool = ExportedTool()
        return tool

    def export_to_bedrock(self, lambda_arn: str = None) -> dict:
        """
        Export tool as Bedrock Agent tool configuration
        """
        if not _HAS_BEDROCK:
            raise RuntimeError("boto3 not installed. Install with `pip install boto3`")

        openapi_schema = self._generate_openapi_schema()

        # Create tool configuration
        tool_config = {
            "toolName": self.name,
            "description": self.definition['function']['description'],
            "openAPISchema": json.dumps(openapi_schema),
            "lambdaArn": lambda_arn or self._create_bedrock_lambda()
        }

        return tool_config

    def _generate_openapi_schema(self) -> dict:
        """Convert Gofannon definition to OpenAPI schema"""
        params = self.definition['function']['parameters']

        openapi_schema = {
            "openapi": "3.0.0",
            "info": {
                "title": self.name,
                "version": "1.0.0"
            },
            "paths": {
                f"/{self.name}": {
                    "post": {
                        "description": self.definition['function']['description'],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            param: {
                                                "type": props['type'],
                                                "description": props['description']
                                            }
                                            for param, props in params['properties'].items()
                                        },
                                        "required": params.get('required', [])
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": {
                                "description": "Successful operation",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "result": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        return openapi_schema

    def _create_bedrock_lambda(self) -> str:
        """Create Lambda function for Bedrock integration"""
        if not _HAS_BEDROCK:
            raise RuntimeError("boto3 required for Lambda creation")

        lambda_client = boto3.client('lambda')
        role_arn = self._get_or_create_bedrock_role()

        # Generate Lambda code
        lambda_code = f'''  
import json  
from {self.__class__.__module__} import {self.__class__.__name__}  
  
def lambda_handler(event, context):  
    tool = {self.__class__.__name__}()  
    result = tool.fn(**event)  
    return {{  
        'statusCode': 200,  
        'body': json.dumps({{'result': result}})  
    }}  
    '''

        try:
            response = lambda_client.create_function(
                FunctionName=f"gofannon-{self.name}",
                Runtime='python3.10',
                Role=role_arn,
                Handler='lambda_function.lambda_handler',
                Code={
                    'ZipFile': create_zip_package(lambda_code)
                },
                Description=f"Gofannon tool: {self.name}",
                Timeout=30,
                MemorySize=256
            )
            return response['FunctionArn']
        except ClientError as e:
            raise RuntimeError(f"Failed to create Lambda: {e}")

    def _get_or_create_bedrock_role(self) -> str:
        """Get or create IAM role for Bedrock integration"""
        iam = boto3.client('iam')
        role_name = 'gofannon-bedrock-execution-role'

        try:
            return iam.get_role(RoleName=role_name)['Role']['Arn']
        except iam.exceptions.NoSuchEntityException:
            assume_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "bedrock.amazonaws.com"},
                        "Action": "sts:AssumeRole"
                    }
                ]
            }

        iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_policy))

        iam.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/AWSLambda_FullAccess')

        return iam.get_role(RoleName=role_name)['Role']['Arn']

# Helper function for AWS
def create_zip_package(code: str) -> bytes:
    """Create in-memory ZIP package for Lambda deployment"""
    from io import BytesIO
    import zipfile

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, 'w') as z:
        z.writestr('lambda_function.py', code)
    buffer.seek(0)
    return buffer.read()

