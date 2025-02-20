# tests/test_integrations.py
import json

import pytest
from gofannon.base import BaseTool
from gofannon.basic_math.addition import Addition

# Add DummyTool subclass implementing abstract methods  
class DummyTool(BaseTool):
    @property
    def definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "properties": self._parameters,
                    "required": self._required
                }
            }
        }

    def fn(self, *args, **kwargs):
        return "dummy result"

        # LangChain tests
def test_langchain_import_export():
    try:
        from langchain.tools import BaseTool as LangchainBaseTool
        from langchain.tools import WikipediaQueryRun
    except ImportError:
        pytest.skip("langchain-core not installed")

        # Use DummyTool instead of BaseTool
    lc_tool = WikipediaQueryRun()
    base_tool = DummyTool()
    base_tool.import_from_langchain(lc_tool)

    assert base_tool.name == "wikipedia"
    assert "Wikipedia" in base_tool.description

    exported_tool = base_tool.export_to_langchain()
    assert isinstance(exported_tool, LangchainBaseTool)

def test_smolagents_import_export():
    try:
        from smolagents.tools import Tool as SmolTool
    except ImportError:
        pytest.skip("smolagents not installed")

    def test_fn(a: int, b: int) -> int:
        return a + b

    smol_tool = SmolTool()
    smol_tool.name="test_addition",
    smol_tool.description="Adds numbers",
    smol_tool.inputs={
            "a": {"type": "int", "description": "First number"},
            "b": {"type": "int", "description": "Second number"}
        },
    smol_tool.output_type="int",
    smol_tool.forward=test_fn

    base_tool = DummyTool()
    base_tool.import_from_smolagents(smol_tool)

    assert base_tool.name == "test_addition"
    assert "Adds numbers" in base_tool.description

    exported_tool = base_tool.export_to_smolagents()
    assert exported_tool.forward(2, 3) == 5

def test_cross_framework_roundtrip():
    native_tool = Addition()
    lc_tool = native_tool.export_to_langchain()

    # Use DummyTool for import test  
    imported_tool = DummyTool()
    imported_tool.import_from_langchain(lc_tool)

    assert imported_tool.fn(2, 3) == 5
    assert imported_tool.name == "addition"

    exported_smol = native_tool.export_to_smolagents()
    assert exported_smol.forward(4, 5) == 9


# tests/test_integrations.py

def test_bedrock_export(monkeypatch):
    # Mock boto3 client
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")

    # Use a concrete tool instead of BaseTool
    tool = Addition()
    bedrock_config = tool.export_to_bedrock(lambda_arn="arn:aws:lambda:us-east-1:123456789012:function:test")

    assert bedrock_config["toolName"] == "addition"
    assert "num1" in bedrock_config["openAPISchema"]
    assert bedrock_config["lambdaArn"] == "arn:aws:lambda:us-east-1:123456789012:function:test"

def test_bedrock_import():
    sample_tool = {
        "toolName": "bedrock_addition",
        "openAPISchema": json.dumps({
            "openapi": "3.0.0",
            "paths": {
                "/add": {
                    "post": {
                        "description": "Add numbers",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "num1": {
                                                "type": "number",
                                                "description": "First number"
                                            },
                                            "num2": {
                                                "type": "number",
                                                "description": "Second number"
                                            }
                                        },
                                        "required": ["num1", "num2"]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }),
        "lambdaArn": "arn:aws:lambda:us-east-1:123456789012:function:add"
    }

    # Use a concrete tool instead of BaseTool
    tool = Addition()
    tool.import_from_bedrock(sample_tool)

    assert tool.name == "bedrock_addition"
    assert "num1" in tool.definition['function']['parameters']['properties']
    assert "num2" in tool.definition['function']['parameters']['properties']
    assert tool.definition['function']['parameters']['properties']['num1']['description'] == "First number"
    assert tool.definition['function']['parameters']['properties']['num2']['description'] == "Second number"