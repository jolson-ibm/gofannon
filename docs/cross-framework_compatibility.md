## Cross-Framework Compatibility

The toolbox supports integration with popular AI frameworks:

### LangChain Integration

```python  
# Import LangChain tool  
from langchain.tools import WikipediaQueryRun  
  
# Convert to BaseTool  
lc_tool = WikipediaQueryRun()  
base_tool = BaseTool()  
base_tool.import_from_langchain(lc_tool)  
  
# Use in gofannon workflows  
result = base_tool.fn("machine learning")  
  
# Export back to LangChain  
exported_tool = base_tool.export_to_langchain()  
```

### SmolAgents Integration

```python
# Import SmolAgents tool  
from smolagents.tools import Tool  
  
# Convert to BaseTool  
smol_tool = Tool(...)  
base_tool = BaseTool()  
base_tool.import_from_smolagents(smol_tool)  
  
# Export back to SmolAgents  
exported_tool = base_tool.export_to_smolagents()  
```

Key features:

- Bi-directional conversion preserves metadata and functionality
- Automatic schema translation between frameworks
- Runtime checks for required dependencies

This implementation provides:
- Full type checking and schema validation
- Error handling for missing dependencies
- Bidirectional conversion between frameworks
- Automatic parameter translation
- Comprehensive test coverage
- Documentation examples

The tests verify:
1. Successful tool conversions
2. Parameter schema preservation
3. Functionality equivalence
4. Error handling for missing dependencies
5. Round-trip conversions between frameworks

## AWS Bedrock Integration

```python  
from gofannon.base import BaseTool  
import boto3  
  
# Create Bedrock client  
bedrock = boto3.client('bedrock-agent')  
  
# Convert to Bedrock Tool  
base_tool = BaseTool()  
bedrock_tool = base_tool.export_to_bedrock()  
  
# Create Bedrock Agent  
response = bedrock.create_agent(  
    agentName='GofannonAgent',  
    foundationModel='anthropic.claude-v2',  
    instruction='Use Gofannon tools for problem solving',  
    agentResourceRoleArn='arn:aws:iam::123456789012:role/AmazonBedrockExecutionRoleForAgents',  
    agentTools=[bedrock_tool]  
)  
  
# Export entire function registry  
all_tools = FunctionRegistry.get_tools()  
bedrock_tools = [tool().export_to_bedrock() for tool in all_tools] 
```

Key Features:

* Automatic OpenAPI schema generation
* Lambda function packaging for tool execution
* IAM role validation
* Parameter type mapping between Gofannon and Bedrock
