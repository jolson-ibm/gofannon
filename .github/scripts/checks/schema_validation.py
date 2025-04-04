import ast
import json

class SchemaValidationCheck:
    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name

    class ToolDefinitionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.tools = []
            self.current_class = None
            self.definition = None

        def visit_ClassDef(self, node):
            self.current_class = node.name
            self.definition = None
            self.generic_visit(node)
            self.current_class = None

        def visit_FunctionDef(self, node):
            if node.name == 'definition' and node.decorator_list:
                if isinstance(node.decorator_list[0], ast.Name) and node.decorator_list[0].id == 'property':
                    self.definition = node
            self.generic_visit(node)

        def get_definition_return(self):
            for node in ast.walk(self.definition):
                if isinstance(node, ast.Return):
                    return self.node_to_value(node.value)
            return None

        def node_to_value(self, node):
            if isinstance(node, ast.Dict):
                return {
                    self.node_to_value(key): self.node_to_value(value)
                    for key, value in zip(node.keys, node.values)
                }
            elif isinstance(node, ast.Constant): # for Python 3.8+
                return node.value
            elif isinstance(node, ast.Str): # for older Python versions
                return node.s
            elif isinstance(node, ast.Num):
                return node.n
            elif isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
            # Return a string representation for non-literal attribute values (e.g., "self.name")
                return f"{self.node_to_value(node.value)}.{node.attr}"
            elif isinstance(node, ast.List):
                return [self.node_to_value(e) for e in node.elts]
            elif isinstance(node, ast.Tuple):
                return tuple(self.node_to_value(e) for e in node.elts)
            elif isinstance(node, ast.Set):
                return {self.node_to_value(e) for e in node.elts}
            else:
                return None

    def analyze_file(self, content):
        tree = ast.parse(content)
        visitor = self.ToolDefinitionVisitor()
        visitor.visit(tree)

        tools = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                base_names = [base.id for base in node.bases if isinstance(base, ast.Name)]
                if 'BaseTool' in base_names:
                    for decorator in node.decorator_list:
                        if (isinstance(decorator, ast.Call) and getattr(decorator.func, 'attr', None) == 'register') \
                                or (isinstance(decorator, ast.Attribute) and decorator.attr == 'register'):
                            tool_info = {
                                'class_name': node.name,
                                'definition': None
                            }
                            for sub_node in node.body:
                                if isinstance(sub_node, ast.FunctionDef) and sub_node.name == 'definition':
                                    visitor.definition = sub_node
                                    tool_info['definition'] = visitor.get_definition_return()
                            tools.append(tool_info)
                        else:
                            print("the last one failed though")
        return tools

    def validate_definition(self, definition):
        schema_check_prompt = f"""Analyze this OpenAI function definition schema. Return JSON with:    
- "valid": boolean    
- "errors": list of strings describing schema violations    
- "missing_fields": list of required missing fields    
  
Schema to validate: {json.dumps(definition, indent=2)}    
  
Follow these validation rules based on the example structure:  
1. Must have 'type' set to 'function'  
2. Must have 'function' object containing:  
   a. 'name' (string, required)  
   b. 'description' (string, required)  
   c. 'parameters' (object, required) following JSON Schema format  
3. Parameters object must contain:  
   a. 'type' set to 'object'  
   b. 'properties' object containing parameter definitions  
   c. 'required' array listing required parameters  
   d. 'additionalProperties' set to false  
4. Each parameter in 'properties' must define:  
   a. 'type' (string, required)  
   b. 'description' (string, required)  
   c. 'enum' (array, optional) if parameter has specific allowed values  
5. The 'function' object should include 'strict' set to true  
6. No markdown formatting in descriptions  
  
Example of valid structure:  
{{  
    "type": "function",  
    "function": {{  
        "name": "get_weather",  
        "description": "Retrieves current weather for the given location.",  
        "parameters": {{  
            "type": "object",  
            "properties": {{  
                "location": {{  
                    "type": "string",  
                    "description": "City and country e.g. Bogotá, Colombia"  
                }},  
                "units": {{  
                    "type": "string",  
                    "enum": [  
                        "celsius",  
                        "fahrenheit"  
                    ],  
                    "description": "Units the temperature will be returned in."  
                }}  
            }},  
            "required": [  
                "location",  
                "units"  
            ],  
            "additionalProperties": false  
        }},  
        "strict": true  
    }}  
}}"""

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "system",
                "content": "You are a schema validation expert. Analyze OpenAI function definitions."
            }, {
                "role": "user",
                "content": schema_check_prompt
            }],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def process_pr_file(self, file, repo, pr):
        comments = []
        analyzed = False

        if file.filename.endswith('.py'):

            analyzed = True
            content = repo.get_contents(file.filename, ref=pr.head.sha).decoded_content.decode()
            tools = self.analyze_file(content)
            print(f"found {len(tools)} tools in file {file.filename}...")
            if len(tools) == 0:
                comments.append({
                    "path": file.filename,
                    "body": f"No tools found in file '{file.filename}'",
                    "line": 1
                })
            for tool in tools:
                if tool['definition']:
                    validation = self.validate_definition(tool['definition'])
                    if validation.get('valid'):
                        comments.append({
                            "path": file.filename,
                            "body": f"File {tool['class_name']} seems to have a valid schema",
                            "line": 1
                        })
                    else:
                        message = f"⚠️ **Schema Issue in {tool['class_name']}**\n"
                        if validation.get('missing_fields'):
                            message += f"Missing fields: {', '.join(validation['missing_fields'])}\n"
                        if validation.get('errors'):
                            message += "Errors:\n- " + "\n- ".join(validation['errors'])
                        comments.append({
                            "path": file.filename,
                            "body": message,
                            "line": 1
                        })

        return comments, analyzed