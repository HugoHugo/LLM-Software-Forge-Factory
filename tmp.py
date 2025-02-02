from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from .db_operations import DBOperations, Task, Component, ComponentStatus
import logging
import black
import ast
import re
from time import sleep

class AgentState(TypedDict):
    task_id: int
    component_id: int
    implementation_id: Optional[int]
    api_code: Optional[str]
    test_code: Optional[str]
    error: Optional[str]

class LangGraphAgent:
    def __init__(self, db: DBOperations, llm):
        self.db = db
        self.llm = llm

    def handle_error(state: AgentState) -> AgentState:
        """Handle any errors that occurred during the process."""
        print(f"Error occurred: {state['error']}")
        if state.get('debug_info'):
            print(f"\nDebug information:\n{state['debug_info']}")
        
        return start_agent_graph()

    def extract_code_sections(self, response: str) -> tuple[Optional[str], Optional[str]]:
        """Extract Python code from markdown code blocks and separate imports."""
        try:
            # Find content between any variation of ```python and ``` markers
            patterns = [
                r'```python\s*\n(.*?)\s*```',  # Standard format
                r'```python(.*?)```',           # No newline after python
                r'```\s*python\s*\n(.*?)\s*```' # Space between ``` and python
            ]
            
            code_block = None
            for pattern in patterns:
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    code_block = match.group(1).strip()
                    break
            
            if not code_block:
                print(f"Failed to extract code block. Response:\n{response}")
                return None, None
                
            # Split imports and main code
            lines = code_block.split('\n')
            import_lines = []
            code_lines = []
            in_import_section = True  # Assume imports come first
            
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines but maintain import section tracking
                    continue
                
                if line.startswith('from ') or line.startswith('import '):
                    import_lines.append(line)
                else:
                    in_import_section = False
                    code_lines.append(line)
            
            imports = '\n'.join(import_lines)
            code = '\n'.join(code_lines)
            
            if not imports and not code:
                print(f"Extracted empty code sections. Response:\n{response}")
                return None, None
                
            return imports, code
            
        except Exception as e:
            print(f"Error extracting code sections: {str(e)}")
            print(f"Response:\n{response}")
            return None, None


    def validate_python_code(self, code: str) -> tuple[bool, Optional[str]]:
        """Validate if the generated code is syntactically correct Python."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def format_python_code(self, code: str) -> str:
        """Format Python code using black and fix indentation."""
        try:
            # First attempt: basic indent fixing for common patterns
            lines = code.split('\n')
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                if not stripped:  # Keep empty lines
                    formatted_lines.append('')
                    continue
                    
                # Decrease indent for lines starting with specific keywords
                if stripped.startswith(('except', 'else:', 'elif', 'finally:')):
                    indent_level = max(0, indent_level - 1)
                    
                # Special handling for function/class definitions and their decorators
                if stripped.startswith('@'):
                    formatted_lines.append('    ' * indent_level + stripped)
                    continue
                    
                # Add the line with current indent level
                formatted_lines.append('    ' * indent_level + stripped)
                
                # Increase indent after lines ending with ':'
                if stripped.endswith(':'):
                    indent_level += 1
                # Decrease indent after 'return' or 'break' statements
                elif stripped.startswith(('return', 'break', 'continue')):
                    indent_level = max(0, indent_level - 1)
            
            # Join lines back together
            pre_formatted = '\n'.join(formatted_lines)
            
            # Then use black for final formatting
            final_formatted = black.format_str(pre_formatted, mode=black.FileMode())
            return final_formatted
            
        except Exception as e:
            print(f"Warning: Code formatting failed: {str(e)}")
            print("Falling back to original code:")
            print(code)
            return code

    def generate_api_code(self, state: AgentState) -> Command[Literal["generate_tests", "handle_error"]]:
        task = self.db.session.query(Task).get(state["task_id"])
        
        """Generate API implementation code based on feature description."""
        system_prompt = """You are an expert Python developer. Generate FastAPI implementation code for the given feature description.
    Provide your response as a Python code block starting with ```python and ending with ```.
    Include all necessary imports at the top of the code.
    Don't include calls to start the server, instead focus on the feature implementation.

    Example format:
    ```python
    from fastapi import FastAPI
    from datetime import datetime

    app = FastAPI()

    @app.get("/endpoint")
    def endpoint():
        return {"result": "value"}
    ```"""

        user_prompt = f"Generate FastAPI implementation for this feature: {task}"
        
        complete_prompt = f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        try:
            response = self.llm.invoke(complete_prompt).content
            print("\nAPI Generation Response:", response)  # Debug print
            
            imports, code = self.extract_code_sections(response)
            
            if not imports or not code:
                return Command(
                    update={
                        "error": "Failed to extract code sections from response",
                        "debug_info": response
                    },
                    goto="handle_error"
                )
                
            complete_code = f"{imports}\n\n{code}"
            complete_code = self.format_python_code(complete_code)
            is_valid, error = self.validate_python_code(complete_code)
            
            if not is_valid:
                return Command(
                    update={
                        "error": f"Invalid Python code generated: {error}",
                        "debug_info": complete_code
                    },
                    goto="handle_error"
                )

            formatted_code = self.format_python_code(complete_code)
            
            return Command(
                update={"api_code": formatted_code},
                goto="generate_tests"
            )
        except Exception as e:
                return Command(
                    update={"error": str(e)},
                    goto="handle_error"
                )

    def generate_tests(self, state: AgentState) -> Command[Literal["save_implementation", "handle_error"]]:
        """Generate test code for the API implementation."""
        system_prompt = """You are an expert Python testing developer. Generate pytest tests for the given API implementation.
    Provide your response as a Python code block starting with ```python and ending with ```.
    Include all necessary imports at the top of the code.

    Example format:
    ```python
    import pytest
    from fastapi.testclient import TestClient
    from datetime import datetime
    from main import app

    def test_endpoint():
        client = TestClient(app)
        response = client.get("/endpoint")
        assert response.status_code == 200
    ```"""

        user_prompt = f"""Generate pytest tests for this API implementation:

    API Code:
    {state['api_code']}

    Feature Description:
    {state['feature_description']}"""

        complete_prompt = f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        
        try:
            response = self.llm.invoke(complete_prompt).content
            print("\nTest Generation Response:", response)  # Debug print
            
            imports, code = self.extract_code_sections(response)
            
            if not imports or not code:
                return Command(
                    update={
                        "error": "Failed to extract code sections from response",
                        "debug_info": response
                    },
                    goto="handle_error"
                )
                
            complete_code = f"{imports}\n\n{code}"
            complete_code = self.format_python_code(complete_code)
            is_valid, error = self.validate_python_code(complete_code)
            
            if not is_valid:
                return Command(
                    update={
                        "error": f"Invalid Python test code generated: {error}",
                        "debug_info": complete_code
                    },
                    goto="handle_error"
                )
            formatted_test_code = self.format_python_code(complete_code)
            
            return Command(
                update={"test_code": formatted_test_code},
                goto="save_implementation"
            )
        except Exception as e:
            return Command(
                update={"error": str(e)},
                goto="handle_error"
            )

    def save_implementation(self, state: AgentState) -> Command[Literal["git_operations", "handle_error"]]:
        try:
            implementation_id = self.db.save_implementation(
                state["component_id"],
                state["api_code"],
                state["test_code"],
                {"task_id": state["task_id"]}
            )
            
            return Command(
                update={"implementation_id": implementation_id},
                goto="git_operations"
            )
        except Exception as e:
            return Command(
                update={"error": str(e)},
                goto="handle_error"
            )

    def git_operations(self, state: AgentState):
        try:
            # Your existing git operations code
            self.db.update_component_status(state["component_id"], ComponentStatus.IMPLEMENTED)
            return Command(goto=END)
        except Exception as e:
            return Command(
                update={"error": str(e)},
                goto="handle_error"
            )

    def create_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        workflow.add_node("generate_api_code", self.generate_api_code)
        workflow.add_node("generate_tests", self.generate_tests)
        workflow.add_node("save_implementation", self.save_implementation)
        workflow.add_node("git_operations", self.git_operations)
        workflow.add_node("handle_error", self.handle_error)
        
        workflow.add_edge(START, "generate_api_code")
        workflow.add_edge("generate_tests", "save_implementation")
        workflow.add_edge("save_implementation", "git_operations")
        workflow.add_edge("git_operations", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()

def start_agent_graph(db_url: str, llm):
    db = DBOperations(db_url)
    agent = LangGraphAgent(db, llm)
    graph = agent.create_graph()
    
    try:
        while True:
            task = db.get_next_task()
            if not task:
                logging.info("No pending tasks")
                break
                
            for event in graph.stream({
                "task_id": task.id,
                "component_id": task.component_id,
            }):
                logging.debug(f"Agent event: {event}")
                
            sleep(3600)
    finally:
        db.close()