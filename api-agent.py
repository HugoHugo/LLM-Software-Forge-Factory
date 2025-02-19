import ast
import re
import sqlite3
import os
import sys
from pathlib import Path
from datetime import datetime
import logging
from time import sleep
from pprint import pprint
from typing import TypedDict, Literal, Optional

from custom_logging import setup_logging

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langchain_ollama import ChatOllama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
import black

DB_CONNECTION_PATH = os.path.expanduser("~/LLM-Software-Forge-Factory/sqlite_features_db.sqlite")

llm = ChatOllama(
    model="deepseek-r1:7b",
    temperature=0.7,
    callbacks=CallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=False,
    cache=False
)


# Define the state that will be passed between nodes
class AgentState(TypedDict):
    feature_description: str
    api_file_path: str
    test_file_path: str
    feature_id: int
    api_code: Optional[str]
    test_code: Optional[str]
    error: Optional[str]
    debug_info: Optional[str]

def append_to_file_using_path(path: Path, content: str):
    with path.open("a") as f:
        f.write(content)

def validate_python_code(code: str) -> tuple[bool, Optional[str]]:
    """Validate if the generated code is syntactically correct Python."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def format_python_code(code: str) -> str:
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

def extract_code_sections(response: str) -> tuple[Optional[str], Optional[str]]:
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

def generate_api_code(state: AgentState) -> Command[Literal["generate_tests", "handle_error"]]:
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

    user_prompt = f"Generate FastAPI implementation for this feature: {state['feature_description']}"
    
    complete_prompt = f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    
    try:
        response = llm.invoke(complete_prompt).content
        print("\nAPI Generation Response:", response)  # Debug print
        
        imports, code = extract_code_sections(response)
        
        if not imports or not code:
            return Command(
                update={
                    "error": "Failed to extract code sections from response",
                    "debug_info": response
                },
                goto="handle_error"
            )
            
        complete_code = f"{imports}\n\n{code}"
        complete_code = format_python_code(complete_code)
        is_valid, error = validate_python_code(complete_code)
        
        if not is_valid:
            return Command(
                update={
                    "error": f"Invalid Python code generated: {error}",
                    "debug_info": complete_code
                },
                goto="handle_error"
            )
        
        formatted_code = format_python_code(complete_code)
        
        return Command(
            update={"api_code": formatted_code},
            goto="generate_tests"
        )
            
    except Exception as e:
        return Command(
            update={
                "error": f"Error generating API code: {str(e)}",
                "debug_info": response if 'response' in locals() else None
            },
            goto="handle_error"
        )

def generate_tests(state: AgentState) -> Command[Literal["write_files", "handle_error"]]:
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
        response = llm.invoke(complete_prompt).content
        print("\nTest Generation Response:", response)  # Debug print
        
        imports, code = extract_code_sections(response)
        
        if not imports or not code:
            return Command(
                update={
                    "error": "Failed to extract code sections from response",
                    "debug_info": response
                },
                goto="handle_error"
            )
            
        complete_code = f"{imports}\n\n{code}"
        complete_code = format_python_code(complete_code)
        is_valid, error = validate_python_code(complete_code)
        
        if not is_valid:
            return Command(
                update={
                    "error": f"Invalid Python test code generated: {error}",
                    "debug_info": complete_code
                },
                goto="handle_error"
            )
        
        formatted_code = format_python_code(complete_code)
        
        return Command(
            update={"test_code": formatted_code},
            goto="write_files"
        )
            
    except Exception as e:
        return Command(
            update={
                "error": f"Error generating test code: {str(e)}",
                "debug_info": response if 'response' in locals() else None
            },
            goto="handle_error"
        )

def write_files(state: AgentState) -> AgentState:
    """Write the generated code to the specified files."""
    try:
        api_path = Path(state["api_file_path"])
        test_path = Path(state["test_file_path"])
        
        # Create directories if they don't exist
        api_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the files
        append_to_file_using_path(
            path=api_path,
            content=state["api_code"]
        )
        append_to_file_using_path(
            path=test_path,
            content=state["test_code"]
        )
        
        return state
        
    except Exception as e:
        return {**state, "error": f"Error writing files: {str(e)}"}

def handle_error(state: AgentState) -> AgentState:
    """Handle any errors that occurred during the process."""
    logging.error(f"Agent Error occurred: {state['error']}")
    if state.get('debug_info'):
        logging.error(f"\n Agent Debug information:\n{state['debug_info']}")
    
    return Command(
                update={},
                goto="generate_api_code"
            )

def git_operations(state: AgentState):
    """Commit and push changes to git repository."""
    try:
        import subprocess
        
        # Create commit message
        commit_message = f"feat: {state['feature_description']}"

        # Git add
        subprocess.run(["git", "add", state["api_file_path"], state["test_file_path"]], 
                      check=True, capture_output=True)
        
        # Git commit
        subprocess.run(["git", "commit", "-m", commit_message], 
                      check=True, capture_output=True)
        
        # Git push
        subprocess.run(["git", "push", "origin", "main"], 
                      check=True, capture_output=True)

        with sqlite3.connect(DB_CONNECTION_PATH, isolation_level=None) as con:
            con.execute(
                f"UPDATE feature_prompts SET is_implemented=TRUE WHERE id = {int(state['feature_id'])};"
            )
        
        return Command(goto=END)
        
    except subprocess.CalledProcessError as e:
        return Command(
            update={"error": f"Git operation failed: {e.stderr.decode()}"},
            goto="handle_error"
        )
    except Exception as e:
        return Command(
            update={"error": f"Git operation failed: {str(e)}"},
            goto="handle_error"
        )

def format_python_files(state: AgentState) -> Command[Literal["git_operations", "handle_error"]]:
    """Format Python files using black and sort imports."""
    try:
        for filepath in [state["api_file_path"], state["test_file_path"]]:
            with open(filepath) as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports, non_imports = [], []
            seen_imports = set()
            
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_str = ast.unparse(node)
                    if import_str not in seen_imports:
                        imports.append(import_str)
                        seen_imports.add(import_str)
                else:
                    non_imports.append(ast.unparse(node))
            
            formatted = '\n'.join(sorted(imports) + ['', ''] + non_imports)
            formatted = black.format_str(formatted, mode=black.FileMode())
            
            with open(filepath, 'w') as f:
                f.write(formatted)
        
        return Command(goto="git_operations")
    except Exception as e:
        return Command(
            update={"error": f"Error formatting files: {str(e)}"},
            goto="handle_error"
        )

# Create the graph
def create_api_agent() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("generate_api_code", generate_api_code)
    workflow.add_node("generate_tests", generate_tests)
    workflow.add_node("write_files", write_files)
    workflow.add_node("format_python_files", format_python_files)
    workflow.add_node("git_operations", git_operations)
    workflow.add_node("handle_error", handle_error)
    
    # Add edges
    workflow.add_edge(START, "generate_api_code")
    workflow.add_edge("write_files", "format_python_files")
    workflow.add_edge("format_python_files", "git_operations")
    workflow.add_edge("git_operations", END)
    workflow.add_edge("handle_error", END)
    
    return workflow.compile()

def start_agent_graph() -> None:
    # Setup logging first
    stderr_log = setup_logging()
    logging.info(f"Starting LLM Software Forge Factory")
    logging.info(f"Logs will be saved to:\nSTDERR: {stderr_log}")
    
    try:
        agent = create_api_agent()
        logging.info("Agent created successfully")

        db_feature_id: int = -1
        description: str = ""
        while True:
            logging.info("Connecting to database and changing to project directory")
            with sqlite3.connect(DB_CONNECTION_PATH, isolation_level=None) as con:
                db_feature_id, description = con.execute(
                    "SELECT id, description FROM feature_prompts WHERE is_implemented=FALSE LIMIT 1;"
                ).fetchone()
            
            os.chdir(os.path.expanduser("~/TessarXchange"))
            logging.info("Fetching unimplemented features from database")
            
            if not db_feature_id or not description:
                logging.warning("No features left to develop")
                raise ValueError("No features left to develop")
            
            logging.info(f"Processing feature ID {db_feature_id}: {description}")
            try:
                for e in agent.stream({
                    "feature_description": description,
                    "api_file_path": "app/endpoints.py",
                    "test_file_path": "tests/test_endpoints.py",
                    "feature_id": db_feature_id
                }):
                    pprint(e)
                    print()  # This will be captured in logs due to TeeWriter
                    logging.debug(f"Agent stream output: {e}")
                    
                logging.info(f"Completed feature ID {db_feature_id}")
                
            except Exception as e:
                logging.error(f"Error processing feature ID {db_feature_id}: {e}", exc_info=True)
                
    except Exception as e:
        logging.error(f"Fatal error in start_agent_graph: {e}", exc_info=True)
        raise
        
    finally:
        logging.info("Shutting down LLM Software Forge Factory")
        # No need to close stdout/stderr as TeeWriter keeps the original streams intact

if __name__ == "__main__":
    start_agent_graph()