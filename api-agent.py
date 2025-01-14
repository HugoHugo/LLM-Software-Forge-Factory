from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, START
from langgraph.types import Command
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import ast
import black
import pytest
from pathlib import Path

# Define the state that will be passed between nodes
class AgentState(TypedDict):
    feature_description: str
    api_file_path: str
    test_file_path: str
    api_code: Optional[str]
    test_code: Optional[str]
    error: Optional[str]

# Output schemas for the LLM
class APIImplementation(BaseModel):
    code: str = Field(description="The Python code implementing the API feature")
    imports: str = Field(description="Required import statements")
    
class TestImplementation(BaseModel):
    code: str = Field(description="The Python test code for the API feature")
    imports: str = Field(description="Required import statements")

# Initialize LlamaCpp with streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Initialize LLM
llm = LlamaCpp(
    model_path="/path/to/your/gguf/model.gguf",  # Replace with your model path
    temperature=0.7,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,
    n_ctx=4096  # Context window size
)

def validate_python_code(code: str) -> tuple[bool, Optional[str]]:
    """Validate if the generated code is syntactically correct Python."""
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def format_python_code(code: str) -> str:
    """Format Python code using black."""
    try:
        return black.format_str(code, mode=black.FileMode())
    except Exception as e:
        return code

def generate_api_code(state: AgentState) -> Command[Literal["generate_tests", "handle_error"]]:
    """Generate API implementation code based on feature description."""
    template = """You are an expert Python developer. Generate a FastAPI implementation for the following feature:

{feature_description}

The response should contain two sections:
1. Required imports
2. Implementation code

Please format your response as follows:
IMPORTS:
<imports>
FROM fastapi import FastAPI
...other imports...
</imports>

CODE:
<code>
# Your implementation here
...
</code>"""
    
    prompt = PromptTemplate.from_template(template)
    
    try:
        response = llm.invoke(prompt.format(feature_description=state["feature_description"]))
        
        # Parse the response
        imports_start = response.find("<imports>") + 9
        imports_end = response.find("</imports>")
        code_start = response.find("<code>") + 6
        code_end = response.find("</code>")
        
        if any(x == -1 for x in [imports_start, imports_end, code_start, code_end]):
            return Command(
                update={"error": "Failed to parse LLM response - invalid format"},
                goto="handle_error"
            )
            
        imports = response[imports_start:imports_end].strip()
        code = response[code_start:code_end].strip()
        
        complete_code = f"{imports}\n\n{code}"
        is_valid, error = validate_python_code(complete_code)
        
        if not is_valid:
            return Command(
                update={"error": f"Invalid Python code generated: {error}"},
                goto="handle_error"
            )
            
        formatted_code = format_python_code(complete_code)
        
        return Command(
            update={"api_code": formatted_code},
            goto="generate_tests"
        )
        
    except Exception as e:
        return Command(
            update={"error": f"Error generating API code: {str(e)}"},
            goto="handle_error"
        )

def generate_tests(state: AgentState) -> Command[Literal["write_files", "handle_error"]]:
    """Generate test code for the API implementation."""
    template = """You are an expert in Python testing. Generate pytest tests for the following API implementation.

API Code:
{api_code}

Feature Description:
{feature_description}

The response should contain two sections:
1. Required imports
2. Test implementation

Please format your response as follows:
IMPORTS:
<imports>
import pytest
from fastapi.testclient import TestClient
...other imports...
</imports>

CODE:
<code>
# Your test implementation here
...
</code>"""
    
    prompt = PromptTemplate.from_template(template)
    
    try:
        response = llm.invoke(prompt.format(
            api_code=state["api_code"],
            feature_description=state["feature_description"]
        ))
        
        # Parse the response
        imports_start = response.find("<imports>") + 9
        imports_end = response.find("</imports>")
        code_start = response.find("<code>") + 6
        code_end = response.find("</code>")
        
        if any(x == -1 for x in [imports_start, imports_end, code_start, code_end]):
            return Command(
                update={"error": "Failed to parse LLM response - invalid format"},
                goto="handle_error"
            )
            
        imports = response[imports_start:imports_end].strip()
        code = response[code_start:code_end].strip()
        
        complete_code = f"{imports}\n\n{code}"
        is_valid, error = validate_python_code(complete_code)
        
        if not is_valid:
            return Command(
                update={"error": f"Invalid Python test code generated: {error}"},
                goto="handle_error"
            )
            
        formatted_code = format_python_code(complete_code)
        
        return Command(
            update={"test_code": formatted_code},
            goto="write_files"
        )
        
    except Exception as e:
        return Command(
            update={"error": f"Error generating test code: {str(e)}"},
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
        api_path.write_text(state["api_code"])
        test_path.write_text(state["test_code"])
        
        return state
        
    except Exception as e:
        return {**state, "error": f"Error writing files: {str(e)}"}

def handle_error(state: AgentState) -> AgentState:
    """Handle any errors that occurred during the process."""
    print(f"Error occurred: {state['error']}")
    return state

# Create the graph
def create_api_agent() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("generate_api_code", generate_api_code)
    workflow.add_node("generate_tests", generate_tests)
    workflow.add_node("write_files", write_files)
    workflow.add_node("handle_error", handle_error)
    
    # Add edges
    workflow.add_edge(START, "generate_api_code")
    workflow.add_edge("write_files", END)
    workflow.add_edge("handle_error", END)
    
    return workflow.compile()

# Usage example
if __name__ == "__main__":
    agent = create_api_agent()
    
    # Example usage
    result = agent.invoke({
        "feature_description": "Create an endpoint that returns the current server time in ISO format",
        "api_file_path": "app/endpoints/time.py",
        "test_file_path": "tests/test_time.py"
    })
