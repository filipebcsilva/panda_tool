from agno.media import Image
from agno.agent import Agent
import base64
import json
from MemoryTool import MemoryTools
from pydantic import BaseModel,Field
from agno.models.google import Gemini
from agno.models.ollama import Ollama
from agno.tools.python import PythonTools
from agno.workflow import Workflow,Step,StepInput,StepOutput
from natsort import natsorted
from agents import saver
import re
from workflow import vilma_workflow
import os

code_model = Ollama(
    id = "qwen2.5-coder:14b",
    )
vision_model = Gemini(id = "gemini-2.5-flash",provider= "gemini",api_key = os.getenv("GEMINI_API_KEY"))
analista = Agent(
    model=vision_model,
    name="Data Analyst",
    description="You are a Python Data Analyst tailored for pandas operations.",
    
   instructions=[
        "GOAL: Answer questions based strictly on 'data.csv'.",
        
        "STEP 1: LOAD THE DATA",
        "   - Use `df = pd.read_csv('data.csv', index_col=0)`.",
        "   - The index (column 0) contains the Word names (e.g., 'Abacate', 'Zebra').",
        
        "STEP 2: UNDERSTAND THE SCHEMA (CRITICAL)",
        "   - The dataset has specific column names. You MUST use them exactly as written below:",
        "       * Column for Fruits: '1 - Is it a fruit?'",
        "       * Column for Animals: '2 - Is it a animal?'",
        
        "STEP 3: FILTERING LOGIC",
        "   - The values in these columns are 'yes' and 'no' (lowercase).",
        "   - To find FRUITS: Filter rows where `df['1 - Is it a fruit?'] == 'yes'`.",
        "   - To find ANIMALS: Filter rows where `df['2 - Is it a animal?'] == 'yes'`.",
        "   - Note: Some rows might be dirty or headers repeated. Filter valid 'yes' values only.",

        "STEP 4: OUTPUT",
        "   - You MUST generate a Python script.",
        "   - You MUST use `print()` inside the python script to show the results (counts, names, etc).",
        "   - Based on the print output, answer the user's question in text."
    ],
    
    tools=[PythonTools(),MemoryTools()], 
    debug_mode=True,
)

analista.print_response(input=["How many animals are on the database",
                               "How many fruits are on the database"]
                        )