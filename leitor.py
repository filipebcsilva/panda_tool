from agno.media import Image
from agno.agent import Agent
from MemoryTool import MemoryTools
from agno.models.google import Gemini
from agno.models.ollama import Ollama
from agno.tools.python import PythonTools
import pandas as pd
import os

vision_model = Gemini(id = "gemini-2.5-flash",provider= "gemini",api_key = os.getenv("GEMINI_API_KEY"))

try:
    df_meta = pd.read_csv('teste.csv', index_col=0, nrows=0)
    lista_colunas = df_meta.columns.tolist()
    info_contexto = f"AS COLUNAS DO ARQUIVO SÃO: {lista_colunas}"
except:
    info_contexto = "Arquivo teste.csv ainda não existe."
    
analista = Agent(
    model=vision_model,
    name="Data Analyst",
    description="You are a Pandas Expert specialized in writing robust, defensive code for data analysis.",
    
   instructions=[
      f"CONTEXT: {info_contexto}",
        
        "GOAL: Answer user questions by writing and running Python scripts using the PythonTools avaiable.",

        "RULES: "
        "1. Try to install the pandas package before writing the code",
        "2. You MUST write the script, don't forget to do that",
        "3. After installing the pandas package, folow the steps:"
        
        "STRATEGY FOR MIXED DATA TYPES (Critical):",
        
        "1. LOADING:",
        "   - `df = pd.read_csv('teste.csv', index_col=0)`",
        
        "2. ANALYZE THE QUESTION TO DECIDE THE DATA TYPE:",
        
        "   Type A: NUMERIC QUESTIONS (How many, Average, Sum, Max, > 5)",
        "       - ACTION: You MUST convert the column to numeric first.",
        "       - CODE: `df['col'] = pd.to_numeric(df['col'], errors='coerce')`",
        "       - REASON: This handles dirty data (e.g., text mixed with numbers) by turning it into NaN.",
        
        "   Type B: CATEGORICAL QUESTIONS (Color, Type, Main object, Yes/No)",
        "       - ACTION: Treat as string. Do NOT use to_numeric.",
        "       - CODE: `df['col'] = df['col'].astype(str).str.lower().str.strip()`",
        "       - OPS: Use `.value_counts()`, `.unique()` or string filtering (`==`).",
        
        "3. COMPLEX FILTERING (Combining types):",
        "   - Example: 'Show images with > 2 people AND Blue color'.",
        "   - Logic: Apply Type A logic to the 'people' column AND Type B logic to the 'color' column.",
        "   - Code: `df[(df['people'] > 2) & (df['color'] == 'blue')]`.",
        
        "4. OUTPUT:",
        "   - Always use `print()` to display the result.",

    ],
    
    tools=[PythonTools()], 
    debug_mode=True,
)

input = [
    "Qual o número total somado de pessoas em todas as 2000 imagens?",
    "Quantas imagens têm mais de 1000 pessoas (multidões)?",
    "Qual é a média de pessoas por imagem?",
    "Qual é a cor predominante mais frequente no dataset inteiro e quantas vezes ela aparece?",
    "Existem imagens que são 'frutas' MAS que têm carros aparecendo? Se sim, liste os nomes.", 
    "Qual a média de carros apenas nas imagens que têm 'urbano' ou 'transito' no nome?"
]

analista.print_response(input=input
                        )