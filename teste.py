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


class LeitorOutuput(BaseModel):
    answers: dict = Field(...,
                               description="A dictionary with the questions and the answer to each question",
                        )
    
# vision_model = Gemini(id = "gemini-2.5-flash",provider= "gemini",api_key = os.getenv("GEMINI_API_KEY"))
text_model = Ollama(
    id="llama3.1:8b",
    options={
            "num_predict": 4096,
            "num_ctx": 8192    
        }
    )
code_model = Ollama(
    id = "qwen2.5-coder:14b",
    )
def insert_name_word(step_input: StepInput) -> StepOutput:
    """
    Insere o nome da palavra no resultado da análise.
    Blindada contra erros de tipo (str vs dict vs object).
    """

    raw_text = step_input.previous_step_content
    
    raw_text = raw_text.answers
    
    if not raw_text:
        return StepOutput(step_name="insert_name_word", content="{}", success=False)

    data_dict = {}
    
    if isinstance(raw_text, str):
        clean_text = re.sub(r'```json\s*|```python\s*|\s*```', '', raw_text).strip()
        
        try:
            data_dict = json.loads(clean_text)
        except:
            try:
                data_dict = eval(clean_text)
            except:
                data_dict = {
                    "analise_texto": clean_text
                }
    
    elif isinstance(raw_text, dict):
        data_dict = raw_text

    name_word = step_input.additional_data.get("word_name", "palavra_desconhecida")

    conteudo = data_dict.get("answers", data_dict)

    final_structure = {
        "answers": {
            name_word: conteudo
        }
    }
    
    return StepOutput(
        step_name="insert_name_word", 
        content=json.dumps(final_structure, ensure_ascii=False), 
        success=True,
    )
    
def image_base64(absolut_path,name_file) -> Image:
    
    caminho_completo = os.path.join(absolut_path, name_file)
    try:
        with open(caminho_completo, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')

            imagem = Image(
                content=base64_string 
            )
            return imagem
    except Exception as e:
        print(f"Erro ao processar {name_file}: {e}")


arquivo_nome = 'palavras.txt'
lista_de_imagens = []

with open(arquivo_nome, 'r', encoding='utf-8') as arquivo:
        lista_palavras = [linha.strip() for linha in arquivo]


leitor = Agent(
    model = text_model,
    name = "Word reader",
    description= "You are an AI agent specialized in analyzing words",
    instructions= """
        Given a word and a list of questions about the word, perform the following action for the word:
        Answer the questions about the word:
        1 - Is it a fruit?
        2 - Is it a animal?
        Always answer the question on this order
        Answer the questions with yes or no
        Don't give ambiguous answers. If in doubt, choose only one direct answer.
        Instructions for output_schema:
        -Don't repeat the user's question more than once!
        -The dictionary must follow a json format
        -Respond in English
    """,
    output_schema= LeitorOutuput,
    use_json_mode= True,
    debug_mode= True,
)

analista = Agent(
    model=code_model,
    name="Data Analyst",
    description="You are a Python Data Analyst tailored for pandas operations.",
    
    instructions=[
        "STRATEGY: You perform analysis by WRITING and EXECUTING python code.",
        
        "DATA SOURCE:",
        "   - Target file: 'data.csv' (in the current directory).",
        "   - Structure: The first column (index 0) contains the Item Names (e.g., 'Abóbora').",
        "   - Format: Use `df = pd.read_csv('data.csv', index_col=0)`.",
        
        "IMPORTANT RULES:",
        "   1. DO NOT try to read the file content into the chat conversation.",
        "   2. USE PANDAS methods (value_counts, query, shape) to find answers.",
        "   3. PRINT the result inside the python script.",
        
        "Example:",
        "   User: 'How many fruits?'",
        "   You write:",
        "   import pandas as pd",
        "   df = pd.read_csv('data.csv', index_col=0)",
        "   print(df[df['answers'].str.contains('fruit', case=False)].shape[0])"
    ],
    
    tools=[PythonTools()], 
    debug_mode=True,
)

insert_name_step = Step(name = "insert name word",executor=insert_name_word)

workflow = Workflow(
    name="Pandas pipeline",
    steps=[
        leitor,
        insert_name_step,
        saver,
    ],
    debug_mode = True,
)


for palavra in lista_palavras:
    workflow.run(
        input = palavra, 
        additional_data={"word_name": palavra}
)
