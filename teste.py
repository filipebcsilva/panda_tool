from agno.media import Image
from agno.agent import Agent
import base64
import json
from typing import Dict, Any
from pydantic import BaseModel,Field
from MemoryTool import MemoryTools
from agno.models.ollama import Ollama
from agno.workflow import Workflow,Step,StepInput,StepOutput
import re
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


questions = ["Is it a fruit",
             "Is it a animal"]

saver = Agent(
    tools=[MemoryTools()],
    model = text_model,
    description = "You are a Agent that uses a tool to save a dictionary on a csv file",
    instructions = [
                "You have acces to a save_memory tool",
                "Always call the tool to save the data",
                "Do not describe things, just call the tool with the input dictionary, dont change its format"
                ],
    debug_mode = True,
            )

leitor = Agent(
    model = text_model,
    name = "Word reader",
    description= "You are an AI agent specialized in analyzing words",
    instructions=[
        "TASK: Answer the questions below regarding the input word/image.",
        f"QUESTIONS:\n{questions}",
        
        "OUTPUT FORMAT (CRITICAL):",
        "1. The output must be a dictionary and the keys must be the exact questions listed above.",
        
        "DATA STANDARDS (Apply based on context):",
        "   - QUANTITIES: If the answer is a number, return a PURE INTEGER or FLOAT (e.g., 5, 3.5). Never add units like '5 kg' or '3 people'.",
        "   - BOOLEANS: If the answer is Yes/No, use lowercase 'yes' or 'no'.",
        "   - TEXT: Keep it concise and lowercase (e.g., 'blue', 'metal').",
        
    ],
    output_schema= LeitorOutuput,
    use_json_mode= True,
    debug_mode= True,
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
