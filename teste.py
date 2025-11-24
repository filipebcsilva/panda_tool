from agno.media import Image
from agno.agent import Agent
import base64
from MemoryTool import MemoryTools
from pydantic import BaseModel,Field
from agno.models.google import Gemini
from agno.workflow import Workflow,Step
from natsort import natsorted
from agents import saver
from workflow import vilma_workflow
import os


class LeitorOutuput(BaseModel):
    answers: dict = Field(...,
                               description="A dictionary that has the name of the word as its key and each of the questions as its value, and within each question the corresponding answer",
                        )
    
vision_model = Gemini(id = "gemini-2.5-flash",provider= "gemini",api_key = os.getenv("GEMINI_API_KEY"))

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
        # Lê cada linha e usa .strip() para remover o caractere de nova linha (\n)
        lista_palavras = [linha.strip() for linha in arquivo]


leitor = Agent(
    model = vision_model,
    name = "Word reader",
    description= "You are an AI agent specialized in analyzing words",
    instructions= """
        Given a word and a list of questions about the word, perform the following action for the word:
        Answer the questions about the word:
        1 - É uma fruta?
        2- É um animal?
        Answer the questions with yes or no
        Don't give ambiguous answers. If in doubt, choose only one direct answer.
        Instructions for output_schema:
        -Don't repeat the user's question more than once!
        -The name of each question must be the sabe given on the {input}
        -The dictionary must contais the name of the word
        -The dictionary must follow the json format with ""
    """,
    output_schema= LeitorOutuput,
    use_json_mode= True,
    debug_mode= True,
)


workflow = Workflow(
    name="Pandas pipeline",
    steps=[
        leitor,
        saver,
    ],
    debug_mode = True,
)


for palavra in lista_palavras:
    workflow.run(
        input = palavra, 
)
