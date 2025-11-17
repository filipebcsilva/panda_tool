from typing import Dict, List

from agno.agent import Agent    
from agno.tools import Toolkit
from agno.media import Image
from agno.workflow import Step, Workflow, StepOutput
from agno.tools.python import PythonTools
from pydantic import BaseModel,Field
import pandas as pd
from natsort import natsorted
from agno.models.google import Gemini
import os



# class LeitorInput(BaseModel):
#     answers: dict = Field(...,
#                                description="A dictionary that has the name of the image as its key and each of the questions as its value, and within each question the corresponding answer",
#                         )
class LeitorOutuput(BaseModel):
    answers: dict= Field(...,
                               description="A dictionary that has the name of the image as its key and each of the questions as its value, and within each question the corresponding answer",
                        )
    
class MemoryTools(Toolkit):
    def __init__(self, **kwargs):
        super().__init__(name="calculator_tools", tools=[self.save_memory,self.read_memory], **kwargs)
    
    def save_memory(self,data: dict):
        """
        Save a dictionary in a csv archive to work as memory
        
        Args:
            data(dict): a dictionary with the format 
            answers:
            {{'Imagem1': {'Pergunta1': 'RespostaG', 'Pergunta2': 'RespostaH'},
            'Imagem5': {'Pergunta1': 'RespostaI', 'Pergunta3': 'RespostaJ'}}
        
        """
        file_path = "data.csv"
    
        new_data_dict = data["answers"]
        
        temp = pd.DataFrame.from_dict(new_data_dict, orient='index')
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            memory = pd.read_csv(file_path, index_col=0)
        else:
            memory = pd.DataFrame()
    
        memory = pd.concat([memory, temp])
        
        memory.to_csv(file_path, index=True)
        
    def read_memory(self):
        """
        Read a csv file and transform into a pandas dataframe
        
        Returns:
            memory (dataframe): a dataframe with images, questions and answers
        """
        file_path = "data.csv"
        memory = pd.read_csv(file_path)
        return memory.to_string()

vision_model = Gemini(id = "gemini-2.5-flash",provider= "gemini",api_key = os.getenv("GEMINI_API_KEY"))

# answers = {
#     'answers':{
#     'Imagem1': {'Pergunta1': 'RespostaA', 'Pergunta2': 'RespostaB'},
#     'Imagem2': {'Pergunta1': 'RespostaC', 'Pergunta2': 'RespostaD'},
#     'Imagem3': {'Pergunta1': 'RespostaE', 'Pergunta2': 'RespostaF'}
#     }
# }


leitor = Agent(
    model = vision_model,
    name = "Image reader",
    description= "You are an AI agent specialized in analyzing and extracting information from images",
    instructions= """
        Given a list of images and the input of data that must be extracted, perform the following action for all images:
        Answer the sequence of questions individually for each image directly, for example:
        Example:
        -Question: How many buildings are in the image?
        -Answer: 8
        Don't give ambiguous answers. If in doubt, choose only one direct answer.
        Instructions for output_schema:
        -Don't repeat the user's question more than once!
        -The name of the image must be something like Image1, Image2...
    """,
    output_schema= LeitorOutuput,
    tools= [MemoryTools()],
    use_json_mode= True,
    debug_mode= True,
)

# agent_teste1 = Agent(tools=[MemoryTools()],
#               model = vision_model,
#               instructions = "Use the MemoryTools to save the dictionary on a csv file",
#               input_schema = LeitorInput,
#               debug_mode = True,
#             )
saver = Agent(tools=[MemoryTools()],
              model = vision_model,
              instructions = "Use the read_memory tool to read csv file and get a pandas dataframe on string format",
               debug_mode = True,
            )
# agent3 = Agent(tools = [PythonTools()],
#                model = vision_model,
#                instructions = "Write a python code that extract the answer of the Pergunta 2 from imagem1",
#                 debug_mode = True,
#                )

workflow = Workflow(
    name="Pandas pipeline",
    steps=[
        leitor,   
        saver,
    ],
    debug_mode = True,
)


caminho_da_pasta = 'imagens'


lista_de_imagens = []

# 1. Define o caminho absoluto da pasta para evitar ambiguidades
# Se 'caminho_da_pasta' for relativo (ex: 'imagens'), isso o torna completo
caminho_absoluto_pasta = os.path.abspath(caminho_da_pasta)

# Verifica se a pasta existe antes de listar
if os.path.exists(caminho_absoluto_pasta):
    
    nomes_dos_arquivos = natsorted([
        f for f in os.listdir(caminho_absoluto_pasta) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg')) # Adicionei .lower() para pegar .JPG ou .PNG
    ])

    for nome_do_arquivo in nomes_dos_arquivos:
        caminho_completo = os.path.join(caminho_absoluto_pasta, nome_do_arquivo)
        
        # Debug: Mostra o que está sendo processado
        print(f"Adicionando imagem: {caminho_completo}")

        # Cria o objeto Image com o caminho absoluto
        imagem = Image(filepath=caminho_completo)
        
        if imagem:
            lista_de_imagens.append(imagem)
else:
    print(f"Erro: A pasta {caminho_absoluto_pasta} não foi encontrada.")


input = ["Quantidade de pessoas"]
workflow.print_response(input = input,images=lista_de_imagens)