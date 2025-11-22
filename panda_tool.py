from typing import Dict, List

from agno.agent import Agent    
from agno.tools import Toolkit
from agno.media import Image
import base64
import json
from agno.workflow import Step, Workflow, StepOutput,StepInput
from agno.tools.python import PythonTools
from pydantic import BaseModel,Field
import pandas as pd
from natsort import natsorted
from agno.models.google import Gemini
import os

class LeitorOutuput(BaseModel):
    answers: dict = Field(...,
                               description="A dictionary that has the name of the image as its key and each of the questions as its value, and within each question the corresponding answer",
                        )
class GeradorOutput(BaseModel):
    dados: List[str] = Field(...,
                                description= "A list of relevant information that should be extracted from the images"
                                )
        
class MemoryTools(Toolkit):
    def __init__(self, **kwargs):
        super().__init__(name="memory_tools", tools=[self.save_memory,self.read_memory], **kwargs)
    
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
        
        answers = data.get("answers")

        if answers is not None:
            temp = pd.DataFrame.from_dict(answers, orient='index')
        else:
            temp = pd.DataFrame.from_dict(data, orient='index')
        
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

def insert_name_image(step_input: StepInput) -> StepOutput:
    """
    Custom function that creates a report using data from multiple previous steps.
    This function has access to ALL previous step outputs and the original workflow message.
    """
    message = step_input.previous_step_content

    name_image = step_input.additional_data.get("image_name")
    
    answers = {
        "answers": {
            name_image: message.answers
        }
    }
    
    answers_str = json.dumps(answers,ensure_ascii=False)
    return StepOutput(
        step_name="insert_name_images", content=answers_str, success=True
    )


    
vision_model = Gemini(id = "gemini-2.5-flash",provider= "gemini",api_key = os.getenv("GEMINI_API_KEY"))


gerador_perguntas = Agent(
    model = vision_model,
    name = "Question generator",
    description= "You are an AI agent that specializes in deciding what to extract from an image based on a series of questions.",
    instructions= """
            Given a list of user questions, create a list of information to extract from all the images.
            Analyse the question and think what information tou need to extract from the image to answer that question.
            For each one of the question, you MUST give only one information.
            For example:
            INPUT: How many buildings are visible in the image.
            OUTPUT: Number of buildings.
            Don't repeat the user's question, you must decide what information is needed to answer the question
            Return a list of information that must be extracted   
    """,
    output_schema= GeradorOutput,
    debug_mode= True,
)

leitor = Agent(
    model = vision_model,
    name = "Image reader",
    description= "You are an AI agent specialized in analyzing and extracting information from images",
    instructions= """
        Given a image and the input of data that must be extracted, perform the following action for the image:
        Answer the sequence of questions for the image directly, for example:
        Example:
        -Question: How many buildings are in the image?
        -Answer: 8
        Don't give ambiguous answers. If in doubt, choose only one direct answer.
        Instructions for output_schema:
        -Don't repeat the user's question more than once!
        -The name of each question must be the sabe given on the {input}
    """,
    output_schema= LeitorOutuput,
    use_json_mode= True,
    debug_mode= True,
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
    
    
saver = Agent(tools=[MemoryTools()],
              model = vision_model,
              instructions = "Use the MemoryTools to save the dictionary on a csv file",
              debug_mode = True,
            )
printer = Agent(tools=[MemoryTools()],
              model = vision_model,
              instructions = "Use the read_memory tool to read csv file and get a pandas dataframe on string format and show the dataframe",
               debug_mode = True,
            )

caminho_da_pasta = 'imagens'
lista_de_imagens = []

caminho_absoluto_pasta = os.path.abspath(caminho_da_pasta)

nomes_dos_arquivos = natsorted([
    f for f in os.listdir(caminho_absoluto_pasta) 
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])

workflow = Workflow(
    name="Pandas pipeline",
    steps=[
        leitor,
        Step(executor = insert_name_image),
        saver,
    ],
    debug_mode = True,
)


input = ["Qual é a média de pessoas por imagem",
         "Qual é a média de carros por imagem"]

reposta_gerador = gerador_perguntas.run(input=input)

input = reposta_gerador.content.dados

for nome in nomes_dos_arquivos:
    image = image_base64(caminho_absoluto_pasta,nome)
    
    workflow.print_response(
        input = input, 
        images = [image],
        additional_data={
            "image_name": nome
        }
    )