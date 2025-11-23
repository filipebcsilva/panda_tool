from typing import List
from agno.agent import Agent    
from pydantic import BaseModel,Field
from agno.models.google import Gemini
from MemoryTool import MemoryTools
import os

class LeitorOutuput(BaseModel):
    answers: dict = Field(...,
                               description="A dictionary that has the name of the image as its key and each of the questions as its value, and within each question the corresponding answer",
                        )
class GeradorOutput(BaseModel):
    dados: List[str] = Field(...,
                                description= "A list of relevant information that should be extracted from the images"
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

saver = Agent(tools=[MemoryTools()],
              model = vision_model,
              instructions = "Use the MemoryTools to save the dictionary on a csv file",
              debug_mode = True,
            )
