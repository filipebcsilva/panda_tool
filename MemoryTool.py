from agno.tools import Toolkit
import pandas as pd
import os

       
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
