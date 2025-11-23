from agno.media import Image
import base64
from natsort import natsorted
from agents import gerador_perguntas
from workflow import vilma_workflow
import os

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


caminho_da_pasta = 'imagens'
lista_de_imagens = []

caminho_absoluto_pasta = os.path.abspath(caminho_da_pasta)

nomes_dos_arquivos = natsorted([
    f for f in os.listdir(caminho_absoluto_pasta) 
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
])


input = ["Qual é a média de pessoas por imagem",
         "Qual é a média de carros por imagem"]

reposta_gerador = gerador_perguntas.run(input=input)

input = reposta_gerador.content.dados

for nome in nomes_dos_arquivos:
    image = image_base64(caminho_absoluto_pasta,nome)
    
    vilma_workflow.print_response(
        input = input, 
        images = [image],
        additional_data={
            "image_name": nome
        }
)
