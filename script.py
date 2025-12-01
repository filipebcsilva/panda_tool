import pandas as pd
import random

# --- CONFIGURAÇÃO ---
QTD_LINHAS = 2000  # Pode aumentar para 10.000 ou 100.000 se quiser estressar mesmo
# --------------------

print(f"🚀 Gerando dataset com {QTD_LINHAS} linhas...")

# Listas para gerar dados aleatórios variados
prefixos = ['viagem_praia', 'centro_urbano', 'salada_frutas', 'transito_sp', 
            'show_rock', 'retrato_familia', 'floresta_amazonica', 'mercado_livre',
            'animal_silvestre', 'evento_esportivo', 'selfie_espelho']

cores = ['Blue', 'Red', 'Green', 'Yellow', 'Black', 'White', 'Purple', 'Orange', 'Grey', 'Pink']

dados = []

for i in range(1, QTD_LINHAS + 1):
    # 1. Gera nome do arquivo (Index)
    prefixo = random.choice(prefixos)
    nome_arquivo = f"{prefixo}_{i:05d}.jpg"  # Ex: viagem_praia_00001.jpg
    
    # 2. Lógica para 'É fruta?' (Se tiver 'fruta' ou 'mercado' no nome, chance alta de ser yes)
    if 'fruta' in prefixo or 'mercado' in prefixo:
        is_fruit = 'yes' if random.random() > 0.2 else 'no' # 80% de chance de ser sim
    else:
        is_fruit = 'yes' if random.random() < 0.05 else 'no' # 5% de chance de falso positivo (sujeira)

    # 3. Lógica para Pessoas (Alguns com 0, alguns com multidão)
    r = random.random()
    if r < 0.3:
        n_people = 0  # 30% de fotos vazias
    elif r < 0.9:
        n_people = random.randint(1, 10) # Maioria tem poucas pessoas
    else:
        n_people = random.randint(50, 5000) # 10% são multidões (shows, eventos)

    # 4. Lógica para Carros
    if 'transito' in prefixo or 'urbano' in prefixo:
        n_cars = random.randint(10, 300)
    else:
        n_cars = random.randint(0, 5)

    # 5. Cor Predominante
    color = random.choice(cores)

    # Adiciona a linha
    dados.append({
        'Nome_Arquivo': nome_arquivo,
        '1 - Is it a fruit?': is_fruit,
        '2 - Number of people': n_people,
        '3 - Dominant Color': color,
        '4 - Number of cars': n_cars
    })

# Cria DataFrame
df = pd.DataFrame(dados)
df.set_index('Nome_Arquivo', inplace=True)

# Salva
df.to_csv('teste.csv')

print(f"✅ Sucesso! Arquivo 'data.csv' criado com {len(df)} registros.")
print(f"📊 Tamanho estimado: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
print("\n--- Amostra das primeiras 5 linhas ---")
print(df.head())