# Use a imagem base do Python para Flask
FROM python:3.9-slim

# Configuração da variável de ambiente para evitar erros de buffering no terminal
ENV PYTHONUNBUFFERED 1

# Define o diretório de trabalho dentro do contêiner
WORKDIR /app

# Instala curl e nmap
RUN apt-get update && apt-get install -y \
    curl \
    nmap \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copia o código da aplicação para o contêiner
COPY ./app .

# Cria a pasta img
RUN mkdir -p /app/img

# Instala as dependências necessárias
# RUN pip install --no-cache-dir -r /app/yolo/requirements.txt
RUN pip install --no-cache-dir flask opencv-python-headless numpy

# Abre a porta que a aplicação Flask vai escutar
EXPOSE 5000

# Comando para rodar a aplicação Flask quando o contêiner iniciar
CMD ["python", "/app/main.py"]
