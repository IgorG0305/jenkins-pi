# Use uma imagem base do Spark (Bitnami é uma boa opção)
FROM bitnami/spark:latest

# Mude para o usuário root para instalar pacotes
USER root

# Instalar dependências do sistema (Python, pip, libs de desenvolvimento e wget)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instalar bibliotecas Python necessárias para o script
RUN pip3 install \
    pandas numpy matplotlib seaborn scikit-learn \
    imbalanced-learn plotly sqlalchemy pymysql nltk cryptography


# Baixar stopwords do NLTK (necessário para o script)
RUN python3 -m nltk.downloader stopwords

# --- COPIAR O JAR LOCAL (ESSA É A LINHA QUE FALTAVA!) ---
# Certifique-se de que 'mysql-connector-j-8.2.0.jar' está na pasta './spark'
COPY mysql-connector-j-8.2.0.jar /opt/bitnami/spark/jars/
RUN chown 1001:1001 /opt/bitnami/spark/jars/mysql-connector-j-8.2.0.jar

# Criar o diretório para os scripts, se não existir
RUN mkdir -p /opt/bitnami/spark/scripts && \
    mkdir -p /opt/bitnami/spark/jars && \
    chown -R 1001:1001 /opt/bitnami/spark/scripts

# Copiar o script Python para dentro do container
COPY projeto_bigdata.py /opt/bitnami/spark/scripts/projeto_bigdata.py


# Dar permissão ao usuário padrão para acessar/executar
COPY projeto_bigdata.py /opt/bitnami/spark/scripts/projeto_bigdata.py
RUN chown 1001:1001 /opt/bitnami/spark/scripts/projeto_bigdata.py
RUN chown 1001:1001 /opt/bitnami/spark/jars/mysql-connector-j-8.2.0.jar

# Copia e dá permissão ao script1.sh (se ele for realmente usado)
COPY script1.sh /opt/bitnami/spark/scripts/script1.sh
RUN chmod +x /opt/bitnami/spark/scripts/script1.sh && \
    chown 1001:1001 /opt/bitnami/spark/scripts/script1.sh

# Voltar ao usuário padrão do Bitnami (recomendado)
USER 1001

# Não defina um CMD aqui.