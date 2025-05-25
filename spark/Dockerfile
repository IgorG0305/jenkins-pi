FROM bitnami/spark:latest

USER root

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y python3-pip python3-dev libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalar bibliotecas Python necessárias
RUN pip3 install \
    pandas numpy matplotlib seaborn scikit-learn imbalanced-learn plotly \
    sqlalchemy nltk pymysql

# Baixar stopwords do NLTK
RUN python3 -m nltk.downloader stopwords

# Copiar os scripts para dentro do container
COPY projeto_bigdata.py /opt/spark/scripts/projeto_bigdata.py
COPY script1.sh /opt/spark/scripts/script1.sh

# Permitir execução
RUN chmod +x /opt/spark/scripts/script1.sh

# Entrypoint padrão
CMD ["/bin/bash"]
