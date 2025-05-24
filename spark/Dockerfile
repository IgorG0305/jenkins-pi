FROM bitnami/spark:latest

# Instala dependências extras (opcional)
RUN apt-get update && apt-get install -y python3-pip \
    && pip3 install pandas seaborn matplotlib scikit-learn imbalanced-learn plotly sqlalchemy nltk pymysql

# Baixar recursos do nltk
RUN python3 -m nltk.downloader stopwords

# Copia o script Python para dentro do container
COPY projeto_bigdata.py /opt/spark/scripts/projeto_bigdata.py

# Comando padrão (altere se quiser rodar automaticamente)
CMD ["bash"]
