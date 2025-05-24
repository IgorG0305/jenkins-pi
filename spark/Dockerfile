FROM python:3.10-slim

# Instala dependências
RUN pip install --no-cache-dir \
    pandas \
    seaborn \
    matplotlib \
    scikit-learn \
    imbalanced-learn \
    plotly \
    sqlalchemy \
    nltk \
    pymysql \
    pyspark

# Baixa stopwords
RUN python -m nltk.downloader stopwords

# Copia o script
COPY projeto_bigdata.py /opt/spark/scripts/projeto_bigdata.py

WORKDIR /opt/spark/scripts

CMD ["python3", "projeto_bigdata.py"]
