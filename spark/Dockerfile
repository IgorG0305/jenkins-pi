FROM bitnami/spark:latest

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install pandas pyspark seaborn matplotlib scikit-learn imbalanced-learn plotly sqlalchemy nltk pymysql

RUN python3 -m nltk.downloader stopwords

RUN mkdir -p /opt/spark/scripts

COPY projeto_bigdata.py /opt/spark/scripts/projeto_bigdata.py

CMD ["bash"]
