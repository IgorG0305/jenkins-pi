FROM bitnami/spark:latest

# Instalar dependências extras se quiser (exemplo com pip)
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install pandas pyspark

# Copiar scripts para dentro do container
COPY ./scripts /opt/spark/scripts

# Comando padrão (pode ser alterado no docker-compose)
CMD ["/opt/bitnami/scripts/spark/entrypoint.sh", "/run.sh"]
