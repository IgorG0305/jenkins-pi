# Use uma imagem base do Spark (Bitnami é uma boa opção)
FROM bitnami/spark:latest

# Mude para o usuário root para instalar pacotes
USER root

# Instalar dependências do sistema (Python, pip e libs de desenvolvimento)
# Adicione 'wget' para baixar o conector se não quiser copiar localmente
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Instalar bibliotecas Python necessárias para o script
# Adicionado 'pymysql' explicitamente, embora já estivesse
RUN pip3 install \
    pandas numpy matplotlib seaborn scikit-learn \
    imbalanced-learn plotly sqlalchemy pymysql nltk

# Baixar stopwords do NLTK (necessário para o script)
RUN python3 -m nltk.downloader stopwords

# --- Opção 1 (Recomendado): Copiar o JAR local ---
# Certifique-se de que 'mysql-connector-j-8.0.33.jar' está na pasta './spark'
COPY mysql-connector-j-8.0.33.jar /opt/bitnami/spark/jars/

# --- Opção 2: Baixar o JAR (Descomente se preferir) ---
# RUN wget https://repo1.maven.org/maven2/com/mysql/mysql-connector-j/8.0.33/mysql-connector-j-8.0.33.jar -P /opt/bitnami/spark/jars/

# Criar o diretório para os scripts, se não existir
RUN mkdir -p /opt/bitnami/spark/scripts && \
    chown -R 1001:1001 /opt/bitnami/spark/scripts

# Copiar o script Python para dentro do container
COPY projeto_bigdata.py /opt/bitnami/spark/scripts/projeto_bigdata.py

# Dar permissão ao usuário padrão para acessar/executar
RUN chown 1001:1001 /opt/bitnami/spark/scripts/projeto_bigdata.py
RUN chown 1001:1001 /opt/bitnami/spark/jars/mysql-connector-j-8.0.33.jar # Se copiou localmente

# Voltar ao usuário padrão do Bitnami (recomendado)
USER 1001

# Não defina um CMD aqui, pois o Jenkins usará 'docker exec'