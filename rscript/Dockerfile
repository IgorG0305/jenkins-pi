FROM r-base

WORKDIR /app

# Instala dependências do sistema para o rmarkdown e outras libs comuns
RUN apt-get update && apt-get install -y \
    pandoc \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libzmq3-dev \
    libgit2-dev \
    libpng-dev \
    libjpeg-dev \
    libfreetype6-dev \
    && apt-get clean

# Instala o pacote rmarkdown (e outros se quiser)
RUN R -e "install.packages('rmarkdown', repos='http://cran.rstudio.com')"

# Copia os arquivos necessários
COPY /rscript/limpeza.r /app/limpeza.r
COPY /backend/alunos_com_erros.csv /app/alunos_com_erros.csv

# Comando para rodar o script e manter o container ativo
CMD ["bash", "-c", "Rscript /app/limpeza.r && tail -f /dev/null"]
