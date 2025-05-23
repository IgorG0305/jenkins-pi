FROM r-base

WORKDIR /app

# Instalar dependências do sistema
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
    libmariadb-dev \
    && apt-get clean

# Instalar pacotes R
RUN R -e "install.packages(c('rmarkdown', 'RMariaDB', 'DBI', 'tools', 'dplyr', 'tidyr'), repos='https://cloud.r-project.org/')"

# Copiar o script
COPY limpeza.r /app/limpeza.r

# Comando de execução padrão
CMD ["Rscript", "limpeza.r"]
