pipeline {
    agent any

    environment {
        DOCKER_HUB_USER = 'superlike1'
        IMAGE_GERADOR   = "${DOCKER_HUB_USER}/gerador-app"
        IMAGE_RSCRIPT   = "${DOCKER_HUB_USER}/rscript-app"
        IMAGE_MYSQL     = "${DOCKER_HUB_USER}/mysql-app"
        IMAGE_FLASK     = "${DOCKER_HUB_USER}/flask-app"
        IMAGE_BACKEND   = "${DOCKER_HUB_USER}/backend-app"
        IMAGE_SPARK     = "${DOCKER_HUB_USER}/spark-app"
        MYSQL_JAR_NAME  = "mysql-connector-j-8.0.33.jar" // <-- IMPORTANTE: Use o nome exato do seu JAR
        SPARK_JAR_PATH  = "/opt/bitnami/spark/jars/${MYSQL_JAR_NAME}"
        PYTHON_SCRIPT_PATH = "/opt/bitnami/spark/scripts/projeto_bigdata.py"
    }

    stages {
        stage('Build Imagens') {
            steps {
                script {
                    echo "Construindo imagens com --no-cache..."
                    // Assumindo que os Dockerfiles estão nas pastas corretas
                    sh "docker build --no-cache -t ${IMAGE_GERADOR}:latest ./backend" // Verifique se gerador usa backend
                    sh "docker build --no-cache -t ${IMAGE_RSCRIPT}:latest ./rscript"
                    sh "docker build --no-cache -t ${IMAGE_MYSQL}:latest ./mysql"
                    sh "docker build --no-cache -t ${IMAGE_FLASK}:latest ./api"
                    sh "docker build --no-cache -t ${IMAGE_BACKEND}:latest ./backend"
                    sh "docker build --no-cache -t ${IMAGE_SPARK}:latest ./spark" // <-- Corrigido o caminho
                }
            }
        }

        stage('Subir Serviços') {
            steps {
                script {
                    echo "Removendo containers do docker-compose..."
                    sh 'docker-compose down -v --remove-orphans || true' // Adicionado --remove-orphans e || true

                    echo "Subindo todos os serviços necessários..."
                    sh 'docker-compose up -d db flaskapi backend frontend grafana prometheus loki spark-master spark-worker'

                    echo "Aguardando 30 segundos para estabilização dos serviços (especialmente Spark)..."
                    sh 'sleep 30' // <-- Tempo extra para o Spark Master/Worker

                    echo "Aguardando banco de dados ficar pronto..."
                    sh '''
                    for i in {1..15}; do
                        MYSQL_CONTAINER=$(docker ps --format '{{.Names}}' | grep "mysql_db" | head -n1)
                        if [ ! -z "$MYSQL_CONTAINER" ] && docker exec $MYSQL_CONTAINER mysqladmin ping -h"localhost" --silent; then
                            echo "MySQL está respondendo ao PING!"
                            break
                        fi
                        echo "Aguardando MySQL ($i/15)..."
                        sleep 5
                    done
                    # Verifica se saiu do loop porque o MySQL respondeu
                    if ! docker exec $MYSQL_CONTAINER mysqladmin ping -h"localhost" --silent; then
                        echo "ERRO: MySQL não ficou pronto a tempo!"
                        currentBuild.result = 'FAILURE'
                        error "MySQL não iniciou."
                    fi
                    '''
                    echo "MySQL respondeu ao ping. Aguardando 10 segundos extras..."
                    sh 'sleep 10'
                }
            }
        }

        stage('Processo Iterativo') {
            steps {
                script {
                    int totalLotes = 2
                    for (int i = 1; i <= totalLotes; i++) {
                        echo "=== Iteração ${i} de ${totalLotes} ==="
                        echo "Executando gerador (1000 alunos)..."
                        sh 'docker-compose run --rm gerador'
                        echo "Executando processamento R..."
                        sh 'docker-compose run --rm rscript'
                        if (i < totalLotes) {
                            echo "Aguardando 3 minutos antes do próximo lote..."
                            sh 'sleep 180'
                        }
                    }
                }
            }
        }

        stage('Executar Projeto Big Data com Spark') {
            steps {
                script {
                    echo "Verificando se o Spark Master está rodando..."
                    try {
                        sh 'docker exec spark-master /bin/bash -c "exit 0"' // Teste simples
                        echo "Spark Master está acessível. Submetendo job..."
                        sh """
                        docker exec spark-master spark-submit \\
                            --master spark://spark-master:7077 \\
                            --jars ${SPARK_JAR_PATH} \\
                            ${PYTHON_SCRIPT_PATH}
                        """
                    } catch (Exception e) {
                        echo "ERRO: Não foi possível executar no container spark-master. Ele está rodando?"
                        sh 'docker ps -a' // Mostra o status dos containers para debug
                        currentBuild.result = 'FAILURE'
                        error "Falha ao submeter job Spark: ${e.message}"
                    }
                }
            }
        }
    }

    post {
        always {
            echo "Limpando Docker..."
            // Opcional: Mostrar logs do Spark antes de limpar, em caso de falha
            // sh 'docker logs spark-master || true'
            // sh 'docker logs spark-worker || true'
            sh 'docker-compose down -v --remove-orphans || true'
            sh 'docker system prune -f || true'
            sh '''
            IMAGENS=$(docker images -f "dangling=true" -q)
            if [ -n "$IMAGENS" ]; then
                echo "Removendo imagens dangling..."
                docker rmi $IMAGENS || true
            else
                echo "Nenhuma imagem dangling para remover."
            fi
            '''
        }
    }
}