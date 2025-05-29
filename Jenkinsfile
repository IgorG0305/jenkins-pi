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
    }

    stages {
        stage('Build Imagens') {
            steps {
                script {
                    echo "Construindo imagens com --no-cache..."
                    sh "docker build --no-cache -t ${IMAGE_GERADOR}:latest ./backend"
                    sh "docker build --no-cache -t ${IMAGE_RSCRIPT}:latest ./rscript"
                    sh "docker build --no-cache -t ${IMAGE_MYSQL}:latest ./mysql"
                    sh "docker build --no-cache -t ${IMAGE_FLASK}:latest ./api"
                    sh "docker build --no-cache -t ${IMAGE_BACKEND}:latest ./backend"
                    sh "docker build --no-cache -t ${IMAGE_SPARK}:latest ./spark"
                }
            }
        }

        stage('Subir Serviços') {
            steps {
                script {
                    echo "Removendo containers conflitantes..."
                    sh '''
                    CONTAINERS_TO_REMOVE="mysql_db spark-master spark-worker backend flaskapi frontend grafana prometheus loki"
                    for c in $CONTAINERS_TO_REMOVE; do
                        if docker ps -a --format '{{.Names}}' | grep -w "$c" > /dev/null; then
                            echo "Removendo container $c"
                            docker rm -f $c || true
                        fi
                    done
                    '''

                    echo "Removendo containers do docker compose..."
                    sh 'docker compose down -v || true'

                    echo "Subindo todos os serviços necessários..."
                    sh 'docker compose up -d db flaskapi backend frontend grafana prometheus loki spark-master spark-worker'

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
                    '''

                    echo "Aguardando spark-master estar ativo na porta 7077..."
                    sh '''
                    for i in {1..30}; do
                        nc -z spark-master 7077 && echo "Spark Master está pronto!" && break
                        echo "Aguardando spark-master ($i/30)..."
                        sleep 1
                    done
                    '''

                    echo "MySQL respondeu ao ping. Aguardando 30 segundos extras por segurança..."
                    sh 'sleep 30'
                }
            }
        }

        stage('Processo Iterativo') {
            steps {
                script {
                    int totalLotes = 1
                    for (int i = 1; i <= totalLotes; i++) {
                        echo "=== Iteração ${i} de ${totalLotes} ==="
                        echo "Executando gerador (1000 alunos)..."
                        sh 'docker compose run --rm gerador'
                
                        echo "Executando processamento R..."
                        sh 'docker compose run --rm rscript'
                
                        echo "Executando script Python para gerar gráficos do projeto bigdata..."
                        sh 'docker compose run --rm backend python3 python3 /opt/bitnami/spark/scripts/projeto_bigdata.py'
                
                        if (i < totalLotes) {
                            echo "Aguardando 3 minutos antes do próximo lote..."
                            sh 'sleep 180'
                        } 
                    }
                }
            }
        }


        stage('Submit Spark Job') {
            steps {
                script {
                    echo "Submetendo job Spark com correção para ivy..."
                    sh '''
                    docker exec spark-master mkdir -p /tmp/.ivy2

                    docker exec spark-master spark-submit \
                        --master spark://spark-master:7077 \
                        --class org.apache.spark.examples.SparkPi \
                        --conf spark.jars.ivy=/tmp/.ivy2 \
                        /opt/bitnami/spark/examples/jars/spark-examples_2.12-*.jar \
                        10
                    '''
                }
            }
        }

        stage('Executar Projeto Big Data') {
            steps {
                script {
                    echo "Executando projeto_bigdata.py no Spark..."
                    sh '''
                    docker exec spark-master python3 /opt/bitnami/spark/scripts/projeto_bigdata.py
                    '''
                }
            }
        }
    }

    post {
        always {
            echo "Parando e removendo todos os containers antes da limpeza..."
            sh '''
            docker ps -q | xargs -r docker stop || true
            docker ps -a -q | xargs -r docker rm -f || true
            '''

            echo "Limpando Docker..."
            sh 'docker system prune -f || true'

            sh '''
            IMAGENS=$(docker images -f "dangling=true" -q)
            if [ -n "$IMAGENS" ]; then
                docker rmi $IMAGENS || true
            else
                echo "Nenhuma imagem dangling para remover."
            fi
            '''
        }
    }
}
