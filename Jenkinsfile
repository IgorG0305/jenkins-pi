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
                    echo "Subindo todos os serviços necessários..."
                    sh 'docker-compose up -d db flaskapi backend frontend grafana prometheus loki spark-master spark-worker'

                    echo "Aguardando banco de dados ficar pronto..."
                    sh '''
                        for i in {1..10}; do
                            if docker exec mysql_db mysqladmin ping -h"localhost" --silent; then
                                echo "MySQL está pronto!"
                                break
                            fi
                            echo "Aguardando MySQL..."
                            sleep 5
                        done
                    '''
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

        stage('Submit Spark Job') {
            steps {
                script {
                    echo "Submetendo job Spark..."
                    sh '''
                    docker exec spark-master spark-submit \
                        --master spark://spark-master:7077 \
                        --class org.apache.spark.examples.SparkPi \
                        /opt/spark/examples/jars/spark-examples_2.12-*.jar \
                        10
                    '''
                }
            }
        }
    }

    post {
        always {
            echo "Limpando Docker..."
            sh 'docker system prune -f || true'
            sh 'docker rmi $(docker images -f "dangling=true" -q) || true'
        }
    }
}
