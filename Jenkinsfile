pipeline {
    agent any

    environment {
        DOCKER_HUB_USER = 'superlike1'
        IMAGE_GERADOR  = "${DOCKER_HUB_USER}/gerador-app"
        IMAGE_RSCRIPT  = "${DOCKER_HUB_USER}/rscript-app"
    }

    stages {
        stage('Build Imagens') {
            steps {
                script {
                    echo "Construindo imagens com --no-cache..."
                    sh "docker build --no-cache -t ${IMAGE_GERADOR}:latest ./backend"
                    sh "docker build --no-cache -t ${IMAGE_RSCRIPT}:latest ./rscript"
                }
            }
        }

        stage('Subir DB') {
            steps {
                script {
                    echo "Subindo banco de dados MySQL..."
                    sh 'docker-compose up -d db'
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
                    int totalLotes = 10
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

        stage('Finalização') {
            steps {
                script {
                    echo "Finalizando e removendo containers..."
                    sh 'docker-compose down --remove-orphans'
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
