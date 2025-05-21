pipeline {
    agent any

    environment {
        DOCKER_HUB_USER = 'superlike1'
        IMAGE_FRONTEND = "${DOCKER_HUB_USER}/frontend-app"
        IMAGE_BACKEND  = "${DOCKER_HUB_USER}/backend-app"
        IMAGE_GERADOR  = "${DOCKER_HUB_USER}/gerador-app"
        IMAGE_MYSQL    = "${DOCKER_HUB_USER}/mysql-app"
        IMAGE_RSCRIPT  = "${DOCKER_HUB_USER}/rscript-app"
        IMAGE_FLASKAPI = "${DOCKER_HUB_USER}/flask-app"
    }

    stages {
        stage('Build Imagens Base') {
            steps {
                script {
                    echo "Construindo imagens base..."
                    sh "docker build -t ${IMAGE_FRONTEND}:latest ./frontend"
                    sh "docker build -t ${IMAGE_BACKEND}:latest ./backend"
                    sh "docker build -t ${IMAGE_GERADOR}:latest ./backend"
                    sh "docker build -t ${IMAGE_MYSQL}:latest ./mysql"
                    sh "docker build -t ${IMAGE_FLASKAPI}:latest ./api"
                }
            }
        }

        stage('Subir DB e Executar Gerador para criar CSV') {
            steps {
                script {
                    echo "Criando diretório backend no workspace e ajustando permissões..."
                    sh """
                    mkdir -p "${env.WORKSPACE}/backend"
                    chmod 777 "${env.WORKSPACE}/backend"
                    """

                    echo "Subindo banco de dados MySQL..."
                    sh 'docker-compose up -d db'

                    echo "Aguardando banco de dados ficar pronto..."
                    sh 'sleep 15' // Ajuste conforme seu tempo de startup do DB

                    echo "Executando container gerador para criar arquivo CSV..."
                    sh 'docker-compose run --rm gerador'

                    echo "Conteúdo do diretório backend após execução do gerador:"
                    sh "ls -lah \"${env.WORKSPACE}/backend\""

                    echo "Verificando se arquivo alunos_com_erros.csv foi criado..."
                    sh """
                    if [ -f "${env.WORKSPACE}/backend/alunos_com_erros.csv" ]; then
                        echo "Arquivo alunos_com_erros.csv encontrado."
                    else
                        echo "Arquivo alunos_com_erros.csv NÃO encontrado! Abortando pipeline."
                        exit 1
                    fi
                    """
                }
            }
        }

        stage('Build Imagem RScript') {
            steps {
                script {
                    echo "Preparando arquivo CSV para build da imagem RScript..."
                    sh """
                    mkdir -p "${env.WORKSPACE}/rscript"
                    cp "${env.WORKSPACE}/backend/alunos_com_erros.csv" "${env.WORKSPACE}/rscript/"
                    chmod 644 "${env.WORKSPACE}/rscript/alunos_com_erros.csv"
                    """

                    echo "Construindo imagem do RScript..."
                    sh "docker build -t ${IMAGE_RSCRIPT}:latest ./rscript"

                    echo "Removendo arquivo CSV temporário do rscript..."
                    sh "rm -f \"${env.WORKSPACE}/rscript/alunos_com_erros.csv\""
                }
            }
        }

        stage('Push Imagens para Docker Hub') {
            steps {
                script {
                    echo "Logando no Docker Hub e enviando imagens..."
                    withCredentials([usernamePassword(
                        credentialsId: 'docker-hub-token', 
                        usernameVariable: 'DOCKER_USER', 
                        passwordVariable: 'DOCKER_PASS'
                    )]) {
                        sh """
                            echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                            docker push ${IMAGE_FRONTEND}:latest
                            docker push ${IMAGE_BACKEND}:latest
                            docker push ${IMAGE_GERADOR}:latest
                            docker push ${IMAGE_MYSQL}:latest
                            docker push ${IMAGE_RSCRIPT}:latest
                            docker push ${IMAGE_FLASKAPI}:latest
                        """
                    }
                }
            }
        }

        stage('Deploy com Docker Compose') {
            steps {
                script {
                    echo "Finalizando e removendo containers antigos..."
                    sh 'docker-compose down --remove-orphans'

                    echo "Subindo containers atualizados em background..."
                    sh 'docker-compose up -d --force-recreate'
                }
            }
        }
    }

    post {
        always {
            echo "Limpeza pós-build..."
            sh """
                docker system prune -f || true
                rm -f "${env.WORKSPACE}/backend/alunos_com_erros.csv" || true
            """
        }
    }
}
