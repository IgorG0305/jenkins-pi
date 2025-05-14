pipeline {
    agent any

    environment {
        DOCKER_HUB_USER = 'superlike1'
        IMAGE_FRONTEND = "${DOCKER_HUB_USER}/frontend-app"
        IMAGE_BACKEND  = "${DOCKER_HUB_USER}/backend-app"
        IMAGE_GERADOR  = "${DOCKER_HUB_USER}/gerador-app"
        IMAGE_MYSQL    = "${DOCKER_HUB_USER}/mysql-app"
        IMAGE_RSCRIPT  = "${DOCKER_HUB_USER}/rscript-app"
    }

    stages {
        stage('Build Imagens Base') {
            steps {
                script {
                    sh "docker build -t ${IMAGE_FRONTEND}:latest ./frontend"
                    sh "docker build -t ${IMAGE_BACKEND}:latest ./backend"
                    sh "docker build -t ${IMAGE_GERADOR}:latest ./backend"
                    sh "docker build -t ${IMAGE_MYSQL}:latest ./mysql"
                }
            }
        }

        stage('Executar Gerador para criar CSV') {
            steps {
                script {
                    // Caminho do diretório backend no workspace
                    def backendPath = "${env.WORKSPACE}/backend"

                    // Executa o container do gerador temporariamente e gera o CSV
                    sh "docker run --rm -v \"${backendPath}:/app\" ${IMAGE_GERADOR}:latest python gerador.py"

                    // Verifica se o CSV foi criado
                    sh "ls -l \"${backendPath}/alunos_com_erros.csv\""
                }
            }
        }

        stage('Build Imagem RScript (com CSV gerado)') {
            steps {
                script {
                    // Primeiro copia o CSV da pasta backend para a pasta rscript
                    sh "cp ${env.WORKSPACE}/backend/alunos_com_erros.csv ${env.WORKSPACE}/rscript/"
            
                    // Agora construa a imagem
                    sh "docker build -t ${IMAGE_RSCRIPT}:latest ./rscript"
            
                    // Opcional: remove o CSV após o build para manter o diretório limpo
                    sh "rm ${env.WORKSPACE}/rscript/alunos_com_erros.csv"
                }
            }
        }

        stage('Push Imagens para Docker Hub') {
            steps {
                script {
                    withCredentials([usernamePassword(
                        credentialsId: 'docker-hub-token', 
                        usernameVariable: 'DOCKER_USER', 
                        passwordVariable: 'DOCKER_PASS')]) {

                        // Login no Docker Hub
                        sh "echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin"

                        // Envia as imagens para o Docker Hub
                        sh "docker push ${IMAGE_FRONTEND}:latest"
                        sh "docker push ${IMAGE_BACKEND}:latest"
                        sh "docker push ${IMAGE_GERADOR}:latest"
                        sh "docker push ${IMAGE_MYSQL}:latest"
                        sh "docker push ${IMAGE_RSCRIPT}:latest"
                    }
                }
            }
        }

        stage('Deploy com Docker Compose') {
            steps {
                script {
                    // Derruba os containers existentes e sobe os novos
                    sh 'docker-compose down'
                    sh 'docker-compose up -d'
                }
            }
        }
    }
}