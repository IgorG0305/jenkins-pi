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
                    // Garante permissões no diretório backend
                    sh """
                        mkdir -p "${env.WORKSPACE}/backend"
                        chmod 777 "${env.WORKSPACE}/backend"
                    """
                    
                    // Executa o container com volume montado corretamente
                    sh """
                        docker run --rm \
                        -v "${env.WORKSPACE}/backend:/app:rw" \
                        -w /app \
                        ${IMAGE_GERADOR}:latest \
                        python gerador.py
                    """
                    
                    // Verificação robusta do arquivo gerado
                    sh """
                        echo "Conteúdo de ${env.WORKSPACE}/backend:"
                        ls -lah "${env.WORKSPACE}/backend"
                        echo "Tamanho do arquivo CSV:"
                        du -sh "${env.WORKSPACE}/backend/alunos_com_erros.csv"
                    """
                }
            }
        }

        stage('Build Imagem RScript') {
            steps {
                script {
                    // Garante que o diretório rscript existe
                    sh """
                        mkdir -p "${env.WORKSPACE}/rscript"
                        cp "${env.WORKSPACE}/backend/alunos_com_erros.csv" "${env.WORKSPACE}/rscript/"
                        chmod 644 "${env.WORKSPACE}/rscript/alunos_com_erros.csv"
                    """
                    
                    sh "docker build -t ${IMAGE_RSCRIPT}:latest ./rscript"
                    
                    // Limpeza opcional
                    sh "rm -f \"${env.WORKSPACE}/rscript/alunos_com_erros.csv\""
                }
            }
        }

        stage('Push Imagens para Docker Hub') {
            steps {
                script {
                    withCredentials([usernamePassword(
                        credentialsId: 'docker-hub-token', 
                        usernameVariable: 'DOCKER_USER', 
                        passwordVariable: 'DOCKER_PASS'
                    ]) {
                        sh """
                            echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin
                            docker push ${IMAGE_FRONTEND}:latest
                            docker push ${IMAGE_BACKEND}:latest
                            docker push ${IMAGE_GERADOR}:latest
                            docker push ${IMAGE_MYSQL}:latest
                            docker push ${IMAGE_RSCRIPT}:latest
                        """
                    }
                }
            }
        }

        stage('Deploy com Docker Compose') {
            steps {
                script {
                    sh 'docker-compose down --remove-orphans'
                    sh 'docker-compose up -d --force-recreate'
                }
            }
        }
    }

    post {
        always {
            // Limpeza final opcional
            sh """
                docker system prune -f || true
                rm -f "${env.WORKSPACE}/backend/alunos_com_erros.csv"
            """
        }
    }
}