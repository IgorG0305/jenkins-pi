pipeline {
    agent any

    environment {
        DOCKER_HUB_USER = 'superlike1'
        IMAGE_FRONTEND = "${DOCKER_HUB_USER}/frontend-app"
        IMAGE_BACKEND = "${DOCKER_HUB_USER}/backend-app"
        IMAGE_GERADOR = "${DOCKER_HUB_USER}/gerador-app"
        IMAGE_MYSQL = "${DOCKER_HUB_USER}/mysql-app"
        IMAGE_RSCRIPT = "${DOCKER_HUB_USER}/rscript-app"
    }

    stages {
        stage('Build e Push Docker Images') {
            steps {
                script {
                    // build
                    sh "docker build -t ${IMAGE_FRONTEND}:latest ./frontend"
                    sh "docker build -t ${IMAGE_BACKEND}:latest ./backend"
                    sh "docker build -t ${IMAGE_GERADOR}:latest ./backend"
                    sh "docker build -t ${IMAGE_MYSQL}:latest ./mysql"
                    sh "docker build -t ${IMAGE_RSCRIPT}:latest ."

                    // login no docker hub token
                    withCredentials([usernamePassword(credentialsId:'docker-hub-token', 
                    usernameVariable: 'DOCKER_USER', 
                    passwordVariable: 'DOCKER_PASS')]) {
                        sh "echo $DOCKER_PASS | docker login -u $DOCKER_USER --password-stdin"
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
                    sh 'docker-compose down'
                    sh 'docker-compose up -d'
                }
            }
        }
    }
}
