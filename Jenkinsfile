pipeline {
    agent any

    options {
        skipDefaultCheckout()
    }

    environment {
        DOCKER_HUB_USER = 'superlike1'
        IMAGE_GERADOR   = "${DOCKER_HUB_USER}/gerador-app"
        IMAGE_RSCRIPT   = "${DOCKER_HUB_USER}/rscript-app"
        IMAGE_MYSQL     = "${DOCKER_HUB_USER}/mysql-app"
        IMAGE_FLASK     = "${DOCKER_HUB_USER}/flask-app"
        IMAGE_BACKEND   = "${DOCKER_HUB_USER}/backend-app"
        GOOGLE_KEY_PATH = '/home/olivia-linux/Documentos/jenkins-pi/api/pi-do-mal.json'
        WORKSPACE_KEY_PATH = 'pi-do-mal.json'
    }

    stages {
        stage('Checkout') {
            steps {
                checkout([$class: 'GitSCM', 
                    branches: [[name: '*/main']],
                    userRemoteConfigs: [[
                        url: 'https://github.com/IgorG0305/jenkins-pi.git', 
                        credentialsId: 'token-github-jenkins' // <-- coloque o ID da sua credencial aqui
                    ]]
                ])
            }
        }

        stage('Preparar chave Google') {
            steps {
                writeFile file: "${env.WORKSPACE}/pi-do-mal.json", text: '''
{
  "type": "service_account",
  "project_id": "pi-data-science",
  "private_key_id": "78187599b4b50731b1964dd152f56b07b6f4e69d",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDEVNKlzV8eNoiF\nF2/JsVkx6TsqzxYMyf+Ljyitalol4etxNab40EMBcR2lAmM6Yo2V7R57CcnhCSVD\nOppn4Z+3u5XDrBFbz7pKiKpKZ8lxcgABu2CE8fvhDTRcXhNtlRc4C0M7PU1L6miG\ndJ6vCaw4HCl1LEV73ROL4kiaGXV3GZdttwuzhjhv9wSkqg8mlWTh2p+ba1N+ELKD\nZLVqnFgjdveG91rky6UOB/0Ah0LAmqucvRnEmkG5DVvqqjSm9UchdPD4QlZYOGS8\nHr0UOZpnuvhVmWnkfevofHdGaQ3qJ8b1q806fLp5aQYvcPGNdNJa2pl++LsBDzyg\nHr7MNlWxAgMBAAECggEAFzJLFcePNUjN1iUVPPglE4u8xQuPouDbk7fNNLlhP6Lo\nMvQwWgBykkE09pyIGyCuyCfVF0YU90TMhXDEnSvKt6YZr/NCCnN0zWlFRCJBaj5v\nuBn/8Y/iVHjqFqm6KOiEbvI3C3uW3jaACrbM/YhVM+dGkgSyNvbBYEsoNyYf5fty\nVcWoR7ZRBUCVUAUCfrSPW6jISAzsAOXdRt1DvSd/I/tAR847VZHRYDgfNTWALblg\nKAVoARLGyBWlCvICgl2/0pMPsNit6KUhhFaZwLQYJ6XzP67bRMm3AHEkFs6ZI0u2\n/1bnJxFKm9i1/8uNoWI7VUHs6TJOxh6gVXseZ6vzCwKBgQD30wodtJK2r9tiy8OF\n/XseISDlKFtYqdYnatSCfwU1DLiZ51ePrBX1HAT9n1jR2lJORQsSj9JX3EOT5m+/\nP/by/njMynKMH9hl8mXSEJ1HPLTov7V3wLTGzjDQYiCP9CzQoJ5fRmoyURIkd1J4\nFxrIi2N7EMmit4WQqSdzA06FUwKBgQDKzuhDHH5vKBf6DqXRTfE3r6UHQEo3N1RA\n+zKnXBcvZ6o9O4T0UWKRM9kbDVYXQzuoDcHML80olw2Y2ff31WvJ6xImxZWZUpuf\nJdl91GuoW/qIB4IKpmlx0nFeUts7Pa3Jhtc5wBtydY9tZeCGfZdZW0SYcINY61Dp\nj2Fy77l0awKBgQCbM6SLM/IJzRpUxg1+FWAMX2ztdpe2cC544xORYiENtxjI8bPJ\ne1kI9vI9L5T9X3/aGq0zg7SUZ5I+xrrHCDUMusdGYabZEHyCZArWQqds3JzjgmQQ\nSjQsqSay6jFAVgfW5DAqtLt/JXx6L+wK31VbsMetY7cTW2GfVgAprDF5lwKBgG60\nbvhjT2jh2+S1pjIQd8HL8St6HojxfN5TwJy2mjlYPwdZvAOZgVJ9mN85cmsUWjYr\n/EO3PCgR/GwZX1A7gbEgzjG33SdqfmRrRsN29qVaP/GNF6E0oY5uL1ArrlwyGPFO\n16FAijr6jSZMXDlNyRYPyevkTu501SAJEkqpLPStAoGBAOAcyGc1Xs+p0GqqQYVB\nX6rhnHG6eNbzit+P7STh2eg69ytvF3bpZfRRtXji/+FS+00YIiT2JN82xz14RqNz\nwwAyAjNkoaLNP1uNiMdmgwt/0aJXXaPvzDs2mRHrwAWCzRycI0PHvv0jEKYskpUY\nErRZFio+UD33J0N6dEda4M7Q\n-----END PRIVATE KEY-----\n",
  "client_email": "drive-access-service@pi-data-science.iam.gserviceaccount.com",
  "client_id": "112039530445888045376",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/drive-access-service%40pi-data-science.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
'''
                sh 'chmod 644 $WORKSPACE/pi-do-mal.json'
            }
        }

        stage('Build Imagens') {
            steps {
                script {
                    echo "Construindo imagens com --no-cache..."
                    sh "docker build --no-cache -t ${IMAGE_GERADOR}:latest ./backend"
                    sh "docker build --no-cache -t ${IMAGE_RSCRIPT}:latest ./rscript"
                    sh "docker build --no-cache -t ${IMAGE_MYSQL}:latest ./mysql"
                    sh "docker build --no-cache -t ${IMAGE_FLASK}:latest ./api"
                    sh "docker build --no-cache -t ${IMAGE_BACKEND}:latest ./backend"
                }
            }
        }

        stage('Subir Serviços') {
            steps {
                script {
                    echo "Removendo containers conflitantes..."
                    sh '''
                    CONTAINERS_TO_REMOVE="mysql_db backend flaskapi frontend grafana prometheus loki"
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
                    sh 'docker compose up -d db flaskapi backend frontend grafana prometheus loki'

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
                    echo "MySQL respondeu ao ping. Aguardando 30 segundos extras por segurança..."
                    sh 'sleep 30'
                }
            }
        }

        stage('Processo Infinito de Geração de Alunos') {
            steps {
                script {
                    while (true) {
                        echo "Executando gerador (1000 alunos)..."
                        sh 'docker compose run --rm gerador'
                        echo "Executando processamento R..."
                        sh 'docker compose run --rm rscript'
                        echo "Acionando API Flask para exportar e enviar dados para o Drive..."
                        sh '''
                        RESPONSE=$(curl -s -w "\\n%{http_code}" http://localhost:5051/exportar_alunos_tratados)
                        BODY=$(echo "$RESPONSE" | head -n1)
                        CODE=$(echo "$RESPONSE" | tail -n1)
                        if [ "$CODE" = "200" ]; then
                            echo "Exportação e upload para o Drive concluídos: $BODY"
                        else
                            echo "Falha ao exportar/upload para Drive: $BODY"
                        fi
                        '''
                        echo "Esperando 60 segundos antes do próximo lote..."
                        sh 'sleep 60'
                    }
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
