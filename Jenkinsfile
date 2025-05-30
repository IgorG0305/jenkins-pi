pipeline {
    agent any

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
        stage('Preparar chave Google') {
            steps {
                writeFile file: "${env.WORKSPACE}/pi-do-mal.json", text: '''
{
  "type": "service_account",
  "project_id": "pi-do-mal",
  "private_key_id": "6d762af1dc6af2f9239a30bddba02dbe20099ad4",
  "private_key": "-----BEGIN PRIVATE KEY-----\\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDOEINK8O9CChgH\\n3fws45oxCVGrbE38f53OQeEQpelC8kJHiiKjt2dIJcZc0AOU2mk18Xb0WZXjQyhb\\nK1RK0at22brO2C4HEVqmSSE4sCfg9IVx7TjFmS2xcry4XAzhdhAeIvLy4PBe6Y/W\\ngZ2SXw0bn6SmuAcPXB39eEg1s1SqrNiKl5Ua+lRY/0yxJ4UVrR+FNHQY8esz8yHW\\nZhQQg2x/MlkOw+TaAYg3zCZ4V9Rt+M7QKtJewOljsSWasDmIPa7Uf2aTuOpK3ZOn\\n76JgESLWVvCnY5OFzwXuvoBahMLA+g/VRvyEUP3+FE48+9uUywqpBODTHHn4az9o\\n/zkuIqLzAgMBAAECggEAGmmoGo6qwaE/lp/lN0UZOi3X8mFt1u6YpfEbIyImYGFj\\nDikY6uPoRil+tTN93LI+MsmM63chgkF/xmfVucgJnVtShDmAyivMSGzZPZs+u/6R\\nAvBa4DZpJ/SYAggyfSJqoHeZrSNt9rcjVmXDcm6NOebgQ5+/r+qpIW3Ye6Giju8L\\nr8Fki4VYJuEjFwD37mBYwV4LQwrHk4SFWWyTXQDgE8wazxG9gKG6zsfRtHGCus5O\\n1xAAXdaeI9pErfp1kI6+Gnr9VxV+EUqAdlCmtmGHA9OU7iDaeKGRwF0uI72TZf/O\\nNa00HZT3QscdCoKdnC0GbRoaxEndhGLx6F5Rq3+e6QKBgQD64RxN4BjzZakHSoWM\\nCVTGxSdA/8dGNglNJs1Bb+HDY78GfSIxCajEMADSi0AtWgYqy/sUOR/Tl/KDpCZi\\ngeDzta3nnSsXI3xZpTmwbFpmAn5Sv4hVhkVsNTy6goEm3v/l+RsRPq9+EMsSat4E\\neoYJ/8+71wbpO0Gd3qdpyoF8tQKBgQDSRTyczm73HUBpS9AdK78F5b684htwX3DH\\nL+tTND655hVPHrGaYuoGzk86pOvLO18CpnNFvC0WfhzIvXHEuqxfVUY4eOPz4WG4\\nwvlTEIvGsM1UAz/2yzslbyPQMpL4GlG40EtGInXS3gbqXMnGw6Pp7tOmGvoNah/h\\nVdZ+UjWSBwKBgB7Xt/wW9dpOgDZGQh7SMtrw9/90spH+KKyUfZ1y3MWBqMVqct6m\\neloMML2xouUwcRun0ilNUI1Z29W1Q4bOwtITXtrfpqGEmlAHEQ2QdJif69nOdDtX\\nc4d3EA056BjYR4uFUX+QPlD4TY7pFnxkd8AY8/f62n2n7Ew1SE2oOL0VAoGAGZ21\\ntKSxgAlgP3Os9uDNdLp4cipZjWcTJjEASjKjMaKGFg13NYe3WvznSg2tbCTffkMo\\n5+X02Dik6Q+rPHxBY5vP4jFYE+3xKcEW/reVT69aVFHRCQ/ZNMZFZqfCn9cU/Z7i\\njLjGAdpqnUKQklZjMayWvDWtINU87Qa4CsuZGyECgYBBGIaoYKm3KjMrfrbtKpdK\\nPYnHcjDg3OSwy3RFOot2pm3LkWIOmCkqXWjttLahXaAF/SSM432R09gy8NEN8rZo\\nNZnJfENQKFlWb3+TviSXXlsB+usmKn82HHYZd95B7xudnFOU/+bqDXw0EmtXq3d0\\noM6Se4uj4lPcrlex3D6Vhg==\\n-----END PRIVATE KEY-----\\n",
  "client_email": "marcelo-malefico@pi-do-mal.iam.gserviceaccount.com",
  "client_id": "104162948907990984092",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/marcelo-malefico%40pi-do-mal.iam.gserviceaccount.com",
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
