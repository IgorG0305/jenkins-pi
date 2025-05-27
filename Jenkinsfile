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
                        echo "Listando containers existentes..."
                        sh 'docker ps -a'

                        echo "Removendo containers antigos do MySQL manualmente, se existirem..."
                        sh '''
                        CONTAINERS=$(docker ps -a --format '{{.Names}}' | grep "mysql_db" || true)
                        if [ ! -z "$CONTAINERS" ]; then
                        echo "Removendo containers: $CONTAINERS"
                        for C in $CONTAINERS; do
                            docker rm -f "$C" || true
                        done
                        # Aguarda até todos sumirem
                        for i in {1..5}; do
                            EXISTS=$(docker ps -a --format '{{.Names}}' | grep "mysql_db" || true)
                            if [ -z "$EXISTS" ]; then
                                echo "Todos containers mysql_db removidos."
                                break
                            fi
                            echo "Aguardando remoção dos containers mysql_db..."
                            sleep 2
                            # NADA MAIS AQUI DENTRO DO LOOP
                        done
                    else
                        echo "Nenhum container mysql_db encontrado."
                    fi
                    '''

                    echo "Removendo containers do docker-compose..."
                    sh 'docker-compose down -v || true'

                    echo "Subindo todos os serviços necessários..."
                    sh 'docker-compose up -d db flaskapi backend frontend grafana prometheus loki spark-master spark-worker'

                    echo "Aguardando banco de dados ficar pronto..."
                    sh '''
                    for i in {1..15}; do # Pode deixar 10 ou 15, tanto faz
                        MYSQL_CONTAINER=$(docker ps --format '{{.Names}}' | grep "mysql_db" | head -n1)
                        if [ ! -z "$MYSQL_CONTAINER" ] && docker exec $MYSQL_CONTAINER mysqladmin ping -h"localhost" --silent; then
                            echo "MySQL está respondendo ao PING!"
                            break
                        fi
                        echo "Aguardando MySQL ($i/15)..."
                        sleep 5
                    done
                    '''
                
                // *** ESTE É O LUGAR CORRETO E ÚNICO PARA O SLEEP EXTRA ***
                    echo "MySQL respondeu ao ping. Aguardando 30 segundos extras por segurança..."
                    sh 'sleep 30'
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

            stage('Executar Projeto Big Data') {
                steps {
                    script {
                        echo "Executando projeto_bigdata.py no Spark..."
                        sh '''
                        docker exec spark-master python3 /opt/spark/scripts/projeto_bigdata.py
                        '''
                    }
                }
            }
        }

        post {
            always {
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