from flask import Flask, request, jsonify
import mysql.connector
from mysql.connector import Error
import time
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

app = Flask(__name__)

db_config = {
    'host': 'db',
    'user': 'dbpi',
    'password': 'walker1207',
    'database': 'faculdades1'
}

def tentar_conectar():
    print("Tentando conectar ao banco de dados...")
    for i in range(10):
        try:
            conn = mysql.connector.connect(**db_config)
            conn.close()
            print("Conectado ao banco de dados com sucesso!")
            return
        except Error as e:
            print(f"Tentativa {i+1}/10 falhou: {e}")
            time.sleep(5)
    print("Não foi possível conectar ao banco de dados após 10 tentativas. Encerrando...")
    exit(1)

# Chama a função para testar conexão antes de iniciar a API
tentar_conectar()

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({'status': 'ok'})

@app.route('/alunos', methods=['GET'])
def listar_alunos():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM alunos")
        alunos = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(alunos)
    except Error as e:
        return jsonify({'erro': str(e)}), 500

@app.route('/alunos_tratados', methods=['GET'])
def listar_alunos_tratados():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM alunos_tratados")
        alunos = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(alunos)
    except Error as e:
        return jsonify({'erro': str(e)}), 500

@app.route('/alunos', methods=['POST'])
def adicionar_aluno():
    dados = request.get_json()
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO alunos (nome, data_nascimento, curso_id, turma_id) VALUES (%s, %s, %s, %s)",
            (dados['nome'], dados['data_nascimento'], dados['curso_id'], dados['turma_id'])
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"mensagem": "Aluno inserido com sucesso!"}), 201
    except Error as e:
        return jsonify({'erro': str(e)}), 500

@app.route('/exportar_alunos_tratados', methods=['GET'])
def exportar_alunos_tratados():
    try:
        conn = mysql.connector.connect(**db_config)
        df = pd.read_sql("SELECT * FROM alunos_tratados", conn)
        conn.close()

        csv_path = "alunos_tratados.csv"
        df.to_csv(csv_path, index=False)

        SCOPES = ['https://www.googleapis.com/auth/drive']
        SERVICE_ACCOUNT_FILE = 'pi-do-mal.json'
        FOLDER_ID = '1fZs-W-0ynbHAgtw9AjT5or3WoBxY4QKG'

        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds)

        file_metadata = {
            'name': 'alunos_tratados_gsheet',
            'parents': [FOLDER_ID],
            'mimeType': 'application/vnd.google-apps.spreadsheet'
        }
        media = MediaFileUpload(csv_path, mimetype='text/csv')
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        return jsonify({'status': 'ok', 'file_id': file.get('id')}), 200

    except Exception as e:
        return jsonify({'erro': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
