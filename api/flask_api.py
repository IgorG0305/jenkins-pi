from flask import Flask, request, jsonify
import mysql.connector
from mysql.connector import Error
import time
import pandas as pd

# --- Adicione estes imports:
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

# ... código já existente do seu Flask ...

@app.route('/exportar_alunos_tratados', methods=['GET'])
def exportar_alunos_tratados():
    try:
        # Consulta os dados
        conn = mysql.connector.connect(**db_config)
        df = pd.read_sql("SELECT * FROM alunos_tratados", conn)
        conn.close()

        # Salva como CSV
        csv_path = "alunos_tratados.csv"
        df.to_csv(csv_path, index=False)

        # Sobe para Google Drive
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
