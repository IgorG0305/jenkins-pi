from flask import Flask, request, jsonify
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

app = Flask(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = 'pi-do-mal.json'
FOLDER_ID = '1fZs-W-0ynbHAgtw9AjT5or3WoBxY4QKG'

def upload_csv_para_drive(caminho_csv, nome_arquivo):
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    file_metadata = {
        'name': nome_arquivo,
        'parents': [FOLDER_ID],
        'mimeType': 'application/vnd.google-apps.spreadsheet'
    }

    media = MediaFileUpload(caminho_csv, mimetype='text/csv')
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    return file.get('id')

@app.route('/exportar_alunos_tratados', methods=['POST'])
def exportar_alunos_tratados():
    data = request.get_json()
    caminho_csv = data.get('caminho_csv', 'alunos_tratados.csv')  # ajuste conforme seu pipeline
    nome_arquivo = data.get('nome_arquivo', 'alunos_tratados_gsheet')

    try:
        file_id = upload_csv_para_drive(caminho_csv, nome_arquivo)
        return jsonify({'status': 'ok', 'file_id': file_id}), 200
    except Exception as e:
        return jsonify({'status': 'erro', 'mensagem': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)
