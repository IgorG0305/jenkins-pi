from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = "pi-data-science.json"
FOLDER_ID = '1pp8wXoa0r-BA2OKiYP0dOOQA3yyZ28hn'

def upload_csv_para_drive(caminho_csv, nome_arquivo):
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    query = f"'{FOLDER_ID}' in parents and name='{nome_arquivo}' and trashed = false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    arquivos = results.get('files', [])

    media = MediaFileUpload(caminho_csv, mimetype='text/csv')

    if arquivos:
        file_id_existente = arquivos[0]['id']
        updated_file = service.files().update(
            fileId=file_id_existente,
            media_body=media,
            fields='id'
        ).execute()
        return updated_file.get('id')
    else:
        file_metadata = {
            'name': nome_arquivo,
            'parents': [FOLDER_ID],
            'mimeType': 'application/vnd.google-apps.spreadsheet'
        }

        novo_arquivo = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        return novo_arquivo.get('id')
