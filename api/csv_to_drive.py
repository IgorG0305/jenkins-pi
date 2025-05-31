from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = {
  "type": "service_account",
  "project_id": "codex-web-drive-367617?",
  "private_key_id": "a42873c2ec86ba5f2c1fc05fffafd3e953934cfa?",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCjDJcLkWueR8AB\nzv3CJaqSfNk2bhQcuwZW6DnTh4yuj5NUVOzcuGI80Vy2tu25CCjUASsbl3xLtaZe\nQ33bXmNrweVKS29Koqxsn19R3QeW8Zh+C/ZLTCvtz2mqOXyr1u8F0Exc0xyQATOS\nrYfO2baDTEH1i48hZzlchxLPJQzDgb+d4tVTnQVEII7kj4KfZUg6tNu2xUNr6m79\nPbxwTaXFhnVs5FdzAFPrqRdMlVNvWhNQY0tx08PI/V5jooGbPRTVJIoXqQM1nzPG\nYnvDdv+1skx5L3rx+z9oJLB5cc144HovefdWGwVH65w/KuWP+7dHxpELyVwxva+Q\nVkQopnlJAgMBAAECggEARfJ5vfG1vGJFgcEd5bMo+MzslkglAqpNLu1TGWb2OFDJ\nwIzqTEohgrvCTXQuiYlxknmp151mKkiURa4oiPp2Jl05E2VHKjqdQ8AfMUxkHCIq\n7DAPif/0fIHVb7xXupRrWBAjIlNC1phdWphtQZTukmcdiMxFK+xPlF0x+YGIdzo7\nwGIRNQoaCaOapxmPE3duX/Nf8P3jxuFKlsfYk9LktNvoCUasiLPWHo0nhYwk8MLU\nSB+p32djOJ/g11elKASr3pGI1guMXn8kWu8uk8uWQ9bY89BMjtrX9JU00AY4UF9G\n9zG7Ks69VycI570CQB1nNcUa2bNr0uh1wTh4fbMgfQKBgQDXn2EV21cBGXtoURoC\nsqa/mzFo3XkfBwAJc/Mu60Wuw9855KKWRao3FLaxTi0GQM0jcG+aVQOMZ18r8WFu\njxr9098O9jVUbZ4Efwhw3CVezXhRqUj+TwZtHwhXaEhaLAaClEYxXo+LLaf4o1yU\nNHaadlOejC25zW4nNFtAI3UupwKBgQDBlOvr26N5WNvMiz5yTYMx4YoYuo8WbbOm\nfCE/R4mj1uXFEY02mQFMAvEkJVPDpQ0T59HNJcxN7SPXvqwBHy7wTfjdLEisg3s8\nUSmbImAYuNTf5kLUv6mFYIKiksTLqc2+nrg+z0padkx9NC/DuJAN64iFhFlzej4+\nlKaerMqGjwKBgHbwpMZAusqReuB9NELet1qkSeoVmTWDUALm4fM0triDtYQi2YjZ\nHt7JX0mI3Q9A+aed4wIX+SAe+YGs4djxuargj3+aUqqi2PKT/Fz9IuQbpU4uemRF\nxT62SoykqpyLAoBPODUNe1MDuU9PFcdu9SmZMeEYZDTE3AlmORu1M3OrAoGAIgcL\ncWoqUipMeKgBUhZN+Xarz6z7efXOnnUzYuwwRZyAxyNOr0o18CaUPMZS8xEQO2+E\ncYszXn9QzZu2oHvaxGLwW/Bs1eZGw1OA/MQOIEpdIP5YwQzvv4I3dFxYO3SdteQw\nftnxN/WTfG6v9rd6mvfMB6w8DRBv4JBAuaOl9gECgYEAnfkDOWgrdoGVUzsSXict\n+1nvwak2HaNq9fFEy6GLd0gsvdcdJG8pjDDr1p2nWr7o45TTvmygW4tZB/t8xU8a\n6cNWBFhFhiraT6nyuUeFuVZ6gvxMUPusSTedBNm4AIjWU2gXUBUHxrnrF9FB15gT\nCQ9uxIhv792BwZMIH0Iux2g=\n-----END PRIVATE KEY-----\n*",
  "client_email": "pi-bigdata@codex-web-drive-367617.iam.gserviceaccount.com*",
  "client_id": "101509423021390035683",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/pi-bigdata%40codex-web-drive-367617.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
SERVICE_ACCOUNT_FILE['project_id'] = SERVICE_ACCOUNT_FILE['project_id'].split('?')[0]
SERVICE_ACCOUNT_FILE['private_key_id'] = SERVICE_ACCOUNT_FILE['private_key_id'].split('?')[0]
SERVICE_ACCOUNT_FILE['private_key'] = SERVICE_ACCOUNT_FILE['private_key'].split('*')[0]
SERVICE_ACCOUNT_FILE['client_email'] = SERVICE_ACCOUNT_FILE['client_email'].split('*')[0]

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
