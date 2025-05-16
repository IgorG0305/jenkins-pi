# flask_api.py
from flask import Flask, request, jsonify
import mysql.connector

app = Flask(__name__)

db_config = {
    'host': 'db',
    'user': 'dbpi',
    'password': 'walker1207',
    'database': 'app_db'
}

@app.route('/alunos', methods=['GET'])
def listar_alunos():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM alunos")
    alunos = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(alunos)

@app.route('/alunos', methods=['POST'])
def adicionar_aluno():
    dados = request.get_json()
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)

