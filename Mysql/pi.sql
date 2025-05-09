CREATE DATABASE IF NOT EXISTS faculdades1;
USE faculdades1;

-- Tabela de usuários (alunos, professores e administradores)
CREATE TABLE usuarios (
id INT AUTO_INCREMENT PRIMARY KEY,
nome VARCHAR(255) NOT NULL,
email VARCHAR(255) UNIQUE NOT NULL,
senha_hash VARCHAR(255) NOT NULL,
tipo ENUM('aluno', 'professor', 'admin') NOT NULL
);

-- Tabela de cursos
CREATE TABLE cursos (
id INT AUTO_INCREMENT PRIMARY KEY,
nome VARCHAR(255) NOT NULL
);

-- Tabela de turmas
CREATE TABLE turmas (
id INT AUTO_INCREMENT PRIMARY KEY,
curso_id INT,
nome VARCHAR(255) NOT NULL,
FOREIGN KEY (curso_id) REFERENCES cursos(id) ON DELETE CASCADE
);

-- Tabela de professores
CREATE TABLE professores (
id INT AUTO_INCREMENT PRIMARY KEY,
nome VARCHAR(255) NOT NULL,
usuario_id INT UNIQUE,
telefone VARCHAR(20),
especializacao VARCHAR(255),
FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabela de alunos
CREATE TABLE alunos (
id INT AUTO_INCREMENT PRIMARY KEY,
nome VARCHAR(255) NOT NULL,
usuario_id INT UNIQUE,
data_nascimento DATE,
curso_id INT,
turma_id INT,
FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE,
FOREIGN KEY (curso_id) REFERENCES cursos(id) ON DELETE SET NULL,
FOREIGN KEY (turma_id) REFERENCES turmas(id) ON DELETE SET NULL
);

-- Tabela de situação do aluno
CREATE TABLE situacao_aluno (
id INT AUTO_INCREMENT PRIMARY KEY,
aluno_id INT UNIQUE,
salario_medio DECIMAL(10,2),
trabalha BOOLEAN,
cidade VARCHAR(255),
estado CHAR(2),
uso_alcool BOOLEAN,
fuma BOOLEAN,
uso_drogas BOOLEAN,
problemas_mentais TEXT,
FOREIGN KEY (aluno_id) REFERENCES alunos(id) ON DELETE CASCADE
);

-- Tabela de matérias
CREATE TABLE materias (
id INT AUTO_INCREMENT PRIMARY KEY,
nome VARCHAR(255) NOT NULL,
curso_id INT,
professor_id INT,
FOREIGN KEY (curso_id) REFERENCES cursos(id) ON DELETE CASCADE,
FOREIGN KEY (professor_id) REFERENCES professores(id) ON DELETE SET NULL
);

-- Tabela de avaliações
CREATE TABLE avaliacoes (
id INT AUTO_INCREMENT PRIMARY KEY,
materia_id INT,
titulo VARCHAR(255) NOT NULL,
descricao TEXT NOT NULL,
data DATE NOT NULL,
FOREIGN KEY (materia_id) REFERENCES materias(id) ON DELETE CASCADE
);

-- Tabela de notas
CREATE TABLE notas (
id INT AUTO_INCREMENT PRIMARY KEY,
aluno_id INT,
avaliacao_id INT,
nota DECIMAL(5,2) NOT NULL,
FOREIGN KEY (aluno_id) REFERENCES alunos(id) ON DELETE CASCADE,
FOREIGN KEY (avaliacao_id) REFERENCES avaliacoes(id) ON DELETE CASCADE
);

-- Tabela de alertas para alunos
CREATE TABLE alertas (
id INT AUTO_INCREMENT PRIMARY KEY,
aluno_id INT,
mensagem TEXT NOT NULL,
data TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
FOREIGN KEY (aluno_id) REFERENCES alunos(id) ON DELETE CASCADE
);

-- Tabela de histórico acadêmico
CREATE TABLE historico_academico (
id INT AUTO_INCREMENT PRIMARY KEY,
aluno_id INT,
curso_id INT,
turma_id INT,
data_inicio DATE,
data_fim DATE,
status ENUM('cursando', 'concluído', 'trancado', 'desistente') NOT NULL,
FOREIGN KEY (aluno_id) REFERENCES alunos(id) ON DELETE CASCADE,
FOREIGN KEY (curso_id) REFERENCES cursos(id) ON DELETE CASCADE,
FOREIGN KEY (turma_id) REFERENCES turmas(id) ON DELETE CASCADE
);

-- Tabela de frequência dos alunos
CREATE TABLE frequencia (
id INT AUTO_INCREMENT PRIMARY KEY,
aluno_id INT,
materia_id INT,
data DATE NOT NULL,
presente BOOLEAN NOT NULL,
FOREIGN KEY (aluno_id) REFERENCES alunos(id) ON DELETE CASCADE,
FOREIGN KEY (materia_id) REFERENCES materias(id) ON DELETE CASCADE
);