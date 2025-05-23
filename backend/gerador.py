from faker import Faker
import random
import pandas as pd
from sqlalchemy import create_engine

from models.aluno import Aluno
from services.desempenho import CalculadoraDesempenho
from services.evasao import CalculadoraRiscoEvasao
from services.erro_injector import ErroInjector
from services.grade_generator import GeradorGrade
from services.nota import GeradorNota
from utils.disciplinas import disciplinas_por_curso, disciplinas_gerais

# Configuração do banco
usuario = 'dbpi'
senha = 'walker1207'
host = 'db'
porta = '3306'
nome_banco = 'faculdades1'
tabela_destino = 'alunos'

# Criar engine de conexão
engine = create_engine(f'mysql+pymysql://{usuario}:{senha}@{host}:{porta}/{nome_banco}')

faker = Faker('pt_BR')

cursos = list(disciplinas_por_curso.keys())
status_list = ['Matriculado', 'Trancado', 'Cancelado', 'Concluso']
turmas = ['A', 'B', 'C', 'D']
sexos = ['Masculino', 'Feminino', 'Outro']
estado_civil_list = ['Solteiro(a)', 'Casado(a)', 'Amasiado(a)', 'Separado(a)', 'Divorciado(a)', 'Viúvo(a)']
professores = [faker.name() for _ in range(10000)]

gerador_grade = GeradorGrade(disciplinas_por_curso, professores, disciplinas_gerais)

def validar_aluno_dict(d):
    """ Garante que os valores críticos sejam válidos após injeção de erro. """
    if d['sexo'] not in sexos:
        d['sexo'] = random.choice(sexos)
    if d['estado_civil'] not in estado_civil_list:
        d['estado_civil'] = random.choice(estado_civil_list)
    if d['status'] not in status_list:
        d['status'] = random.choice(status_list)

    # Valida e corrige idade
    try:
        d['idade'] = int(d['idade'])
        if d['idade'] < 0 or d['idade'] > 120:
            d['idade'] = random.randint(17, 70)
    except (ValueError, TypeError):
        d['idade'] = random.randint(17, 70)

    # Valida e corrige renda_familiar
    try:
        d['renda_familiar'] = float(d['renda_familiar'])
        if d['renda_familiar'] < 0:
            d['renda_familiar'] = random.randint(800, 10000)
        elif d['renda_familiar'] > 99999:
            d['renda_familiar'] = 99999
    except (ValueError, TypeError):
        d['renda_familiar'] = random.randint(800, 10000)

    return d

def gerar_alunos(inicio_id, quantidade):
    alunos = []
    for aluno_id in range(inicio_id, inicio_id + quantidade):
        aluno = Aluno(
            aluno_id=aluno_id,
            nome=faker.name(),
            email=faker.email(),
            curso=random.choice(cursos),
            status=random.choice(status_list),
            turma=random.choice(turmas),
            sexo=random.choice(sexos),
            idade=random.randint(17, 70),
            trabalha=random.choice([0, 1]),
            renda_familiar=random.randint(800, 10000),
            acompanhamento_medico=random.choice([0, 1]),
            tem_filho=random.choice([0, 1]),
            estado_civil=random.choice(estado_civil_list),
            semestre=random.randint(1, 8),
            bimestre=random.randint(1, 4)
        )

        grade = gerador_grade.obter_grade(aluno.turma, aluno.semestre, aluno.bimestre, aluno.curso)

        for aula, professor in grade:
            nota = GeradorNota.gerar_nota()
            faltas = GeradorNota.gerar_faltas()
            desempenho = CalculadoraDesempenho.avaliar(nota, faltas)

            aluno.disciplinas.append(aula)
            aluno.professores.append(professor)
            aluno.notas.append(nota)
            aluno.faltas.append(faltas)
            aluno.desempenhos.append(desempenho)

        aluno.risco_evasao = CalculadoraRiscoEvasao.calcular(aluno)

        aluno_dict = aluno.to_dict()
        aluno_dict = ErroInjector.aplicar(aluno_dict)

        # Valida os dados críticos após injeção de erro
        aluno_dict = validar_aluno_dict(aluno_dict)

        # Atualiza o objeto com os dados validados
        for key, value in aluno_dict.items():
            if hasattr(aluno, key):
                setattr(aluno, key, value)

        alunos.append(aluno)
    return alunos

def main():
    # Ajuste para sempre começar do último ID + 1
    with engine.connect() as conn:
        result = conn.execute(f"SELECT MAX(aluno_id) FROM {tabela_destino}")
        max_id = result.scalar()
        if max_id is None:
            max_id = 0

    aluno_id_atual = max_id + 1
    lote = 1000

    print(f"Iniciando geração de {lote} alunos a partir do ID {aluno_id_atual}...")

    alunos = gerar_alunos(aluno_id_atual, lote)

    # Converte para dicionário e adiciona o campo "processado" = 0
    dados = []
    for aluno in alunos:
        d = aluno.to_dict()
        d['processado'] = 0
        dados.append(d)

    df = pd.DataFrame(dados)
    df.to_sql(tabela_destino, con=engine, index=False, if_exists='append')

    print(f"{len(alunos)} alunos inseridos. Último aluno_id: {aluno_id_atual + lote - 1}")

if __name__ == "__main__":
    main()
