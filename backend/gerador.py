import pandas as pd
import random
from faker import Faker

faker = Faker('pt_BR')

# Parâmetros
num_alunos = 10000

# Opções
cursos = ['Administração', 'Direito', 'Engenharia', 'Pedagogia', 'Psicologia', 'Engenharia Civil', 'Engenharia Elétrica', 'Engenharia Mecânica', 'Engenharia de Produção', 'Arquitetura e Urbanismo', 'Medicina', 'Enfermagem', 'Biomedicina', 'Educação Física', 'Fisioterapia', 'Odontologia', 'Farmácia', 'Veterinária', 'Nutrição', 'Computação', 'Ciência da Computação', 'Sistemas de Informação', 'Análise e Desenvolvimento de Sistemas', 'Jogos Digitais', 'Redes de Computadores', 'Banco de Dados', 'Matemática', 'Física', 'Química', 'Biologia', 'Geografia', 'História', 'Letras', 'Serviço Social', 'Relações Internacionais', 'Jornalismo', 'Publicidade e Propaganda', 'Design Gráfico', 'Marketing', 'Recursos Humanos', 'Engenharia Ambiental', 'Engenharia de Alimentos', 'Engenharia Química', 'Zootecnia', 'Gastronomia', 'Moda', 'Teatro', 'Música', 'Dança', 'Cinema', 'Artes Visuais', 'Ciências Contábeis', 'Ciências Econômicas', 'Teologia', 'Fonoaudiologia', 'Terapia Ocupacional', 'Gestão Pública', 'Gestão Comercial', 'Logística', 'Secretariado Executivo', 'Turismo', 'Hotelaria', 'Ciências Sociais', 'Estatística', 'Biblioteconomia', 'Museologia', 'Educação Especial', 'Segurança do Trabalho', 'Radiologia']
status_list = ['Matriculado', 'Trancado', 'Formado']
turmas = ['A', 'B', 'C', 'D']
sexos = ['Masculino', 'Feminino', 'Outro']
estado_civil_list = ['Solteiro(a)', 'Relacionado(a)']
# Disciplinas específicas por curso
disciplinas_por_curso = {
  'Administração': ['Administração Geral', 'Gestão de Pessoas', 'Gestão Financeira', 'Marketing', 'Economia'],
    'Direito': ['Direito Civil', 'Direito Penal', 'Processo Civil', 'Processo Penal', 'Constitucional'],
    'Engenharia': ['Cálculo', 'Física Aplicada', 'Mecânica', 'Circuitos Elétricos', 'Desenho Técnico'],
    'Pedagogia': ['Psicopedagogia', 'Teoria da Educação', 'Didática', 'Educação Infantil', 'Gestão Escolar'],
    'Psicologia': ['Psicologia Geral', 'Teorias da Personalidade', 'Psicopatologia', 'Psicologia do Desenvolvimento', 'Psicologia Social'],
    'Engenharia Civil': ['Construção Civil', 'Cálculo Estrutural', 'Mecânica dos Solos', 'Topografia', 'Materiais de Construção'],
    'Engenharia Elétrica': ['Circuitos Elétricos', 'Eletrônica', 'Sistemas Elétricos', 'Máquinas Elétricas', 'Transformadores'],
    'Engenharia Mecânica': ['Mecânica dos Fluidos', 'Termodinâmica', 'Resistência dos Materiais', 'Desenho Técnico', 'Sistemas Mecânicos'],
    'Engenharia de Produção': ['Gestão de Processos', 'Gestão da Qualidade', 'Planejamento de Produção', 'Logística', 'Gestão de Estoques'],
    'Arquitetura e Urbanismo': ['Desenho Arquitetônico', 'Planejamento Urbano', 'História da Arquitetura', 'Construção Civil', 'Estruturas'],    'Medicina': ['Anatomia', 'Fisiologia', 'Patologia', 'Farmacologia', 'Clínica Médica'],
    'Enfermagem': ['Enfermagem Geral', 'Cuidado Intensivo', 'Anatomia e Fisiologia', 'Saúde Pública', 'Psicologia da Saúde'],
    'Biomedicina': ['Genética', 'Imunologia', 'Microbiologia', 'Biologia Molecular', 'Farmacologia'],
    'Educação Física': ['Fisiologia do Exercício', 'Cinesiologia', 'Psicologia do Esporte', 'Nutrição Esportiva', 'Gestão do Esporte'],
    'Fisioterapia': ['Terapia Física', 'Fisiologia', 'Reabilitação', 'Técnicas de Imagem', 'Anatomia'],
    'Odontologia': ['Anatomia Dental', 'Periodontia', 'Implantes', 'Radiologia Odontológica', 'Clínica Odontológica'],
    'Farmácia': ['Química Farmacêutica', 'Farmacologia', 'Análises Clínicas', 'Microbiologia', 'Cosmetologia'],
    'Veterinária': ['Anatomia Veterinária', 'Clínica Veterinária', 'Microbiologia Veterinária', 'Farmacologia Veterinária', 'Cirurgia Veterinária'],
    'Nutrição': ['Nutrição Clínica', 'Nutrição Esportiva', 'Técnicas de Cozinha', 'Nutrição Infantil', 'Psicologia da Alimentação'],
    'Computação': ['Algoritmos', 'Estruturas de Dados', 'Banco de Dados', 'Sistemas Operacionais', 'Programação'],    'Ciência da Computação': ['Estruturas de Dados', 'Algoritmos', 'Inteligência Artificial', 'Redes de Computadores', 'Sistemas Distribuídos'],
    'Sistemas de Informação': ['Banco de Dados', 'Análise de Sistemas', 'Gestão de Projetos de TI', 'Sistemas de Informação Empresarial', 'Desenvolvimento de Software'],
    'Análise e Desenvolvimento de Sistemas': ['Programação', 'Banco de Dados', 'Sistemas Operacionais', 'Engenharia de Software', 'Segurança da Informação'],
    'Jogos Digitais': ['Design de Jogos', 'Programação de Jogos', 'Gráficos de Computação', 'Realidade Virtual', 'Inteligência Artificial para Jogos'],
    'Redes de Computadores': ['Redes de Dados', 'Protocolos de Comunicação', 'Segurança em Redes', 'Administração de Redes', 'Arquitetura de Redes'],
    'Banco de Dados': ['Modelagem de Dados', 'SQL Avançado', 'Administração de Banco de Dados', 'Sistemas de Informação', 'Segurança de Banco de Dados'],
    'Matemática': ['Cálculo', 'Geometria', 'Álgebra Linear', 'Probabilidade e Estatística', 'Teoria dos Números'],
    'Física': ['Mecânica', 'Eletromagnetismo', 'Termodinâmica', 'Óptica', 'Física Nuclear'],
    'Química': ['Química Orgânica', 'Química Inorgânica', 'Físico-Química', 'Bioquímica', 'Química Analítica'],
    'Biologia': ['Biologia Celular', 'Genética', 'Ecologia', 'Botânica', 'Zoologia'],
    'Geografia': ['Geografia Física', 'Geopolítica', 'Cartografia', 'Geografia Humana', 'Planejamento Urbano'],    'História': ['História Geral', 'História do Brasil', 'História Contemporânea', 'Teoria Histórica', 'Historiografia'],
    'Letras': ['Literatura Brasileira', 'Literatura Portuguesa', 'Gramática', 'Linguística', 'Produção Textual'],
    'Serviço Social': ['Política Social', 'Direitos Humanos', 'Ética no Serviço Social', 'Gestão Social', 'Cultura e Sociedade'],
    'Relações Internacionais': ['Teoria das Relações Internacionais', 'Política Internacional', 'Direitos Humanos', 'Economia Internacional', 'Geopolítica'],
    'Jornalismo': ['Redação Jornalística', 'Ética Jornalística', 'Edição de Vídeo', 'Produção de Conteúdo', 'Comunicação e Mídia'],
    'Publicidade e Propaganda': ['Comunicação Publicitária', 'Marketing Digital', 'Publicidade Criativa', 'Comportamento do Consumidor', 'Relações Públicas'],
    'Design Gráfico': ['Teoria do Design', 'Design Digital', 'Tipografia', 'Design Editorial', 'Design de Interfaces'],
    'Marketing': ['Comportamento do Consumidor', 'Gestão de Marketing', 'Pesquisa de Mercado', 'Publicidade', 'Marketing Digital'],
    'Recursos Humanos': ['Gestão de Pessoas', 'Desenvolvimento Organizacional', 'Recrutamento e Seleção', 'Treinamento e Desenvolvimento', 'Gestão de Carreira'],
    'Engenharia Ambiental': ['Gestão Ambiental', 'Tecnologias Ambientais', 'Ecologia', 'Política Ambiental', 'Sistemas de Gestão Ambiental'],    'Engenharia de Alimentos': ['Tecnologia de Alimentos', 'Química de Alimentos', 'Microbiologia de Alimentos', 'Higiene e Sanidade', 'Gestão de Processos'],
    'Engenharia Química': ['Processos Químicos', 'Termodinâmica', 'Cinética Química', 'Reatores Químicos', 'Química Industrial'],
    'Zootecnia': ['Nutrição Animal', 'Genética Animal', 'Fisiologia Animal', 'Medicina Veterinária', 'Produção Animal'],
    'Gastronomia': ['Culinária Internacional', 'Gestão de Restaurante', 'Nutrição e Dietética', 'Cultura Gastronômica', 'Técnicas Culinárias'],
    'Moda': ['Design de Moda', 'História da Moda', 'Gestão de Moda', 'Marketing da Moda', 'Produção de Moda'],
    'Teatro': ['Direção Teatral', 'Roteiro Teatral', 'Cenografia', 'Iluminação Teatral', 'História do Teatro'],
    'Música': ['Teoria Musical', 'História da Música', 'Prática Instrumental', 'Composição Musical', 'Arranjo Musical'],
    'Dança': ['História da Dança', 'Técnicas de Dança', 'Coreografia', 'Gestão de Eventos de Dança', 'Dança Contemporânea'],
    'Cinema': ['Roteiro de Cinema', 'Direção de Cinema', 'Produção de Cinema', 'Montagem de Cinema', 'História do Cinema'],
    'Artes Visuais': ['Pintura', 'Escultura', 'Arte Digital', 'Fotografia', 'História da Arte'],    'Ciências Contábeis': ['Contabilidade Geral', 'Contabilidade de Custos', 'Contabilidade Tributária', 'Auditoria', 'Análise Financeira'],
    'Ciências Econômicas': ['Microeconomia', 'Macroeconomia', 'Economia Internacional', 'Teoria Econômica', 'Política Econômica'],
    'Teologia': ['História da Igreja', 'Teologia Sistemática', 'Exegese Bíblica', 'Teologia Moral', 'Filosofia da Religião'],
    'Fonoaudiologia': ['Fonologia', 'Audiologia', 'Reabilitação Fonoaudiológica', 'Psicologia do Desenvolvimento', 'Anatomia e Fisiologia'],
    'Terapia Ocupacional': ['Técnicas de Reabilitação', 'Fisioterapia', 'Psicologia Clínica', 'Avaliação Terapêutica', 'Desenvolvimento Humano'],
    'Gestão Pública': ['Gestão de Políticas Públicas', 'Administração Pública', 'Finanças Públicas', 'Gestão de Projetos', 'Gestão de Pessoas'],
    'Gestão Comercial': ['Gestão de Vendas', 'Comportamento do Consumidor', 'Marketing e Vendas', 'Gestão de Estoques', 'Gestão de Relacionamento'],
    'Logística': ['Gestão de Transportes', 'Gestão de Estoques', 'Gestão de Cadeia de Suprimentos', 'Transporte e Distribuição', 'Logística Internacional'],
    'Secretariado Executivo': ['Gestão de Escritórios', 'Gestão de Pessoas', 'Comunicação Empresarial', 'Administração do Tempo', 'Atendimento ao Cliente'],
    'Turismo': ['Gestão de Turismo', 'Planejamento de Turismo', 'Marketing Turístico', 'História do Turismo', 'Turismo Cultural'],   'Hotelaria': ['Gestão de Hotelaria', 'Marketing de Hotelaria', 'Gestão de Hospedagem', 'Gestão de Alimentos e Bebidas', 'Hospitalidade'],
    'Ciências Sociais': ['Sociologia Geral', 'Antropologia', 'Teoria Social', 'Movimentos Sociais', 'Psicologia Social'],
    'Estatística': ['Estatística Descritiva', 'Probabilidade', 'Análise de Dados', 'Métodos Estatísticos', 'Estatística Aplicada'],
    'Biblioteconomia': ['Catalogação', 'Gestão de Bibliotecas', 'Preservação de Documentos', 'Leitura e Literatura', 'Gestão da Informação'],
    'Arquivologia': ['Gestão de Arquivos', 'Documentação Digital', 'Teoria Arquivística', 'Preservação de Documentos', 'Arquivos Públicos'],
}

disciplinas_gerais = ['Matemática', 'História', 'Física', 'Química', 'Português',
                      'Sociologia', 'Biologia', 'Filosofia', 'Economia', 'Geografia']
professores = [faker.name() for _ in range(10000)]

# Funções
def gerar_nota():
    return round(random.uniform(0, 10), 2)

def gerar_faltas():
    return random.randint(0, 15)

def calcular_desempenho(nota, faltas):
    if nota >= 8 and faltas <= 3:
        return 'Excelente'
    elif nota >= 7 and faltas <= 5:
        return 'Bom'
    elif nota >= 5 and faltas <= 10:
        return 'Regular'
    else:
        return 'Ruim'

def calcular_risco_evasao(aluno):
    risco = 0
    if aluno['renda_familiar'] and aluno['renda_familiar'] < 2000:
        risco += 1
    if aluno['trabalha'] == 1:
        risco += 1
    if aluno['tem_filho'] == 1:
        risco += 1
    if aluno['estado_civil'] in ['Casado(a)', 'Divorciado(a)']:
        risco += 1
    if aluno['acompanhamento_medico'] == 1:
        risco += 1
    if aluno['idade'] and aluno['idade'] > 30:
        risco += 1

    desempenhos = [aluno.get(f'desempenho_{i}') for i in range(1, 6)]
    desempenho_ruim = sum(1 for d in desempenhos if d in ['Ruim', 'Regular'])
    if desempenho_ruim >= 3:
        risco += 1

    return 1 if risco >= 4 else 0

def inserir_erros(aluno, prob_erro=0.10):
    if random.random() < prob_erro:
        aluno[1] = random.choice(['#######', '22333123', '', None, 'NULL','0923023','notexist'])  # nome

    if random.random() < prob_erro:
        aluno[2] = random.choice(['@email', 'user@@site', 'none.com', '?','kapewodkad', 'XX0294393LLL', None])  # email

    if random.random() < prob_erro:
        aluno[6] = random.choice(['????', 'XXXXXXXXX', 'Feminno', 'INVALID', '2923!!!k2', 'AVIAO', 'SEMANAL', None])  # sexo

    if random.random() < prob_erro:
        aluno[7] = random.choice([-5, 3, 130, 999, 00, 10039, 99999, 300, 500, None])  # idade

    if random.random() < prob_erro:
        aluno[9] = random.choice([-1000, 0, -231039, -932932, -230, '', 10000000000, 9230243042304, None])  # renda

    if random.random() < prob_erro:
        aluno[3] = random.choice(['1233E', '', 'Engenhariawewea', 'curso', 'insira aqui', None])  # curso

    if random.random() < prob_erro:
        aluno[12] = random.choice(['Casadoo', 'clique aqui', '','INSERT DATA', 'XW0S2#$@43', None])  # estado civil

    return aluno

# Início
grade_por_combinacao = {}
dados = []

for aluno_id in range(1, num_alunos + 1):
    nome = faker.name()
    email = faker.email()
    curso = random.choice(cursos)
    status = random.choice(status_list)
    turma = random.choice(turmas)
    sexo = random.choice(sexos)
    idade = random.randint(17, 45)
    trabalha = random.choice([0, 1])
    renda_familiar = random.randint(800, 10000)
    acompanhamento_medico = random.choice([0, 1])
    tem_filho = random.choice([0, 1])
    estado_civil = random.choice(estado_civil_list)
    semestre = random.randint(1, 2)
    bimestre = random.randint(1, 4)

    aluno = [
        aluno_id, nome, email, curso, status, turma, sexo, idade, trabalha,
        renda_familiar, acompanhamento_medico, tem_filho, estado_civil,
        semestre, bimestre
    ]

    chave = (turma, semestre, bimestre)

    if chave not in grade_por_combinacao:
        if curso in disciplinas_por_curso:
            disciplinas_base = disciplinas_por_curso[curso]
        else:
            disciplinas_base = disciplinas_gerais

        disciplinas_sorteadas = random.sample(disciplinas_base, 5)

        professores_sorteados = random.sample(professores, 5)
        grade = list(zip(disciplinas_sorteadas, professores_sorteados))
        grade_por_combinacao[chave] = grade
    else:
        grade = grade_por_combinacao[chave]

    for aula, professor in grade:
        nota = gerar_nota()
        faltas = gerar_faltas()
        desempenho = calcular_desempenho(nota, faltas)
        aluno.extend([aula, professor, nota, faltas, desempenho])

    # Inserir erros intencionais
    aluno = inserir_erros(aluno, prob_erro=0.05)

    # Criar dicionário temporário para risco
    aluno_dict = dict(zip([
        'aluno_id', 'nome_aluno', 'email_aluno', 'curso', 'status', 'turma',
        'sexo', 'idade', 'trabalha', 'renda_familiar', 'acompanhamento_medico',
        'tem_filho', 'estado_civil', 'semestre', 'bimestre'
    ] + [f'aula_{i}' for i in range(1, 6)] +
        [f'professor_{i}' for i in range(1, 6)] +
        [f'notas_{i}' for i in range(1, 6)] +
        [f'faltas_{i}' for i in range(1, 6)] +
        [f'desempenho_{i}' for i in range(1, 6)], aluno))

    risco = calcular_risco_evasao(aluno_dict)
    aluno.append(risco)

    dados.append(aluno)

# Colunas do DataFrame
colunas = [
    'aluno_id', 'nome_aluno', 'email_aluno', 'curso', 'status', 'turma', 'sexo', 'idade',
    'trabalha', 'renda_familiar', 'acompanhamento_medico', 'tem_filho', 'estado_civil',
    'semestre', 'bimestre'
]

for i in range(1, 6):
    colunas.extend([
        f'aula_{i}', f'professor_{i}', f'notas_{i}', f'faltas_{i}', f'desempenho_{i}'
    ])

colunas.append('risco_evasao')

# Criar DataFrame e salvar
df = pd.DataFrame(dados, columns=colunas)
df.to_csv('alunos_com_erros.csv', index=False, encoding='utf-8-sig')

print("Arquivo com dados sujos salvo como 'alunos_com_erros.csv'")

