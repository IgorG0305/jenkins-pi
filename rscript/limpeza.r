library(DBI)
library(RMariaDB)
library(dplyr)
library(tidyr)
library(tools)

# Conexão ao banco de dados
con = dbConnect(MariaDB(),
                 host = "db",
                 user = "dbpi",
                 password = "walker1207",
                 dbname = "faculdades1")

# Leitura dos dados não processados
query = "SELECT * FROM alunos WHERE processado = 0"
dados = dbGetQuery(con, query)

if (nrow(dados) > 0) {

  # Renomeando colunas para letras minúsculas
  colnames(dados) = tolower(names(dados))

  limpar_nomes = function(dados) {
    dados$nome_aluno = as.character(dados$nome_aluno)
    dados$email_aluno = as.character(dados$email_aluno)
    nomes_invalidos = c('#######', '22333123', '', NA, 'INVALID', 'AAAAAAAAAAAAAA2', 'IJIJIJII', 0, 10, '101010101')
    regex_nome_valido = "^[A-Za-zÀ-ÿ ]{2,}$"
    nome_do_email = sub("@.*", "", dados$email_aluno)
    nome_invalido_logico = (
      is.na(dados$nome_aluno) |
      dados$nome_aluno %in% nomes_invalidos |
      !grepl(regex_nome_valido, dados$nome_aluno)
    )
    nome_email_valido = grepl(regex_nome_valido, nome_do_email)
    substituir_nome = nome_invalido_logico & nome_email_valido
    dados$nome_aluno[substituir_nome] = nome_do_email[substituir_nome]
    nome_invalido_final = (
      is.na(dados$nome_aluno) |
      dados$nome_aluno %in% nomes_invalidos |
      !grepl(regex_nome_valido, dados$nome_aluno)
    )
    dados = dados[!nome_invalido_final, ]
    nome_suspeito = grepl("^[a-z]+$", dados$nome_aluno) & !grepl(" ", dados$nome_aluno)
    dados = dados[!nome_suspeito, ]
    dados$nome_aluno = gsub("\\b(Sr\\.|Sra\\.|Dr\\.|Dra\\.)\\s*", "", dados$nome_aluno)
    dados$nome_aluno = tools::toTitleCase(tolower(dados$nome_aluno))
    return(dados)
  }

  limpar_emails = function(dados) {
    dados$email_aluno = as.character(dados$email_aluno)
    regex_email_valido = "^[^@\\s]+@[^@\\s]+\\.[a-zA-Z]{2,}$"
    emails_invalidos = c('@email', 'user@@site', 'none.com', '?', 'INVALID', 'amazonas', 'NOTEXIST', 'ilegivel')
    email_invalido_logico = (
      is.na(dados$email_aluno) |
      dados$email_aluno %in% emails_invalidos |
      !grepl(regex_email_valido, dados$email_aluno)
    )
    dados$email_aluno[email_invalido_logico] = NA
    nome_limpo = tolower(gsub(" ", "", dados$nome_aluno))
    dados$email_aluno[is.na(dados$email_aluno)] = paste0(nome_limpo[is.na(dados$email_aluno)], "@unifeob.com")
    dados$email_aluno = tolower(dados$email_aluno)
    return(dados)
  }

  limpar_cursos = function(dados) {
    colnames(dados) = tolower(colnames(dados))

    dados$curso = as.character(dados$curso)

    insercoes = c("Administração", "Direito", "Engenharia", "Pedagogia", "Psicologia", "Engenharia Civil", "Engenharia Elétrica",
                   "Engenharia Mecânica", "Engenharia de Produção", "Arquitetura e Urbanismo", "Medicina", "Enfermagem",
                   "Biomedicina", "Educação Física", "Fisioterapia", "Odontologia", "Farmácia", "Veterinária", "Nutrição",
                   "Computação", "Ciência da Computação", "Sistemas de Informação", "Análise e Desenvolvimento de Sistemas",
                   "Jogos Digitais", "Redes de Computadores", "Banco de Dados", "Matemática", "Física", "Química", "Biologia",
                   "Geografia", "História", "Letras", "Serviço Social", "Relações Internacionais", "Jornalismo",
                   "Publicidade e Propaganda", "Design Gráfico", "Marketing", "Recursos Humanos", "Engenharia Ambiental",
                   "Engenharia de Alimentos", "Engenharia Química", "Zootecnia", "Gastronomia", "Moda", "Teatro", "Música",
                   "Dança", "Cinema", "Artes Visuais", "Ciências Contábeis", "Ciências Econômicas", "Teologia",
                   "Fonoaudiologia", "Terapia Ocupacional", "Gestão Pública", "Gestão Comercial", "Logística",
                   "Secretariado Executivo", "Turismo", "Hotelaria", "Ciências Sociais", "Estatística", "Biblioteconomia",
                   "Museologia", "Educação Especial", "Segurança do Trabalho", "Radiologia")
    
    valores_invalidos = grepl("@|\\d|\\$|concluido|alguns|veteri|WEWEO", dados$curso, ignore.case = TRUE) | dados$curso == "" | is.na(dados$curso)
    dados$curso[valores_invalidos] = NA
    
    na_indices = which(is.na(dados$curso))
    num_na = length(na_indices)
    
    if (num_na > 0) {
      
      cursos_para_inserir = rep(insercoes, length.out = num_na)
      dados$curso[na_indices] = cursos_para_inserir
    }
    
    return(dados)
  }

  limpar_sexo = function(dados) {
    dados$sexo = tolower(dados$sexo)
    dados$sexo[dados$sexo %in% c("masculino", "masc", "Masculinas")] = "masculino"
    dados$sexo[dados$sexo %in% c("feminino", "fem", "Femeadoa")] = "feminino"
    dados$sexo[!dados$sexo %in% c("masculino", "feminino")] = NA
    sexo_tab = table(dados$sexo)
    if (length(sexo_tab) >= 2) {
      sexo_prop = as.numeric(sexo_tab) / sum(sexo_tab)
      sexo_names = names(sexo_tab)
      n_na = sum(is.na(dados$sexo))
      set.seed(123)
      dados$sexo[is.na(dados$sexo)] = sample(sexo_names, size = n_na, replace = TRUE, prob = sexo_prop)
    } else {
      warning("Não há dados suficientes para aplicar o rateio proporcional em sexo.")
    }
    return(dados)
  }

  limpar_idade = function(dados) {
    dados$idade = as.numeric(as.character(dados$idade))
    mediana_idade = median(dados$idade[dados$idade >= 17 & dados$idade <= 80], na.rm = TRUE)
    dados$idade[dados$idade < 17 | dados$idade > 80 | is.na(dados$idade)] = mediana_idade
    return(dados)
  }

  limpar_renda = function(dados) {
    dados$renda_familiar = as.numeric(as.character(dados$renda_familiar))
    Q1 = quantile(dados$renda_familiar, 0.25, na.rm = TRUE)
    Q3 = quantile(dados$renda_familiar, 0.75, na.rm = TRUE)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    mediana = median(dados$renda_familiar[dados$renda_familiar >= limite_inferior & dados$renda_familiar <= limite_superior], na.rm = TRUE)
    dados$renda_familiar[dados$renda_familiar < limite_inferior | dados$renda_familiar > limite_superior | is.na(dados$renda_familiar)] = mediana
    return(dados)
  }

  limpar_estado_civil = function(dados) {
    dados$estado_civil = tolower(trimws(as.character(dados$estado_civil)))
    
    valores_invalidos = c("", "xw0s2#$@43", "naoencontrado", "xxxoxo0", "error", NA)
    
    
    dados$estado_civil[dados$estado_civil %in% c("solteiro", "solteira", "s", "solte")] = "solteiro"
    dados$estado_civil[dados$estado_civil %in% c("casado", "casada", "c", "casadoo", "relacioname")] = "casado"
    dados$estado_civil[dados$estado_civil %in% c("divorciado", "divorciada", "d")] = "divorciado"
    dados$estado_civil[dados$estado_civil %in% c("viuvo", "viúva", "v")] = "viuvo"
    
    dados$estado_civil[dados$estado_civil %in% valores_invalidos | !dados$estado_civil %in% c("solteiro", "casado", "divorciado", "viuvo")] = NA
    
    tbl = table(dados$estado_civil)
    if (length(tbl) == 0) {
      warning("Não há dados suficientes para aplicar o rateio proporcional em estado civil.")
    } else {
      moda_estado_civil = names(tbl)[which.max(tbl)]
      dados$estado_civil[is.na(dados$estado_civil)] = moda_estado_civil
    }
  }

  # Substituir strings vazias por NA
  dados[dados == ""] = NA
  
  # Aplicar funções
  dados = limpar_nomes(dados)
  dados = limpar_emails(dados)
  dados = limpar_cursos(dados)
  dados = limpar_sexo(dados)
  dados = limpar_idade(dados)
  dados = limpar_renda(dados)
  dados = limpar_estado_civil(dados)

  # Tratamento de valores ausentes
  dados = dados %>%
    mutate(
      across(where(is.numeric), ~replace_na(., 0)),
      across(where(is.character), ~replace_na(., "desconhecido"))
    )

  # Cálculo do risco de evasão
  calcula_risco = function(linha) {
    notas = unlist(linha[grep("^notas_", names(linha))])
    faltas = unlist(linha[grep("^faltas_", names(linha))])
    if (mean(as.numeric(notas)) < 6 || any(as.numeric(faltas) > 5)) {
      return(1)
    } else {
      return(0)
    }
  }
  dados$risco_evasao = apply(dados, 1, calcula_risco)

  # Grava os dados tratados
  dbWriteTable(con, "alunos_tratados", dados, append = TRUE, row.names = FALSE)

  # Atualiza os registros como processados
  ids = paste(dados$aluno_id, collapse = ",")
  dbExecute(con, sprintf("UPDATE alunos SET processado = 1 WHERE aluno_id IN (%s)", ids))

  cat(paste(Sys.time(), "- Processados", nrow(dados), "registros.\n"))

} else {
  cat(paste(Sys.time(), "- Nenhum novo registro encontrado.\n"))
}

dbDisconnect(con)
