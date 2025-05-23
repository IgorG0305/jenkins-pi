library(DBI)
library(RMariaDB)
library(dplyr)
library(tidyr)
library(tools)

# Conexão ao banco de dados
con <- dbConnect(MariaDB(),
                 host = "db",
                 user = "dbpi",
                 password = "walker1207",
                 dbname = "faculdades1")

# Leitura dos dados não processados
query <- "SELECT * FROM alunos WHERE processado = 0"
dados <- dbGetQuery(con, query)

if (nrow(dados) > 0) {

  # Renomeando colunas para letras minúsculas
  colnames(dados) <- tolower(c(
    "aluno_id", "nome_aluno", "email_aluno", "curso", "status", "turma", "sexo", "idade",
    "trabalha", "renda_familiar", "acompanhamento_medico", "tem_filho", "estado_civil",
    "semestre", "bimestre",
    "aula_1", "professor_1", "notas_1", "faltas_1", "desempenho_1",
    "aula_2", "professor_2", "notas_2", "faltas_2", "desempenho_2",
    "aula_3", "professor_3", "notas_3", "faltas_3", "desempenho_3",
    "aula_4", "professor_4", "notas_4", "faltas_4", "desempenho_4",
    "aula_5", "professor_5", "notas_5", "faltas_5", "desempenho_5",
    "risco_evasao", "processado"
  ))

  limpar_nomes <- function(dados) {
    dados$nome_aluno <- as.character(dados$nome_aluno)
    dados$email_aluno <- as.character(dados$email_aluno)
    nomes_invalidos <- c("Desconhecido", "NULL", "#######", "0923023", "user", "notexist", "22333123", "?", "", "none.com")
    regex_nome_valido <- "^[A-Za-zÀ-ÿ ]{2,}$"
    nome_do_email <- sub("@.*", "", dados$email_aluno)
    nome_invalido_logico <- (
      is.na(dados$nome_aluno) |
      dados$nome_aluno %in% nomes_invalidos |
      !grepl(regex_nome_valido, dados$nome_aluno)
    )
    nome_email_valido <- grepl(regex_nome_valido, nome_do_email)
    substituir_nome <- nome_invalido_logico & nome_email_valido
    dados$nome_aluno[substituir_nome] <- nome_do_email[substituir_nome]
    nome_invalido_final <- (
      is.na(dados$nome_aluno) |
      dados$nome_aluno %in% nomes_invalidos |
      !grepl(regex_nome_valido, dados$nome_aluno)
    )
    dados <- dados[!nome_invalido_final, ]
    nome_suspeito <- grepl("^[a-z]+$", dados$nome_aluno) & !grepl(" ", dados$nome_aluno)
    dados <- dados[!nome_suspeito, ]
    dados$nome_aluno <- gsub("\\b(Sr\\.|Sra\\.|Dr\\.|Dra\\.)\\s*", "", dados$nome_aluno)
    dados$nome_aluno <- tools::toTitleCase(tolower(dados$nome_aluno))
    return(dados)
  }

  limpar_emails <- function(dados) {
    dados$email_aluno <- as.character(dados$email_aluno)
    regex_email_valido <- "^[^@\\s]+@[^@\\s]+\\.[a-zA-Z]{2,}$"
    emails_invalidos <- c("@email", "NULL", "#######", "XX0294393LLL", "user", "notexist", "22333123", "?", "", "none.com")
    email_invalido_logico <- (
      is.na(dados$email_aluno) |
      dados$email_aluno %in% emails_invalidos |
      !grepl(regex_email_valido, dados$email_aluno)
    )
    dados$email_aluno[email_invalido_logico] <- NA
    nome_limpo <- tolower(gsub(" ", "", dados$nome_aluno))
    dados$email_aluno[is.na(dados$email_aluno)] <- paste0(nome_limpo[is.na(dados$email_aluno)], "@unifeob.com")
    dados$email_aluno <- tolower(dados$email_aluno)
    return(dados)
  }

  limpar_cursos <- function(dados) {
    dados$curso <- as.character(dados$curso)
    insercoes <- c("Administração", "Direito", "Engenharia", "Pedagogia", "Psicologia", "Engenharia Civil", "Engenharia Elétrica",
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
    i <- 1
    while (sum(is.na(dados$curso)) > 0) {
      posicao <- which(is.na(dados$curso))[1]
      dados$curso[posicao] <- insercoes[i]
      i <- i + 1
      if (i > length(insercoes)) {
        i <- 1
      }
    }
    return(dados)
  }

  limpar_sexo <- function(dados) {
    dados$sexo <- tolower(dados$sexo)
    dados$sexo[dados$sexo %in% c("masculino", "masc", "m")] <- "masculino"
    dados$sexo[dados$sexo %in% c("feminino", "fem", "feminno")] <- "feminino"
    dados$sexo[!dados$sexo %in% c("masculino", "feminino")] <- NA
    sexo_tab <- table(dados$sexo)
    if (length(sexo_tab) >= 2) {
      sexo_prop <- as.numeric(sexo_tab) / sum(sexo_tab)
      sexo_names <- names(sexo_tab)
      n_na <- sum(is.na(dados$sexo))
      set.seed(123)
      dados$sexo[is.na(dados$sexo)] <- sample(sexo_names, size = n_na, replace = TRUE, prob = sexo_prop)
    } else {
      warning("Não há dados suficientes para aplicar o rateio proporcional em sexo.")
    }
    return(dados)
  }

  limpar_idade <- function(dados) {
    dados$idade <- as.numeric(as.character(dados$idade))
    mediana_idade <- median(dados$idade[dados$idade >= 17 & dados$idade <= 80], na.rm = TRUE)
    dados$idade[dados$idade < 17 | dados$idade > 80 | is.na(dados$idade)] <- mediana_idade
    return(dados)
  }

  limpar_renda <- function(dados) {
    dados$renda_familiar <- as.numeric(as.character(dados$renda_familiar))
    Q1 <- quantile(dados$renda_familiar, 0.25, na.rm = TRUE)
    Q3 <- quantile(dados$renda_familiar, 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    limite_inferior <- Q1 - 1.5 * IQR
    limite_superior <- Q3 + 1.5 * IQR
    mediana <- median(dados$renda_familiar[dados$renda_familiar >= limite_inferior & dados$renda_familiar <= limite_superior], na.rm = TRUE)
    dados$renda_familiar[dados$renda_familiar < limite_inferior | dados$renda_familiar > limite_superior | is.na(dados$renda_familiar)] <- mediana
    return(dados)
  }

  limpar_estado_civil <- function(dados) {
    dados$estado_civil <- tolower(trimws(dados$estado_civil))
    dados$estado_civil[dados$estado_civil %in% c("solteiro", "solteira", "s")] <- "solteiro"
    dados$estado_civil[dados$estado_civil %in% c("casado", "casada", "c", "casadoo")] <- "casado"
    dados$estado_civil[dados$estado_civil %in% c("divorciado", "divorciada", "d")] <- "divorciado"
    dados$estado_civil[dados$estado_civil %in% c("viuvo", "viúva", "v")] <- "viuvo"
    dados$estado_civil[!dados$estado_civil %in% c("solteiro", "casado", "divorciado", "viuvo")] <- NA
    estado_tab <- table(dados$estado_civil)
    if (length(estado_tab) >= 2) {
      estado_prop <- as.numeric(estado_tab) / sum(estado_tab)
      estado_names <- names(estado_tab)
      n_na <- sum(is.na(dados$estado_civil))
      set.seed(456)
      dados$estado_civil[is.na(dados$estado_civil)] <- sample(estado_names, size = n_na, replace = TRUE, prob = estado_prop)
    } else {
      warning("Não há dados suficientes para aplicar o rateio proporcional em estado_civil.")
    }
    return(dados)
  }

  # Aplicar funções
  dados <- limpar_nomes(dados)
  dados <- limpar_emails(dados)
  dados <- limpar_cursos(dados)
  dados <- limpar_sexo(dados)
  dados <- limpar_idade(dados)
  dados <- limpar_renda(dados)
  dados <- limpar_estado_civil(dados)

  # Tratamento de valores ausentes
  dados <- dados %>%
    mutate(
      across(where(is.numeric), ~replace_na(., 0)),
      across(where(is.character), ~replace_na(., "desconhecido"))
    )

  # Cálculo do risco de evasão
  calcula_risco <- function(linha) {
    notas <- unlist(linha[grep("^notas_", names(linha))])
    faltas <- unlist(linha[grep("^faltas_", names(linha))])
    if (mean(as.numeric(notas)) < 6 || any(as.numeric(faltas) > 5)) {
      return(1)
    } else {
      return(0)
    }
  }
  dados$risco_evasao <- apply(dados, 1, calcula_risco)

  # Grava os dados tratados
  dbWriteTable(con, "alunos_tratados", dados, append = TRUE, row.names = FALSE)

  # Atualiza os registros como processados
  ids <- paste(dados$aluno_id, collapse = ",")
  dbExecute(con, sprintf("UPDATE alunos SET processado = 1 WHERE id IN (%s)", ids))

  cat(paste(Sys.time(), "- Processados", nrow(dados), "registros.\n"))

} else {
  cat(paste(Sys.time(), "- Nenhum novo registro encontrado.\n"))
}

dbDisconnect(con)
