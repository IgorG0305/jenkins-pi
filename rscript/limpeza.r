# Instalação dos pacotes necessários
if (!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.r-project.org")
if (!require(dplyr)) install.packages("dplyr", repos = "http://cran.r-project.org")
if (!require(tidyr)) install.packages("tidyr", repos = "http://cran.r-project.org")

# Carregamento das bibliotecas
library(ggplot2)
library(dplyr)
library(tidyr)

# Carregamento dos dados
dados <- read.csv("/app/alunos_com_erros.csv", sep = ",", na.strings = "", stringsAsFactors = TRUE)

# Renomeando colunas
colnames(dados) <- c(
  "ID", "Nome", "Email", "Curso", "Status", "Turma", "Sexo", "Idade", 
  "Trabalha", "Renda", "Acompanhamento Medico", "Tem Filho", "Estado Civil", 
  "Semestre", "Bimestre", 
  "Aula 1", "Professor 1", "Nota 1", "Falta Materia 1", "Desempenho 1", 
  "Aula 2", "Professor 2", "Nota 2", "Falta Materia 2", "Desempenho 2", 
  "Aula 3", "Professor 3", "Nota 3", "Falta Materia 3", "Desempenho 3", 
  "Aula 4", "Professor 4", "Nota 4", "Falta Materia 4", "Desempenho 4", 
  "Aula 5", "Professor 5", "Nota 5", "Falta Materia 5", "Desempenho 5", 
  "Risco de Evasao"
)

# Função para limpar nomes
limpar_nomes <- function(dados) {
  dados$Nome <- as.character(dados$Nome)
  dados$Email <- as.character(dados$Email)
  
  nomes_invalidos <- c("Desconhecido", "NULL", "#######", "0923023", "user", "notexist", "22333123", "?", "", "none.com")
  regex_nome_valido <- "^[A-Za-zÀ-ÿ ]{2,}$"
  nome_do_email <- sub("@.*", "", dados$Email)
  
  nome_invalido_logico <- (
    is.na(dados$Nome) |
    dados$Nome %in% nomes_invalidos |
    !grepl(regex_nome_valido, dados$Nome)
  )
  
  nome_email_valido <- grepl(regex_nome_valido, nome_do_email)
  substituir_nome <- nome_invalido_logico & nome_email_valido
  dados$Nome[substituir_nome] <- nome_do_email[substituir_nome]
  
  nome_invalido_final <- (
    is.na(dados$Nome) |
    dados$Nome %in% nomes_invalidos |
    !grepl(regex_nome_valido, dados$Nome)
  )
  dados <- dados[!nome_invalido_final, ]
  
  nome_suspeito <- grepl("^[a-z]+$", dados$Nome) & !grepl(" ", dados$Nome)
  dados <- dados[!nome_suspeito, ]
  
  dados$Nome <- gsub("\\b(Sr\\.|Sra\\.|Dr\\.|Dra\\.)\\s*", "", dados$Nome)
  dados$Nome <- tools::toTitleCase(tolower(dados$Nome))
  
  return(dados)
}

dados <- limpar_nomes(dados)

# Função para limpar emails
limpar_emails <- function(dados) {
  dados$Nome <- as.character(dados$Nome)
  dados$Email <- as.character(dados$Email)
  
  emails_invalidos <- c("@email", "NULL", "#######", "XX0294393LLL", "user", "notexist", "22333123", "?", "", "none.com")
  regex_email_valido <- "^[^@\\s]+@[^@\\s]+\\.[a-zA-Z]{2,}$"
  
  email_invalido_logico <- (
    is.na(dados$Email) |
    dados$Email %in% emails_invalidos |
    !grepl(regex_email_valido, dados$Email)
  )
  dados$Email[email_invalido_logico] <- NA
  
  nome_limpo <- tolower(gsub(" ", "", dados$Nome))
  dados$Email[is.na(dados$Email)] <- paste0(nome_limpo[is.na(dados$Email)], "@unifeob.com")
  
  dados$Email <- tolower(dados$Email)
  
  return(dados)
}

dados <- limpar_emails(dados)

# Tratamento da coluna Curso
dados$Curso <- as.character(dados$Curso)

insercoes <- c(
  'Administração', 'Direito', 'Engenharia', 'Pedagogia', 'Psicologia', 
  'Engenharia Civil', 'Engenharia Elétrica', 'Engenharia Mecânica', 
  'Engenharia de Produção', 'Arquitetura e Urbanismo', 'Medicina', 
  'Enfermagem', 'Biomedicina', 'Educação Física', 'Fisioterapia', 
  'Odontologia', 'Farmácia', 'Veterinária', 'Nutrição', 'Computação', 
  'Ciência da Computação', 'Sistemas de Informação', 'Análise e Desenvolvimento de Sistemas', 
  'Jogos Digitais', 'Redes de Computadores', 'Banco de Dados', 'Matemática', 
  'Física', 'Química', 'Biologia', 'Geografia', 'História', 'Letras', 
  'Serviço Social', 'Relações Internacionais', 'Jornalismo', 
  'Publicidade e Propaganda', 'Design Gráfico', 'Marketing', 
  'Recursos Humanos', 'Engenharia Ambiental', 'Engenharia de Alimentos', 
  'Engenharia Química', 'Zootecnia', 'Gastronomia', 'Moda', 'Teatro', 
  'Música', 'Dança', 'Cinema', 'Artes Visuais', 'Ciências Contábeis', 
  'Ciências Econômicas', 'Teologia', 'Fonoaudiologia', 'Terapia Ocupacional', 
  'Gestão Pública', 'Gestão Comercial', 'Logística', 'Secretariado Executivo', 
  'Turismo', 'Hotelaria', 'Ciências Sociais', 'Estatística', 'Biblioteconomia', 
  'Museologia', 'Educação Especial', 'Segurança do Trabalho', 'Radiologia'
)

i <- 1
while (sum(is.na(dados$Curso)) > 0) {
  posicao <- which(is.na(dados$Curso))[1]
  dados$Curso[posicao] <- insercoes[i]
  i <- i + 1
  if (i > length(insercoes)) i <- 1
}

# Correção da coluna Sexo
dados$Sexo <- tolower(dados$Sexo)
dados$Sexo[dados$Sexo %in% c("masculino", "masc", "m")] <- "masculino"
dados$Sexo[dados$Sexo %in% c("feminino", "fem", "feminno")] <- "feminino"
dados$Sexo[!dados$Sexo %in% c("masculino", "feminino")] <- NA

sexo_tab <- table(dados$Sexo)
if (length(sexo_tab) >= 2) {
  sexo_prop <- as.numeric(sexo_tab) / sum(sexo_tab)
  sexo_names <- names(sexo_tab)
  n_na <- sum(is.na(dados$Sexo))
  set.seed(123)
  dados$Sexo[is.na(dados$Sexo)] <- sample(sexo_names, size = n_na, replace = TRUE, prob = sexo_prop)
}

# Correção da coluna Idade
dados$Idade <- as.numeric(as.character(dados$Idade))
mediana_idade <- median(dados$Idade[dados$Idade >= 17 & dados$Idade <= 80], na.rm = TRUE)
dados$Idade[dados$Idade < 17 | dados$Idade > 80 | is.na(dados$Idade)] <- mediana_idade

# Tratamento da Renda
dados$Renda <- as.numeric(as.character(dados$Renda))
Q1_renda <- quantile(dados$Renda, 0.25, na.rm = TRUE)
Q3_renda <- quantile(dados$Renda, 0.75, na.rm = TRUE)
IQR_renda <- Q3_renda - Q1_renda
limite_inferior_renda <- Q1_renda - 1.5 * IQR_renda
limite_superior_renda <- Q3_renda + 1.5 * IQR_renda
mediana_renda <- median(dados$Renda[dados$Renda >= limite_inferior_renda & dados$Renda <= limite_superior_renda], na.rm = TRUE)
dados$Renda[dados$Renda < limite_inferior_renda | dados$Renda > limite_superior_renda | is.na(dados$Renda)] <- mediana_renda

# Correção da coluna Estado Civil
dados$`Estado Civil` <- tolower(trimws(dados$`Estado Civil`))
dados$`Estado Civil`[dados$`Estado Civil` %in% c("solteiro", "solteira", "s")] <- "solteiro"
dados$`Estado Civil`[dados$`Estado Civil` %in% c("casado", "casada", "c", "casadoo")] <- "casado"
dados$`Estado Civil`[dados$`Estado Civil` %in% c("divorciado", "divorciada", "d")] <- "divorciado"
dados$`Estado Civil`[dados$`Estado Civil` %in% c("viuvo", "viúva", "v")] <- "viuvo"
dados$`Estado Civil`[!dados$`Estado Civil` %in% c("solteiro", "casado", "divorciado", "viuvo")] <- NA

estado_tab <- table(dados$`Estado Civil`)
if (length(estado_tab) >= 2) {
  estado_prop <- as.numeric(estado_tab) / sum(estado_tab)
  estado_names <- names(estado_tab)
  n_na_estado <- sum(is.na(dados$`Estado Civil`))
  set.seed(456)
  dados$`Estado Civil`[is.na(dados$`Estado Civil`)] <- sample(estado_names, size = n_na_estado, replace = TRUE, prob = estado_prop)
}

write.csv(dados, "/app/alunos_corrigido.csv", row.names = FALSE)

