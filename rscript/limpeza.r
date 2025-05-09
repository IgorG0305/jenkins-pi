alunose = read.csv("/app/alunos_com_erros.csv", sep = ',', stringsAsFactors = T)
alunose[alunose == ""] <- NA
idade_mediana <- median(alunose$idade, na.rm = TRUE)
alunose$idade[is.na(alunose$idade)] <- idade_mediana
renda_mediana <- median(alunose$renda_familiar, na.rm = TRUE)
alunose$renda_familiar[is.na(alunose$renda_familiar)] <- renda_mediana
alunose <- alunose[alunose$nome_aluno != "", ]
any(alunose$nome_aluno == "")
idx_invalidos <- which(alunose$idade < 10 | alunose$idade > 100)
media_idade_valida <- mean(alunose$idade[alunose$idade >= 10 & alunose$idade <= 100], na.rm = TRUE)
alunose$idade[idx_invalidos] <- media_idade_valida
write.csv(alunose, "/app/alunos_corrigido.csv", row.names = FALSE)
