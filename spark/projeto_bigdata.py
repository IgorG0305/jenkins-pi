# ===============================
# 📦 Importação das Dependências
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge # Usando Ridge como no original
from sklearn.metrics import (
    classification_report, ConfusionMatrixDisplay, silhouette_score,
    r2_score, mean_squared_error, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier as SparkRF # Renomeado para evitar conflito
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sqlalchemy import create_engine
import warnings

# Ignorar warnings para limpeza da saída
warnings.filterwarnings('ignore')

# Tentar baixar stopwords (já feito no Dockerfile, mas garante)
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

# ================================
# ⚙️ Configuração Inicial
# ================================
# Configuração da conexão (Usada para Pandas e talvez Spark)
usuario = 'dbpi'
senha = 'walker1207'
host = 'db' # <-- MUITO IMPORTANTE: Usar o nome do serviço Docker
porta = 3306
nome_banco_1 = 'faculdades1'
nome_banco_2 = 'faculdade' # <-- ATENÇÃO: É outra base?

# String de conexão 1
try:
    engine1 = create_engine(f'mysql+pymysql://{usuario}:{senha}@{host}:{porta}/{nome_banco_1}')
    print("Conexão com 'faculdades1' estabelecida (Pandas).")
except Exception as e:
    print(f"Erro ao conectar a 'faculdades1': {e}")
    exit() # Se não conectar, não adianta continuar

# String de conexão 2 (Ajustado para 'db', mas verifique se é 'faculdade' ou 'faculdades1')
try:
    # ATENÇÃO: Verifique se as credenciais (root sem senha) e o banco 'faculdade' existem no container 'db'
    engine2 = create_engine(f'mysql+pymysql://root:@{host}:{porta}/{nome_banco_2}') # <-- Ajustado para 'db', mas verifique as credenciais!
    print("Conexão com 'faculdade' estabelecida (Pandas).")
except Exception as e:
    print(f"Erro ao conectar a 'faculdade': {e}")
    # Decida se deve sair ou continuar se esta conexão falhar
    # exit()


# ================================
# 📥 Leitura e Análise Inicial (Pandas)
# ================================
print("\n--- Lendo dados de 'alunos_tratados' ---")
try:
    df = pd.read_sql("SELECT * FROM alunos_tratados", con=engine1)
    print("Dados lidos com sucesso.")
    print(df.head())
    print(df.info())
    print("\nValores Nulos:")
    print(df.isnull().sum())
except Exception as e:
    print(f"Erro ao ler 'alunos_tratados': {e}")
    exit()

# ===============================
# 🔧 Iniciar sessão Spark
# ===============================
print("\n--- Iniciando Sessão Spark ---")
spark = SparkSession.builder \
    .appName("PrevisaoRiscoEvasao") \
    .getOrCreate() # <-- JAR e HDFS removidos, serão gerenciados externamente/não usados

print("Sessão Spark iniciada.")

# ===============================
# 📥 Leitura dos dados via JDBC (Spark)
# ===============================
print("\n--- Lendo dados via JDBC (Spark) ---")
try:
    df_spark = spark.read.format("jdbc").options(
        url=f"jdbc:mysql://{host}:{porta}/{nome_banco_1}", # <-- Usando 'db'
        driver="com.mysql.cj.jdbc.Driver",
        dbtable="alunos_tratados",
        user=usuario,
        password=senha
    ).load()

    print("Dados lidos com sucesso pelo Spark.")
    df_spark.printSchema()
    df_spark.show(5)
except Exception as e:
    print(f"Erro ao ler dados com Spark JDBC: {e}")
    spark.stop()
    exit()

# ===============================
# 🧼 Limpeza e conversão (Spark)
# ===============================
print("\n--- Processando dados no Spark ---")
colunas_numericas = ["idade", "renda_familiar", "faltas_1", "notas_1", "faltas_2", "notas_2",
                     "faltas_3", "notas_3", "faltas_4", "notas_4", "faltas_5", "notas_5"]
colunas_categoricas = ["sexo", "trabalha", "acompanhamento_medico", "tem_filho",
                       "estado_civil", "desempenho_1", "desempenho_2",
                       "desempenho_3", "desempenho_4", "desempenho_5", "risco_evasao"]

# Garante que as colunas existem antes de processar
df_spark_cols = df_spark.columns

for coln in colunas_numericas:
    if coln in df_spark_cols:
        df_spark = df_spark.withColumn(coln, col(coln).cast("double"))
        df_spark = df_spark.na.fill({coln: -1.0}) # Preencher com float se a coluna é double
    else:
        print(f"Aviso: Coluna numérica '{coln}' não encontrada no DataFrame Spark.")


for colc in colunas_categoricas:
    if colc in df_spark_cols:
        df_spark = df_spark.na.fill({colc: "desconhecido"})
        indexer = StringIndexer(inputCol=colc, outputCol=colc + "_idx", handleInvalid="keep")
        df_spark = indexer.fit(df_spark).transform(df_spark)
    else:
         print(f"Aviso: Coluna categórica '{colc}' não encontrada no DataFrame Spark.")

# ===============================
# 🔄 Vetorização (Spark)
# ===============================
# Apenas colunas que realmente existem e foram indexadas/convertidas
feature_cols_spark = [c for c in colunas_numericas if c in df_spark_cols] + \
                     [c + "_idx" for c in colunas_categoricas[:-1] if c + "_idx" in df_spark.columns]

print(f"\nFeatures usadas no Spark: {feature_cols_spark}")

if "risco_evasao_idx" not in df_spark.columns:
    print("Erro: Coluna target 'risco_evasao_idx' não foi criada. Verifique os dados de entrada.")
    spark.stop()
    exit()

assembler = VectorAssembler(inputCols=feature_cols_spark, outputCol="features_vec")
df_spark = assembler.transform(df_spark)

# ===============================
# 🌳 Random Forest Classifier (Spark)
# ===============================
print("\n--- Treinando Modelo Spark (Random Forest) ---")
rf_spark = SparkRF(labelCol="risco_evasao_idx", featuresCol="features_vec", numTrees=100)
model_spark = rf_spark.fit(df_spark)
predictions = model_spark.transform(df_spark)

# ===============================
# 📊 Avaliação (Spark)
# ===============================
print("\n--- Avaliando Modelo Spark ---")
evaluator = MulticlassClassificationEvaluator(
    labelCol="risco_evasao_idx", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)
print(f"Acurácia do modelo Spark: {accuracy:.4f}")

# Obter Matriz de Confusão (Coletando dados - CUIDADO com datasets grandes)
print("\n--- Gerando Matriz de Confusão (Spark -> Pandas) ---")
preds_pd = predictions.select("risco_evasao_idx", "prediction").toPandas()
conf_mat = confusion_matrix(preds_pd["risco_evasao_idx"], preds_pd["prediction"])
print("Matriz de Confusão:")
print(conf_mat)

# ===============================
# 📈 Visualizações (Comentadas para Jenkins)
# ===============================
# A exibição interativa não funciona no Jenkins. Salve em arquivos se necessário.

print("\n--- Geração de gráficos (comentada/adaptada para salvar) ---")

# --- Exemplo: Salvando um gráfico Matplotlib ---
# plt.figure(figsize=(8, 6))
# sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
# plt.title("Matriz de Confusão - Spark")
# plt.xlabel("Predito")
# plt.ylabel("Real")
# plt.savefig("/opt/bitnami/spark/scripts/matriz_confusao_spark.png") # Salva em um arquivo
# plt.close() # Fecha a figura para liberar memória
# print("Matriz de confusão salva em /opt/bitnami/spark/scripts/matriz_confusao_spark.png")

# --- Exemplo: Salvando um gráfico Plotly (requer 'kaleido') ---
# Para salvar Plotly, instale: pip install kaleido
# df_plot = pd.DataFrame({'PCA1': ..., 'PCA2': ..., 'Cluster': ...})
# fig = px.scatter(df_plot, x='PCA1', y='PCA2', color='Cluster')
# fig.write_image("/opt/bitnami/spark/scripts/cluster_pca.png")
# print("Gráfico de cluster salvo em /opt/bitnami/spark/scripts/cluster_pca.png")

# ===============================
# 🛑 Parar Sessão Spark
# ===============================
print("\n--- Parando Sessão Spark ---")
spark.stop()
print("Sessão Spark finalizada.")
print("\n--- Script Python Concluído ---")