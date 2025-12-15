import os
import pandas as pd
import numpy as np

# estatística / testes
import scipy.stats as stats
from scipy.stats import chi2_contingency, shapiro, ttest_ind, anderson, kstest, mannwhitneyu #explicitando os testes
from scipy.stats import chi2

# visualização (opcional)
import matplotlib.pyplot as plt
import seaborn as sns

# modelos / pré-processamento
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# métricas
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve
from sklearn.metrics import f1_score


# Acurácia
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, precision_score, recall_score


# caminho para o arquivo csv

Path_Credito = r"C:\Users\Jeislan\Documents\CURSOS\UDEMY\Projeto Prático Final\bases\TB_CREDITO.csv"
credito_wkd = pd.read_csv(Path_Credito, sep=',', encoding='utf-8')
print(credito_wkd.head())

# --------------------------------------------------------------------------------------------------------------------
# 1- Estatísticas descritivas
#---------------------------------------------------------------------------------------------------------------------

# separando as variáveis em numericas e categóricas

num_cols = credito_wkd.select_dtypes(include=['number']).columns
cat_cols = credito_wkd.select_dtypes(exclude=['number']).columns.tolist()

# variáveis numéricas - estatísicas descritivas (0 vs 1)
estat_num = credito_wkd.groupby('target')[num_cols].agg(['mean','median','std','min','max']).T

# variáveis numéricas - somente média e mediana para facilitar a visualização (0 vs 1)
estat_num2 = credito_wkd.groupby('target')[num_cols].agg(['mean','median']).T

# categóricas (proporções) — organizadas em tabelinha
tabelas = []
for col in cat_cols:
    tab = (pd.crosstab(credito_wkd[col], credito_wkd['target'], normalize='index')
             .reset_index() # transforma índice em coluna
             .rename(columns={col: "categoria"})) # deixa o nome claro
    tab.insert(0, "variavel", col) # coloca o nome da variável na 1ª coluna
    tabelas.append(tab)
estat_cat = pd.concat(tabelas, ignore_index=True)

print("\n=== Variáveis Numéricas ===")
print(estat_num)
print("\n=== Variáveis Categóricas (proporções) ===")
print(estat_cat)

# -------------------------------------------------------------------------------------------------------------------
# analisando as estatísticas descritivas percebe-se que o target parece estar invertido. O banco de dados não 
# apresenta uma qualidade muito boa, com algumas variáveis dando dicas de pouco poder de discrimanão. 
# Sendo assim, estamos fazendo essa # alteração no dataframe credito_wkd (1- mau pagador e 0- bom pagador)

credito_wkd["target"] = 1 -credito_wkd["target"].astype(int)
# -------------------------------------------------------------------------------------------------------------------

# pelas estatísticas apresentadas, não há expectativas de um modelo muito bom para classificar bons e maus pagadores

# variáveis numéricas - estatísicas descritivas (0 vs 1) após a alteração do target
estat_num_troca_target = credito_wkd.groupby('target')[num_cols].agg(['mean','median','std','min','max']).T

# variáveis numéricas - somente média e mediana para facilitar a visualização (0 vs 1) após a alteração do target
estat_num2_troca_target = credito_wkd.groupby('target')[num_cols].agg(['mean','median']).T


# categóricas (proporções) — organizadas em tabelas após a alteração do target
tabelas = []
for col in cat_cols:
    tab = (pd.crosstab(credito_wkd[col], credito_wkd['target'], normalize='index')
             .reset_index() # transforma índice em coluna
             .rename(columns={col: "categoria"})) # deixa o nome claro
    tab.insert(0, "variavel", col) # coloca o nome da variável na 1ª coluna
    tabelas.append(tab)
estat_cat_troca_target = pd.concat(tabelas, ignore_index=True)

print("\n=== Variáveis Numéricas (após trocar target) ===")
print(estat_num_troca_target)

print("\n=== Variáveis Numéricas (mean/median, após troca target) ===")
print(estat_num2_troca_target)

print("\n=== Variáveis Categóricas (proporções, após troca target) ===")
print(estat_cat_troca_target)

# -------------------------------------------------------------------------------------------------------------------
# nesse bloco faremos uma contagem de casos por categorias das variáveis não numéricas

tabelas = []

for col in cat_cols:  # aproveitando a lista já criada
    tab = (
        credito_wkd[col]
        .value_counts(dropna=False)
        .to_frame(name="count")
    )
    tab["pct"] = (tab["count"] / len(credito_wkd)).round(4)
    tab = tab.reset_index().rename(columns={"index": "categoria"})
    tab.insert(0, "variavel", col)
    tabelas.append(tab)

freq_df = pd.concat(tabelas, ignore_index=True)

freq_df


#-----------Transformações nas variáveis ordinais e nominas

# - residencia_desde: nulo vai para "desconhecido" e cd_residencia_desde: nulo vai para 0

# - emprestimo_existente: juntar Alto (5 ou mais empréstimos) com Médio (3-4 empréstimos) 
# e cd_emprestimo_existente: o 4 vai para 3

# - histórco_credito: nulo vai para "desconhecido"	

# - proposito: não alterar, qualificacao é uma resposta possível e correta

# - faixa_tempo_emprego: nulo vai para "desconhecido" e cd_emprego:  nulo vai para 0

# - possui fiador: agrupar co requerente e co aplicador em co requerente/aplicador

# - habitação: nulo vai para "desconhecido"

# - qualificacao_profissional: nulo vai para "desconhecido"

# ---------------------------------------
# residencia_desde
# ---------------------------------------
credito_wkd["residencia_desde"] = credito_wkd["residencia_desde"].fillna("desconhecido")
credito_wkd["cd_residencia_desde"] = credito_wkd["cd_residencia_desde"].fillna(0).astype(int)

# ---------------------------------------
# emprestimo_existente (agrupamento)
# ---------------------------------------
# Texto
credito_wkd["emprestimo_existente"] = credito_wkd["emprestimo_existente"].replace(
    {"Alto (5 ou mais empréstimos)": "Médio (3-4 empréstimos)"}
)

# Código (4 -> 3)
credito_wkd["cd_emprestimo_existente"] = credito_wkd["cd_emprestimo_existente"].replace({4: 3})

# ---------------------------------------
# historico_credito
# ---------------------------------------
credito_wkd["historico_credito"] = credito_wkd["historico_credito"].fillna("desconhecido")

# ---------------------------------------
# proposito
# (nenhuma alteração)

# ---------------------------------------
# faixa_tempo_emprego
# ---------------------------------------
credito_wkd["faixa_tempo_emprego"] = credito_wkd["faixa_tempo_emprego"].fillna("desconhecido")
credito_wkd["cd_emprego"] = credito_wkd["cd_emprego"].fillna(0).astype(int)

# ---------------------------------------
# possui_fiador (agrupamento)
# ---------------------------------------
credito_wkd["possui_fiador"] = credito_wkd["possui_fiador"].replace(
    {
        "co requerente": "co requerente/aplicador",
        "co aplicante": "co requerente/aplicador"
    }
)

# ---------------------------------------
# habitacao
# ---------------------------------------
credito_wkd["habitacao"] = credito_wkd["habitacao"].fillna("desconhecido")

# ---------------------------------------
# qualificacao_profissional
# ---------------------------------------
credito_wkd["qualificacao_profissional"] = credito_wkd["qualificacao_profissional"].fillna("desconhecido")


# -------------------------------------------------------------------------------------------------------------------
# Verificação de missuing em variaveis contínuas

vars_para_verificar = [
    "duracao",
    "valor",
    "cd_tempo_parcelamento",
    "cd_residencia_desde",
    "idade",
    "cd_emprestimo_existente",
    "dependentes",
    "socio_empresa",
    "cd_estrangeiro",
    "cd_investimentos",
    "cd_emprego",
    "cd_profissao"
]

credito_wkd[vars_para_verificar].isna().sum()


# -------------------------------------------------------------------------------------------------------------------
# Gerando box plots para variáveis contínuas

sns.set(style="whitegrid") # garantir estilo

vars_box = ["duracao", "valor", "idade"]

plt.figure(figsize=(15, 5))

for i, col in enumerate(vars_box, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(data=credito_wkd, x="target", y=col)
    plt.title(f"Boxplot de {col} por target")
    plt.xlabel("Target (0 = bom pagador | 1 = mau pagador)")
    plt.ylabel(col)

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------------------------------------------------
# Existem outliers, mas podem ser dados naturais do mercado financeiro. Não é conveniente tratar, isso poderia afetar
# variáveis aparente pouco explicativas, mal distribuídas e com categorias pouco informativas são problemas maiores
# Muitos modelos não são afetados por dados, extremos. A Regressão Logística pode ser o modelo mais afetado, mas é preciso
# testar na prática.


# -------------------------------------------------------------------------------------------------------------------
# Analisando a proporção de bons e maus pagadores pela variável target

credito_wkd['target'].value_counts(normalize=True)

prop_target=pd.DataFrame({
    "contagem": credito_wkd['target'].value_counts(),
    "proporcao": credito_wkd['target'].value_counts(normalize=True).round(4)
})


# Gráfico de barras do target
prop = credito_wkd['target'].value_counts(normalize=True)

plt.figure(figsize=(5,4))
prop.plot(kind='bar')
plt.title("Distribuição do Target (%)")
plt.ylabel("Proporção")
plt.ylim(0,1)

# Mostrar porcentagens nas barras
for i, v in enumerate(prop):
    plt.text(i, v + 0.02, f"{v*100:.1f}%", ha="center")
    
    # arrumar os rótulos
plt.xticks(rotation=0)

plt.show()

# A base apresenta um desbalanceamento moderado, com aproximadamente 70% de bons pagadores e 30% de maus pagadores. 
# Esse cenário, combinado com variáveis de baixa capacidade explicativa, limita o desempenho máximo alcançável pelos
# modelos, o que gera uma expectativa de acurácias em torno de 70%, próximas do classificador da base.”


# --------------------------------------------------------------------------------------------------------------------
# 2- Testes de hipótese por tipo de variável
# --------------------------------------------------------------------------------------------------------------------

# confirmando o tipo de variável 

# --- Listas conforme a construção do pipeline em sql

continuas = ["duracao", "valor", "idade"]

ordinais = [
    "cd_tempo_parcelamento",
    "cd_residencia_desde",
    "cd_emprestimo_existente",
    "cd_investimentos",
    "cd_emprego",
    "dependentes"
]

nominais = [
    "socio_empresa",
    "cd_estrangeiro",
    "historico_credito",
    "proposito",
    "estado_civil",
    "possui_fiador",
    "habitacao",
    "outros_financiamentos",
    "qualificacao_profissional"
]

alpha = 0.05
target = "target"

bons = credito_wkd[credito_wkd[target] == 0]
maus = credito_wkd[credito_wkd[target] == 1]

results = []

def interpret(p):
    return "Significativo" if p <= alpha else "Não significativo"

# --------------------------------------------------------------------------------------------------------------------
# Variáveis CONTÍNUAS: Teste t e Mann-Whitney

for col in continuas:
    x1 = bons[col].dropna()
    x2 = maus[col].dropna()

    # teste t
    t_stat, t_p = stats.ttest_ind(x1, x2, equal_var=False, nan_policy='omit')
    results.append([col, "contínua", "t-test", t_stat, t_p, interpret(t_p)])

    # Mann-Whitney
    mw_stat, mw_p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    results.append([col, "contínua", "Mann-Whitney", mw_stat, mw_p, interpret(mw_p)])


# --------------------------------------------------------------------------------------------------------------------
# Variáveis ORDINAIS: Mann-Whitney e Spearman

for col in ordinais:
    x1 = bons[col].dropna()
    x2 = maus[col].dropna()

    # Mann-Whitney
    mw_stat, mw_p = stats.mannwhitneyu(x1, x2, alternative="two-sided")
    results.append([col, "ordinal", "Mann-Whitney", mw_stat, mw_p, interpret(mw_p)])

    # Spearman
    spear_r, spear_p = stats.spearmanr(credito_wkd[col], credito_wkd[target])
    results.append([col, "ordinal", "Spearman", spear_r, spear_p, interpret(spear_p)])


# --------------------------------------------------------------------------------------------------------------------
# Variáveis NOMINAIS -> Qui-Quadrado

for col in nominais:
    tabela = pd.crosstab(credito_wkd[col], credito_wkd[target])
    chi2_stat, p, dof, exp = stats.chi2_contingency(tabela)
    results.append([col, "nominal", "Qui-Quadrado", chi2_stat, p, interpret(p)])


# --------------------------------------------------------------------------------------------------------------------
# Tabela final consolidada


resultados_testes = pd.DataFrame(
    results,
    columns=["variavel", "tipo", "teste", "estatística", "p_valor", "conclusao"]
)

print("\n================ RESULTADOS DOS TESTES DE HIPÓTESE ================\n")
print(resultados_testes)

# Aparece automaticamente no variable explorer do Spyder
resultados_testes

# Várias variáveis apresentaram diferenças estatisticamente significativas entre bons e maus pagadores,
# demonstrando capacidade discriminativa univariada. A próxima etapa consiste em avaliar o poder preditivo
# dessas variáveis dentro de modelos estatísticos e de machine learning, verificando seu impacto conjunto
# na performance final e na capacidade de generalização dos modelos.

# --------------------------------------------------------------------------------------------------------------------
# 3- PREPARAÇÃO PARA MODELOS ESTATÍSTICOS E DE MACHINE LEARNING
# --------------------------------------------------------------------------------------------------------------------

# Pré-processamento para modelos estatísticos
#    a) OneHot para categóricas
#    b) manter numéricas e ordinais originais (sem scaler) para os modelos do statsmodels
#    c) aplicar o scaler nos modelos do sklearn 

# --------------------------------------------------------------------------------------------------------------------
# PRÉ-PROCESSAMENTO PARA MODELO ESTATÍSTICO (statsmodels)


# OneHot para nominais (mantém ordinais e contínuas originais, sem scaler)
X_stats = pd.get_dummies(
    credito_wkd[continuas + ordinais + nominais],
    drop_first=True
)

y_stats = credito_wkd[target].astype(int)

# Adicionando intercepto para statsmodels

X_stats_const = sm.add_constant(X_stats)

print("\nDimensões para o modelo estatístico:", X_stats_const.shape)


# --------------------------------------------------------------------------------------------------------------------
# PRÉ-PROCESSAMENTO PARA MODELOS DE MACHINE LEARNING (sklearn)


# Para ML definimos:
#    a) scaler para contínuas(padronização)
#    b) Método OneHot Encoder para nominais
#    c) ordinais já estão na forma correta (passthrough)


preprocess_ml = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), continuas),        # padroniza contínuas
        ("ord", "passthrough", ordinais),            # mantém ordinais como estão
        ("cat", OneHotEncoder(handle_unknown="ignore"), nominais)  # one-hot nominais
    ]
)

# X utiliza exatamente os mesmos nomes das listas definidas para os testes de hipóteses
X = credito_wkd[continuas + ordinais + nominais]
y = credito_wkd[target].astype(int)


# --------------------------------------------------------------------------------------------------------------------
# Split treino/teste

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nDimensões para modelos ML:")
print("Treino:", X_train.shape, " | Teste:", X_test.shape)


# --------------------------------------------------------------------------------------------------------------------
# 4- TREINAMENTO/AJUSTE DOS MODELOS DE MACHINE LEARNING E ESTATÍSTICO
# --------------------------------------------------------------------------------------------------------------------

modelos = {}

# -----------------------------
# A) Regressão Logística
# -----------------------------
pipeline_lr = Pipeline(steps=[
    ("prep", preprocess_ml),
    ("clf", LogisticRegression(max_iter=500))
])
pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)
modelos["Logistic Regression"] = (pipeline_lr, y_pred_lr)


# -----------------------------
# B) Árvore de Decisão
# -----------------------------
pipeline_dt = Pipeline(steps=[
    ("prep", preprocess_ml),
    ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))
])
pipeline_dt.fit(X_train, y_train)
y_pred_dt = pipeline_dt.predict(X_test)
modelos["Decision Tree"] = (pipeline_dt, y_pred_dt)

# -----------------------------
# C) Random Forest
# -----------------------------

pipeline_rf = Pipeline(steps=[
    ("prep", preprocess_ml),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    ))
])

pipeline_rf.fit(X_train, y_train)
y_pred_rf = pipeline_rf.predict(X_test)
modelos["Random Forest"] = (pipeline_rf, y_pred_rf)

# -----------------------------
# D) Gradient Boosting
# -----------------------------
pipeline_gb = Pipeline(steps=[
    ("prep", preprocess_ml),
    ("clf", GradientBoostingClassifier(random_state=42))
])
pipeline_gb.fit(X_train, y_train)
y_pred_gb = pipeline_gb.predict(X_test)
modelos["Gradient Boosting"] = (pipeline_gb, y_pred_gb)


# ----------------------------------------
# E) SVM - support vector machine
# ----------------------------------------
pipeline_svm = Pipeline(steps=[
    ("prep", preprocess_ml),
    ("clf", SVC(probability=True, random_state=42))
])
pipeline_svm.fit(X_train, y_train)
y_pred_svm = pipeline_svm.predict(X_test)
modelos["SVM"] = (pipeline_svm, y_pred_svm)


# ---------------------------------------------
# F) Rede Neural (MLP)- Multilayer Perceptron
# ----------------------------------------------
pipeline_mlp = Pipeline(steps=[
    ("prep", preprocess_ml),
    ("clf", MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=800, random_state=42))
])
pipeline_mlp.fit(X_train, y_train)
y_pred_mlp = pipeline_mlp.predict(X_test)
modelos["MLP"] = (pipeline_mlp, y_pred_mlp)


print("\nTreinamento concluído para todos os modelos!")
print("Modelos armazenados no dicionário 'modelos' para avaliação posterior.")


# --------------------------------------------------------------------------------------------
# G) MODELO ESTATÍSTICO (STATSMODELS) — STEPWISE FORWARD POR BLOCO (ESTILO SAS/SPSS/R)
# --------------------------------------------------------------------------------------------

print("\n==========================================")
print(" STEPWISE FORWARD POR BLOCO (SAS/SPSS/R) ")
print("==========================================\n")

# Construir dummies
X_stats = pd.get_dummies(
    credito_wkd[continuas + ordinais + nominais],
    drop_first=True
).astype(float)

y_stats = credito_wkd[target].astype(int)

# -----------------------------------------------------------
# Identificação de blocos (variáveis categóricas)
# -----------------------------------------------------------
blocos = {
    nom: [c for c in X_stats.columns if c.startswith(nom + "_")]
    for nom in nominais
}

# Variáveis contínuas e ordinais entram individualmente
simples = [
    c for c in X_stats.columns
    if not any(c.startswith(n + "_") for n in nominais)
]

candidatos = simples + list(blocos.keys())
incluidos = []


# -----------------------------------------------------------
# STEPWISE FORWARD - LIKELIHOOD RATIO TEST POR BLOCO
# -----------------------------------------------------------

while True:
    melhor_p = 1
    melhor_var = None

    for var in candidatos:
        if var in incluidos:
            continue

        # Variáveis já incluídas
        cols_base = []
        for v in incluidos:
            if v in blocos:
                cols_base += blocos[v]
            else:
                cols_base.append(v)

        # Variáveis a testar
        if var in blocos:
            cols_teste = cols_base + blocos[var]
        else:
            cols_teste = cols_base + [var]

        # Modelo restrito
        if len(cols_base) > 0:
            modelo_restrito = sm.Logit(y_stats, sm.add_constant(X_stats[cols_base])).fit(disp=0)
        else:
            modelo_restrito = sm.Logit(y_stats, np.ones((len(y_stats), 1))).fit(disp=0)

        # Modelo completo
        modelo_completo = sm.Logit(y_stats, sm.add_constant(X_stats[cols_teste])).fit(disp=0)

        # Likelihood Ratio Test
        lr = 2 * (modelo_completo.llf - modelo_restrito.llf)
        df = modelo_completo.df_model - modelo_restrito.df_model
        pval = 1 - stats.chi2.cdf(lr, df) 

        if pval < melhor_p:
            melhor_p = pval
            melhor_var = var

    if melhor_p < 0.05:
        print(f"✔ Incluindo variável (bloco): {melhor_var}  — p-LRT = {melhor_p:.5f}")
        incluidos.append(melhor_var)
    else:
        print("\nNenhum bloco adicional significativo. Encerrando Stepwise.")
        break

# -----------------------------------------------------------
# Expandir blocos selecionados
# -----------------------------------------------------------
variaveis_finais = []
for v in incluidos:
    if v in blocos:
        variaveis_finais += blocos[v]
    else:
        variaveis_finais.append(v)

print("\n================ VARIÁVEIS FINAIS DO MODELO =================\n")
print(variaveis_finais)

# -----------------------------------------------------------
# Rodar o modelo final com TODAS AS VARIÁVEIS DO BLOCO
# -----------------------------------------------------------

modelo_final = sm.Logit(
    y_stats,
    sm.add_constant(X_stats[variaveis_finais])
).fit()

print("\n================= MODELO FINAL (LOGIT) =================")
print(modelo_final.summary())

# -----------------------------------------------------------
# DataFrame final com coeficientes
# -----------------------------------------------------------

params = modelo_final.params
stderr = modelo_final.bse
zstat = modelo_final.tvalues
pvals = modelo_final.pvalues
conf  = modelo_final.conf_int()
conf.columns = ["IC 2.5%", "IC 97.5%"]

df_modelo_final = pd.DataFrame({
    "Variável": params.index,
    "Coeficiente": params.values,
    "StdErr": stderr.values,
    "z": zstat.values,
    "p-valor": pvals.values,
    "IC 2.5%": conf["IC 2.5%"].values,
    "IC 97.5%": conf["IC 97.5%"].values
})

print("\n================ DATAFRAME COMPLETO DO MODELO FINAL ================\n")
print(df_modelo_final)

# Exportar lista para uso no SKLearn:
lista_variaveis_para_sklearn = [
    v.replace("_", "").split("_")[0] if v in blocos else v for v in incluidos
]



df_modelo_final["Variável"].tolist()

# --------------------------------------------------------------------------------------------------------------------
# 4- AVALIAÇÃO DOS MODELOS DE MACHINE LEARNING 
# --------------------------------------------------------------------------------------------------------------------

# --------------------------------------
# 4.0 — Função auxiliar para métricas ML

def ks_score(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return max(tpr - fpr)

def avaliar_modelo(nome, y_test, y_pred, y_proba=None):
    print(f"\n==================== {nome} ====================")

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        ks  = ks_score(y_test, y_proba)
    else:
        auc = None
        ks  = None

    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"AUC      : {auc:.4f}" if auc is not None else "AUC      : (não disponível)")
    print(f"KS       : {ks:.4f}"  if ks  is not None else "KS       : (não disponível)")

    print("\nMatriz de Confusão:")
    print(confusion_matrix(y_test, y_pred))


# --------------------------------------------------------------------------------------------------------------------
# 4.1 — MODELOS DE MACHINE LEARNING
# --------------------------------------------------------------------------------------------------------------------

# ----------------------------- Logistic Regression -----------------------------
y_pred_lr  = pipeline_lr.predict(X_test)
y_proba_lr = pipeline_lr.predict_proba(X_test)[:, 1]
avaliar_modelo("Logistic Regression (sklearn)", y_test, y_pred_lr, y_proba_lr)


# ----------------------------- Decision Tree -----------------------------------
y_pred_dt  = pipeline_dt.predict(X_test)
y_proba_dt = pipeline_dt.predict_proba(X_test)[:, 1]
avaliar_modelo("Decision Tree", y_test, y_pred_dt, y_proba_dt)


# ----------------------------- Random Forest -----------------------------------
y_pred_rf  = pipeline_rf.predict(X_test)
y_proba_rf = pipeline_rf.predict_proba(X_test)[:, 1]
avaliar_modelo("Random Forest", y_test, y_pred_rf, y_proba_rf)


# ----------------------------- Gradient Boosting -------------------------------
y_pred_gb  = pipeline_gb.predict(X_test)
y_proba_gb = pipeline_gb.predict_proba(X_test)[:, 1]
avaliar_modelo("Gradient Boosting", y_test, y_pred_gb, y_proba_gb)


# ----------------------------- SVM ---------------------------------------------
y_pred_svm  = pipeline_svm.predict(X_test)
y_proba_svm = pipeline_svm.predict_proba(X_test)[:, 1]
avaliar_modelo("SVM (probability=True)", y_test, y_pred_svm, y_proba_svm)


# ----------------------------- MLP ---------------------------------------------
y_pred_mlp  = pipeline_mlp.predict(X_test)
y_proba_mlp = pipeline_mlp.predict_proba(X_test)[:, 1]
avaliar_modelo("MLP (Neural Network)", y_test, y_pred_mlp, y_proba_mlp)


# --------------------------------------------------------------------------------------------------------------------
# 5 — DATAFRAME FINAL DE COMPARAÇÃO DE MODELOS
# --------------------------------------------------------------------------------------------------------------------

resultados_finais = []

def registrar_resultado(nome, y_test, y_pred, y_proba=None):
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    ks   = ks_score(y_test, y_proba)       if y_proba is not None else None
    
    resultados_finais.append({
        "Modelo": nome,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "AUC": auc,
        "KS": ks
    })


# -------------------------------------------------------
# Registrar modelos de Machine Learning
# -------------------------------------------------------

registrar_resultado("Logistic Regression (sklearn)", y_test, y_pred_lr, y_proba_lr)
registrar_resultado("Decision Tree", y_test, y_pred_dt, y_proba_dt)
registrar_resultado("Random Forest", y_test, y_pred_rf, y_proba_rf)
registrar_resultado("Gradient Boosting", y_test, y_pred_gb, y_proba_gb)
registrar_resultado("SVM", y_test, y_pred_svm, y_proba_svm)
registrar_resultado("MLP Neural Network", y_test, y_pred_mlp, y_proba_mlp)


# -------------------------------------------------------
# Registrar modelo estatístico (STEPWISE)
# -------------------------------------------------------

# Criar dummies NO TESTE com as MESMAS colunas do X_stats
X_test_dummies = pd.get_dummies(X_test, drop_first=True).astype(int)

# Garantir que X_test_dummies tenha TODAS as colunas que existem em X_stats
X_test_dummies = X_test_dummies.reindex(columns=X_stats.columns, fill_value=0)

# Selecionar apenas as variáveis escolhidas pelo Stepwise
X_test_sw = X_test_dummies[variaveis_finais]


# Adicionar intercepto
X_test_sw_const = sm.add_constant(X_test_sw, has_constant='add')

# Previsões do modelo
y_proba_step = modelo_final.predict(X_test_sw_const)
y_pred_step  = (y_proba_step >= 0.5).astype(int)

# Registrar resultados no DataFrame final
registrar_resultado("Logistic Regression (Stepwise)", y_test, y_pred_step, y_proba_step)


# ------------------------------------------------------------------------------------------------------------------------
# Criar o DataFrame organizado com a comparação dos modelos 
# ------------------------------------------------------------------------------------------------------------------------
def avaliar_train_test(modelo, X_train, y_train, X_test, y_test, nome):
    
    # ----- PREVISÕES -----
    y_pred_train = modelo.predict(X_train)
    y_pred_test  = modelo.predict(X_test)

    y_proba_train = modelo.predict_proba(X_train)[:,1] if hasattr(modelo, "predict_proba") else None
    y_proba_test  = modelo.predict_proba(X_test)[:,1] if hasattr(modelo, "predict_proba") else None

    # ----- RESULTADO -----
    return {
        "Modelo": nome,
        "Acc Treino": accuracy_score(y_train, y_pred_train),
        "Acc Teste":  accuracy_score(y_test, y_pred_test),

        "Precision Treino": precision_score(y_train, y_pred_train),
        "Precision Teste":  precision_score(y_test, y_pred_test),

        "Recall Treino": recall_score(y_train, y_pred_train),
        "Recall Teste":  recall_score(y_test, y_pred_test),

        "F1 Treino": f1_score(y_train, y_pred_train),
        "F1 Teste":  f1_score(y_test, y_pred_test),

        "AUC Treino": roc_auc_score(y_train, y_proba_train) if y_proba_train is not None else None,
        "AUC Teste":  roc_auc_score(y_test,  y_proba_test ) if y_proba_test  is not None else None,

        "KS Treino": ks_score(y_train, y_proba_train) if y_proba_train is not None else None,
        "KS Teste":  ks_score(y_test,  y_proba_test ) if y_proba_test  is not None else None
    }




# --------------------------------------------------------------------------------------------------------------------
# 5.1 — AVALIAÇÃO DOS MODELOS SKLEARN
# --------------------------------------------------------------------------------------------------------------------

comparacao = []

comparacao.append(avaliar_train_test(pipeline_lr,  X_train, y_train, X_test, y_test, "Logistic Regression"))
comparacao.append(avaliar_train_test(pipeline_dt,  X_train, y_train, X_test, y_test, "Decision Tree"))
comparacao.append(avaliar_train_test(pipeline_rf,  X_train, y_train, X_test, y_test, "Random Forest"))
comparacao.append(avaliar_train_test(pipeline_gb,  X_train, y_train, X_test, y_test, "Gradient Boosting"))
comparacao.append(avaliar_train_test(pipeline_svm, X_train, y_train, X_test, y_test, "SVM"))
comparacao.append(avaliar_train_test(pipeline_mlp, X_train, y_train, X_test, y_test, "MLP Neural Network"))


# --------------------------------------------------------------------------------------------------------------------
# 5.2 — AVALIAÇÃO DO MODELO STEPWISE
# --------------------------------------------------------------------------------------------------------------------

# ---------- TREINO ----------
X_train_dummies = pd.get_dummies(X_train, drop_first=True).astype(int)
X_train_dummies = X_train_dummies.reindex(columns=X_stats.columns, fill_value=0)
X_train_sw = X_train_dummies[variaveis_finais]
X_train_sw_const = sm.add_constant(X_train_sw, has_constant='add')

y_proba_train_step = modelo_final.predict(X_train_sw_const)
y_pred_train_step  = (y_proba_train_step >= 0.5).astype(int)

# ---------- TESTE ----------
X_test_dummies = pd.get_dummies(X_test, drop_first=True).astype(int)
X_test_dummies = X_test_dummies.reindex(columns=X_stats.columns, fill_value=0)
X_test_sw = X_test_dummies[variaveis_finais]
X_test_sw_const = sm.add_constant(X_test_sw, has_constant='add')

y_proba_test_step = modelo_final.predict(X_test_sw_const)
y_pred_test_step  = (y_proba_test_step >= 0.5).astype(int)

# ---------- INSERE NO DATAFRAME ----------
comparacao.append({
    "Modelo": "Logistic Regression (Stepwise)",
    "Acc Treino": accuracy_score(y_train, y_pred_train_step),
    "Acc Teste":  accuracy_score(y_test,  y_pred_test_step),

    "Precision Treino": precision_score(y_train, y_pred_train_step),
    "Precision Teste":  precision_score(y_test,  y_pred_test_step),

    "Recall Treino": recall_score(y_train, y_pred_train_step),
    "Recall Teste":  recall_score(y_test,  y_pred_test_step),

    "F1 Treino": f1_score(y_train, y_pred_train_step),
    "F1 Teste":  f1_score(y_test,  y_pred_test_step),

    "AUC Treino": roc_auc_score(y_train, y_proba_train_step),
    "AUC Teste":  roc_auc_score(y_test,  y_proba_test_step),

    "KS Treino": ks_score(y_train, y_proba_train_step),
    "KS Teste": ks_score(y_test, y_proba_test_step)
})


# --------------------------------------------------------------------------------------------------------------------
# 5.3 — DATAFRAME FINAL
# --------------------------------------------------------------------------------------------------------------------

df_comparacao = pd.DataFrame(comparacao)
df_comparacao = df_comparacao.sort_values(by="AUC Teste", ascending=False)

print("\n==================== COMPARAÇÃO COMPLETA TREINO × TESTE ====================\n")
print(df_comparacao)

df_comparacao



# --------------------------------------------------------------------------------------------------------------------
# 6 — MODELO FINAL EM SKLEARN USANDO APENAS AS VARIÁVEIS SELECIONADAS PELO STEPWISE
# --------------------------------------------------------------------------------------------------------------------

"""
OBJETIVO DA SEÇÃO:
------------------
Esta seção implementa no SKLearn um modelo de Regressão Logística contendo 
EXCLUSIVAMENTE as variáveis originais selecionadas pelo modelo estatístico 
(Logit Stepwise Forward por bloco).

Aqui também produzimos validações completas separadas para TREINO e TESTE:
- Accuracy
- Precision
- Recall
- F1
- AUC
- KS

Além disso, construímos:
- Decis
- Curva de Lift
- DataFrame final consolidado de métricas

O modelo reduzido será a versão oficial para deploy.
"""

# ============================================================
# 6.1 — Variáveis originais selecionadas pelo Stepwise
# ============================================================

variaveis_originais = [
    'historico_credito',
    'proposito',
    'habitacao',
    'estado_civil',
    'outros_financiamentos',
    'duracao',
    'valor',
    'idade',
    'cd_tempo_parcelamento'
]

print("\nVariáveis utilizadas no modelo reduzido SKLearn:")
print(variaveis_originais)

# ============================================================
# 6.2 — Construção do dataset reduzido
# ============================================================

X_reduzido = credito_wkd[variaveis_originais]
y_reduzido = credito_wkd[target].astype(int)

X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduzido, y_reduzido, test_size=0.30, random_state=42, stratify=y_reduzido
)

print("\nDimensões do dataset reduzido:")
print("Treino:", X_train_red.shape)
print("Teste :", X_test_red.shape)

# ============================================================
# 6.3 — Definição dos tipos de variáveis finais
# ============================================================

cont_final = [v for v in continuas if v in variaveis_originais]
ord_final  = [v for v in ordinais  if v in variaveis_originais]
cat_final  = [v for v in nominais  if v in variaveis_originais]

print("\nVariáveis contínuas no modelo:", cont_final)
print("Variáveis ordinais no modelo:", ord_final)
print("Variáveis categóricas no modelo:", cat_final)

# ============================================================
# 6.4 — Pipeline SKLearn
# ============================================================

preprocess_red = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), cont_final),
        ("ord", "passthrough", ord_final),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_final),
    ]
)

pipeline_lr_red = Pipeline(steps=[
    ("prep", preprocess_red),
    ("clf", LogisticRegression(max_iter=500))
])

pipeline_lr_red.fit(X_train_red, y_train_red)

print("\nModelo SKLearn reduzido treinado com sucesso!")

# ============================================================
# 6.5 — Previsões
# ============================================================

y_pred_train_red = pipeline_lr_red.predict(X_train_red)
y_pred_test_red  = pipeline_lr_red.predict(X_test_red)

y_proba_train_red = pipeline_lr_red.predict_proba(X_train_red)[:, 1]
y_proba_test_red  = pipeline_lr_red.predict_proba(X_test_red)[:, 1]

# ============================================================
# 6.6 — Funções Auxiliares
# ============================================================

def ks_score(y_true, y_proba):
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    return max(tpr - fpr)

# Decis e Lift
def gerar_decis(y_true, y_proba, n=10):
    df = pd.DataFrame({"y": y_true, "proba": y_proba})
    df["decil"] = pd.qcut(df["proba"], n, labels=False, duplicates="drop")
    resumo = df.groupby("decil").agg(
        total=("y","count"),
        maus=("y","sum"),
        bons=("y",lambda x: (1-x).sum()),
        media_proba=("proba","mean")
    ).reset_index()
    return resumo

def lift_curve(df):
    df = df.copy()
    df["acum_maus"] = df["maus"].cumsum()
    df["pct_maus_acum"] = df["acum_maus"] / df["maus"].sum()
    df["pct_clientes_acum"] = df["total"].cumsum() / df["total"].sum()
    df["lift"] = df["pct_maus_acum"] / df["pct_clientes_acum"]
    return df

# ============================================================
# 6.7 — DataFrame Final de Métricas (Treino e Teste)
# ============================================================

df_metrics_full = pd.DataFrame({
    "Conjunto": ["Treino", "Teste"],
    "Accuracy": [
        accuracy_score(y_train_red, y_pred_train_red),
        accuracy_score(y_test_red, y_pred_test_red)
    ],
    "Precision": [
        precision_score(y_train_red, y_pred_train_red),
        precision_score(y_test_red, y_pred_test_red)
    ],
    "Recall": [
        recall_score(y_train_red, y_pred_train_red),
        recall_score(y_test_red, y_pred_test_red)
    ],
    "F1": [
        f1_score(y_train_red, y_pred_train_red),
        f1_score(y_test_red, y_pred_test_red)
    ],
    "AUC": [
        roc_auc_score(y_train_red, y_proba_train_red),
        roc_auc_score(y_test_red, y_proba_test_red)
    ],
    "KS": [
        ks_score(y_train_red, y_proba_train_red),
        ks_score(y_test_red, y_proba_test_red)
    ]
})

print("\n=================== MÉTRICAS COMPLETAS (TREINO × TESTE) ===================\n")
print(df_metrics_full)

# ============================================================
# 6.8 — Decis e Lift (apenas para Teste)
# ============================================================

decis_red = gerar_decis(y_test_red, y_proba_test_red)
lift_red  = lift_curve(decis_red)

print("\n================== DECIS (MODELO REDUZIDO) ==================\n")
print(decis_red)

print("\n================== LIFT (MODELO REDUZIDO) ==================\n")
print(lift_red)

# ============================================================
# 6.9 — Salvar modelo reduzido (.pkl)
# ============================================================
import joblib
joblib.dump(pipeline_lr_red, "modelo_logistic_reduzido.pkl")
print("\nModelo reduzido salvo como: modelo_logistic_reduzido.pkl")

# --------------------------------------------------------------------------------------------------------------------
# 7 — SIMULAÇÃO DA ESCORAGEM DE CRÉDITO - DEPLOY DO MODELO REDUZIDO (SKLearn)
# --------------------------------------------------------------------------------------------------------------------

# O modelo selecionado para deploy é o Logistic Regression (SKLearn reduzido),
# contendo APENAS as variáveis selecionadas pelo Stepwise (por bloco).
# Isso garante:
#   - modelo mais parcimonioso
#   - interpretabilidade alinhada ao modelo estatístico
#   - consistência entre análise estatística e machine learning
#   - menor risco de overfitting

# Variáveis originais que entraram no modelo (por bloco)
variaveis_modelo = [
    'historico_credito',
    'proposito',
    'habitacao',
    'estado_civil',
    'outros_financiamentos',
    'duracao',
    'valor',
    'idade',
    'cd_tempo_parcelamento'
]

# --------------------------------------------------------------------------------------------------------------------
# 7.1 — Carregar nova base (simulação de dados reais)
# --------------------------------------------------------------------------------------------------------------------

Path_Credito = r"C:\Users\Jeislan\Documents\CURSOS\UDEMY\Projeto Prático Final\bases\TB_CREDITO_DEPLOY.csv"
dp_credito_wkd = pd.read_csv(Path_Credito, sep=',', encoding='utf-8')

print("\nPrévia da base de deploy:")
print(dp_credito_wkd.head())


# --------------------------------------------------------------------------------------------------------------------
# 7.2 — Aplicar as MESMAS transformações manuais feitas no treino
# --------------------------------------------------------------------------------------------------------------------

# residencia_desde
dp_credito_wkd["residencia_desde"] = dp_credito_wkd["residencia_desde"].fillna("desconhecido")
dp_credito_wkd["cd_residencia_desde"] = dp_credito_wkd["cd_residencia_desde"].fillna(0).astype(int)

# emprestimo_existente
dp_credito_wkd["emprestimo_existente"] = dp_credito_wkd["emprestimo_existente"].replace(
    {"Alto (5 ou mais empréstimos)": "Médio (3-4 empréstimos)"}
)
dp_credito_wkd["cd_emprestimo_existente"] = dp_credito_wkd["cd_emprestimo_existente"].replace({4: 3})

# historico_credito
dp_credito_wkd["historico_credito"] = dp_credito_wkd["historico_credito"].fillna("desconhecido")

# faixa_tempo_emprego
dp_credito_wkd["faixa_tempo_emprego"] = dp_credito_wkd["faixa_tempo_emprego"].fillna("desconhecido")
dp_credito_wkd["cd_emprego"] = dp_credito_wkd["cd_emprego"].fillna(0).astype(int)

# possui_fiador
dp_credito_wkd["possui_fiador"] = dp_credito_wkd["possui_fiador"].replace(
    {"co requerente": "co requerente/aplicador",
     "co aplicante": "co requerente/aplicador"}
)

# habitacao
dp_credito_wkd["habitacao"] = dp_credito_wkd["habitacao"].fillna("desconhecido")

# qualificacao_profissional
dp_credito_wkd["qualificacao_profissional"] = dp_credito_wkd["qualificacao_profissional"].fillna("desconhecido")


# --------------------------------------------------------------------------------------------------------------------
# 7.3 — Selecionar SOMENTE as variáveis usadas no modelo reduzido
# --------------------------------------------------------------------------------------------------------------------

X_dp_credito_wkd = dp_credito_wkd[variaveis_modelo]


# --------------------------------------------------------------------------------------------------------------------
# 7.4 — Aplicar o modelo reduzido (SKLearn)
# --------------------------------------------------------------------------------------------------------------------

y_pred_new  = pipeline_lr_red.predict(X_dp_credito_wkd)
y_proba_new = pipeline_lr_red.predict_proba(X_dp_credito_wkd)[:, 1]


# --------------------------------------------------------------------------------------------------------------------
# 7.5 — Adicionar resultados ao dataframe
# --------------------------------------------------------------------------------------------------------------------

dp_credito_wkd["score_modelo"] = y_pred_new
dp_credito_wkd["prob_mau_pagador"] = y_proba_new

print("\nAmostra das previsões:")
print(dp_credito_wkd[["idcredito", "score_modelo", "prob_mau_pagador"]].head())


# --------------------------------------------------------------------------------------------------------------------
# 7.6 — Salvar o modelo reduzido
# --------------------------------------------------------------------------------------------------------------------

import joblib

caminho_modelo_reduzido = "modelo_logistic_reduzido.pkl"
joblib.dump(pipeline_lr_red, caminho_modelo_reduzido)

print(f"\nModelo reduzido salvo como: {caminho_modelo_reduzido}")

# --------------------------------------------------------------------------------------------------------------------
# 8 — SIMULAÇÃO DA DISPONIBILIZAÇÃO DO MODELO PARA A ÁREA DE ENGENHARIA DE DADOS
# --------------------------------------------------------------------------------------------------------------------

import joblib

# --------------------------------------------------------------------------------------------------------------------
# 8.1 — Carregar o modelo reduzido
# --------------------------------------------------------------------------------------------------------------------

modelo_prod = joblib.load("modelo_logistic_reduzido.pkl")

print("\nModelo reduzido carregado com sucesso.")


# --------------------------------------------------------------------------------------------------------------------
# 8.2 — Selecionar as variáveis necessárias para o modelo
# --------------------------------------------------------------------------------------------------------------------

X_deploy = dp_credito_wkd[variaveis_modelo]


# --------------------------------------------------------------------------------------------------------------------
# 8.3 — Aplicar o modelo
# --------------------------------------------------------------------------------------------------------------------

dp_credito_wkd["score_modelo"] = modelo_prod.predict(X_deploy)
dp_credito_wkd["prob_mau_pagador"] = modelo_prod.predict_proba(X_deploy)[:, 1]


print("\nAmostra da escoragem gerada para deploy:")
print(dp_credito_wkd[["idcredito", "score_modelo", "prob_mau_pagador"]].head())


# --------------------------------------------------------------------------------------------------------------------
# 8.4 — Entrega final para Engenharia
# --------------------------------------------------------------------------------------------------------------------

print("""
ENTREGA PARA ENGENHARIA DE DADOS:
---------------------------------
1) Arquivo do modelo treinado:
   modelo_logistic_reduzido.pkl

2) Lista de variáveis obrigatórias (na ordem correta):
   ['historico_credito', 'proposito', 'habitacao', 'estado_civil',
    'outros_financiamentos', 'duracao', 'valor', 'idade', 'cd_tempo_parcelamento']

3) Exemplo de escoragem:
   modelo = joblib.load("modelo_logistic_reduzido.pkl")
   pred = modelo.predict(X_novo)
   proba = modelo.predict_proba(X_novo)[:, 1]

Pronto para produção.
""")

