# case_risco_credito_wkd
Case completo de Engenharia de Dados e Modelagem de Risco de Crédito usando SQL (PostgreSQL), Python, EDA e Machine Learning. Inclui pipeline incremental, versão one-shot, tabela final e preparação para modelagem.

Este projeto faz parte da formação:

**Formação Cientista de Dados: O Curso Completo - 2025 (Fernando Amaral)**  
Plataforma: Udemy

O case foi expandido com boas práticas de Engenharia de Dados, Git, documentação estruturada e preparação avançada para modelagem de risco de crédito.

--

## Objetivo do projeto

Este case implementa um pipeline completo de engenharia de dados e modelagem de risco de crédito.
Abrange desde a ingestão e manipulação de tabelas normalizadas até a construção de uma tabela analítica final, preparada para análises estatísticas e aplicação de modelos de machine learning.

O objetivo é demonstrar, de forma integrada, todas as etapas necessárias para transformar dados brutos em informações prontas para uso em processos de decisão e modelagem preditiva.

O projeto aplica:

- boas práticas de Data Engineering  
- versionamento Git  
- SQL estruturado (incremental e one-shot)  
- criação de dataset final `TB_CREDITO`  
- preparação para EDA e modelagem em Python

---

## 🗂 Estrutura do Repositório

case_risco_credito_wkd/
│
├── sql/
│ ├── pipeline_sql_incremental.sql # Pipeline incremental usando views
│ ├── pipeline_sql_one_shot.sql # Pipeline completo em join único
│ └── .gitkeep
│
├── data/
│ └── .gitkeep # Arquivos CSV ou dumps (a preencher)
│
├── python/
│ └── .gitkeep # Scripts e notebooks de modelagem (em breve)
│
├── docs/
│ └── .gitkeep # Documentação complementar
│
└── README.md

---

## Tecnologias Utilizadas

- **PostgreSQL** – base relacional e enriquecimento de dados  
- **SQL** – joins, materializações, views e limpeza  
- **Git & GitHub** – versionamento e organização do repositório  
- **Python (futuro)** – EDA, feature engineering, modelagem e métricas  

---

## Pipeline Incremental (Views)

O arquivo: sql/pipeline_sql_incremental.sql

Contém 9 etapas sequenciais, cada uma adicionando uma dimensão ao dataset principal.

As views formam o fluxo:

CREDITO
→ vw_credito_1 (histórico)
→ vw_credito_2 (propósito)
→ vw_credito_3 (investimentos)
→ vw_credito_4 (emprego)
→ vw_credito_5 (estado civil)
→ vw_credito_6 (fiador)
→ vw_credito_7 (habitação)
→ vw_credito_8 (outros financiamentos)
→ vw_credito_9 (profissão)


### Por que usar incremental?

- facilita debugging  
- permite validação etapa a etapa  
- deixa o pipeline mais didático e auditável  

---

## Pipeline One-Shot

O arquivo: sql/pipeline_sql_one_shot.sql

Implementa uma abordagem otimizada em duas etapas:

### 1. TB_CREDITO_BRUTO  
Join único com todas as tabelas dimensão.

### 2. TB_CREDITO  
Tabela final, com nomes padronizados, pronta para análise e modelagem.

### ✔ Por que usar one-shot?

- útil para cargas completas (full load)  
- ideal para Data Warehouse e Lakehouse  
- simplifica a materialização final  

---

## Tabela Final: TB_CREDITO

A tabela contém variáveis categóricas e numéricas sobre:

- perfil do cliente  
- situação socioeconômica  
- características do crédito  
- fatores de risco  
- variável-alvo (`target`) indicando inadimplência  

Essa tabela será utilizada no Python para:

- EDA  
- construção de variáveis  
- modelagem preditiva  
- métricas de risco (ROC, Gini, KS, AUC)  

---

## Próximos Passos (Python)

O diretório `/python` receberá:

### 1. Carregamento da TB_CREDITO via pandas  
### 2. EDA completa (boxplot, distplots, correlações)  
### 3. Tratamento de valores ausentes  
### 4. Feature engineering  
### 5. Modelagem:  
- Regressão Logística  
- Árvore  
- Random Forest  
- Gradient Boosting  
- XGBoost / LightGBM  

### 6. Avaliação de modelos  
### 7. Explainability (SHAP)

---

## Contato

**Autor:** JEYEST (Jeislan Carlos de Souza)  
Repositório criado para fins educacionais e demonstração de boas práticas de Engenharia e Ciência de Dados.





