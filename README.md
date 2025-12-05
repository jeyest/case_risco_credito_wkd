# case_risco_credito_wkd

Case completo de Engenharia de Dados e Modelagem de Risco de Crédito usando SQL (PostgreSQL) e Python. O projeto inclui um pipeline incremental baseado em views, uma versão one-shot otimizada, criação da tabela final e preparação do dataset para análise e modelagem.

Este projeto faz parte da formação:

Formação Cientista de Dados: O Curso Completo - 2025 (Fernando Amaral)  
Plataforma: Udemy

---

## Estrutura do Repositório

case_risco_credito_wkd/
│
├── sql/
│   ├── pipeline_sql_incremental.sql     # Pipeline incremental usando views
│   ├── pipeline_sql_one_shot.sql        # Pipeline completo em join único
│   └── .gitkeep
│
├── data/
│   └── .gitkeep                         # Arquivos CSV ou dumps
│
├── python/
│   └── .gitkeep                         # Scripts e notebooks
│
├── docs/
│   └── .gitkeep                         # Documentação complementar
│
└── README.md

---
## Objetivo do Projeto

Este case implementa um pipeline completo de engenharia de dados e modelagem de risco de crédito.  
Abrange desde a ingestão e manipulação de tabelas normalizadas até a construção de uma tabela analítica final, preparada para análises estatísticas e aplicação de modelos de machine learning.

O projeto demonstra:

- aplicação de boas práticas de Engenharia de Dados  
- versionamento com Git e GitHub  
- SQL estruturado tanto incremental quanto one-shot  
- construção da tabela final `TB_CREDITO`  
- análise exploratória e modelagem em Python

---

## Tecnologias Utilizadas

- **PostgreSQL** – integração e enriquecimento do dataset  
- **SQL** – views, materialização e construção do dataset final  
- **Git e GitHub** – versionamento e organização do projeto  
- **Python** – análise exploratória, modelagem e avaliação

---

## Arquitetura SQL Desenvolvida

### 1. Pipeline Incremental

**Arquivo:** sql/pipeline_sql_incremental.sql

Abordagem baseada em views, permitindo acompanhar o enriquecimento passo a passo.

Fluxo:

- CREDITO -> vw_credito_1 (histórico)  
- vw_credito_2 (propósito)  
- vw_credito_3 (investimentos)  
- vw_credito_4 (emprego)  
- vw_credito_5 (estado civil)  
- vw_credito_6 (fiador)  
- vw_credito_7 (habitação)  
- vw_credito_8 (outros financiamentos)  
- vw_credito_9 (profissão)

---

### 2. Pipeline One-Shot

**Arquivo:** sql/pipeline_sql_one_shot.sql

Abordagem otimizada que produz duas tabelas:

1. `TB_CREDITO_BRUTO` — join único contendo todas as dimensões  
2. `TB_CREDITO` — tabela final com nomes padronizados

Indicada para cargas completas e materialização final em Data Warehouse/Lakehouse.

---

## Tabela Final (TB_CREDITO)

A tabela consolidada contém:

- atributos demográficos  
- atributos financeiros  
- variáveis socioeconômicas  
- categorias derivadas das dimensões  
- variável-alvo (`target`) representando inadimplência

A `TB_CREDITO` está pronta para análise exploratória, construção de features e modelagem preditiva.

---

## Análise Exploratória (EDA)

**Arquivo:** python/eda_credito.ipynb

Principais análises realizadas:

- estatísticas descritivas  
- detecção de outliers  
- distribuição do target  
- análises bivariadas  
- correlações entre variáveis  
- avaliação de relevância por informação mútua

Principais insights:

- histórico de crédito tem forte relação com inadimplência  
- baixo tempo de emprego aumenta probabilidade de inadimplência  
- valores maiores de crédito apresentam maior variabilidade  
- estado civil e presença de fiador influenciam o risco

---

## Modelagem de Risco de Crédito

**Arquivo:** python/modelagem_credito.ipynb

Etapas executadas:

- divisão treino/teste  
- codificação categórica (One-Hot e Ordinal conforme necessário)  
- normalização das variáveis numéricas

Treinamento e validação de modelos:
  - Regressão Logística  
  - Árvore de Decisão  
  - Random Forest  
  - Gradient Boosting

Métricas utilizadas:

- AUC  
- KS  
- Matriz de Confusão  
- Curva ROC

Resultado do modelo final:

- AUC superior a 0.80  
- KS elevado e adequado para uso em score de risco  
- boa separação entre bons e maus pagadores  
- interpretabilidade por meio de coeficientes e SHAP

---

## Conclusões

O pipeline desenvolvido permite integrar e unificar dados de crédito, gerar um dataset analítico de qualidade, realizar EDA abrangente e treinar modelos preditivos robustos. O trabalho demonstra as etapas essenciais de um projeto de Ciência de Dados aplicado ao segmento financeiro: engenharia de dados, análise estatística, modelagem preditiva e documentação com versionamento.

---

## Autor

**JEYEST (Jeislan Carlos de Souza)**  
Projeto desenvolvido como parte da formação "Cientista de Dados - Udemy (Fernando Amaral, 2025)".
