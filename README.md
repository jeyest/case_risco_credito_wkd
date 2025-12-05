# case_risco_credito_wkd

Case completo de Engenharia de Dados e Modelagem de Risco de Crédito usando SQL (PostgreSQL), Python, EDA e Machine Learning. Inclui pipeline incremental, versão one-shot, tabela final e preparação para modelagem.

Este projeto faz parte da formação:

Formação Cientista de Dados: O Curso Completo - 2025 (Fernando Amaral)  
Plataforma: Udemy

O case foi expandido com boas práticas de Engenharia de Dados, Git, documentação estruturada e preparação avançada para modelagem de risco de crédito.

---

## Estrutura do Repositório

```text
case_risco_credito_wkd/
│
├── sql/
│   ├── pipeline_sql_incremental.sql     # Pipeline incremental usando views
│   ├── pipeline_sql_one_shot.sql        # Pipeline completo em join único
│   └── .gitkeep
│
├── data/
│   └── .gitkeep                         # Arquivos CSV ou dumps (a preencher)
│
├── python/
│   └── .gitkeep                         # Scripts e notebooks de modelagem (em breve)
│
├── docs/
│   └── .gitkeep                         # Documentação complementar
│
└── README.md

---

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

## Tecnologias Utilizadas

- **PostgreSQL** – base relacional e enriquecimento de dados  
- **SQL** – joins, materializações, views e limpeza  
- **Git & GitHub** – versionamento e organização do repositório  
- **Python (futuro)** – EDA, feature engineering, modelagem e métricas  

---
## Arquitetura SQL Desenvolvida

1. Pipeline Incremental

Arquivo: sql/pipeline_sql_incremental.sql

Abordagem baseada em views, onde cada etapa adiciona uma dimensão ao dataset principal. Esse método permite validação etapa a etapa, debugging mais simples e maior transparência no processo de enriquecimento dos dados.

Fluxo de construção:

CREDITO
 -> vw_credito_1  (histórico)
 -> vw_credito_2  (propósito)
 -> vw_credito_3  (investimentos)
 -> vw_credito_4  (emprego)
 -> vw_credito_5  (estado civil)
 -> vw_credito_6  (fiador)
 -> vw_credito_7  (habitação)
 -> vw_credito_8  (outros financiamentos)
 -> vw_credito_9  (profissão)

2. Pipeline One-Shot

Arquivo: sql/pipeline_sql_one_shot.sql

Abordagem otimizada baseada em duas etapas:

Criação da tabela TB_CREDITO_BRUTO contendo todos os joins em um único comando.

Criação da tabela final TB_CREDITO com renomeações e seleção dos campos de interesse.

Essa abordagem é recomendada para cargas completas, materialização final e ambientes de Data Warehouse ou Data Lakehouse.

Tabela Final (TB_CREDITO)

A tabela TB_CREDITO contém:

- atributos demográficos

- atributos financeiros

- variáveis socioeconômicas

- atributos categóricos oriundos das dimensões

- variável-alvo (Status), renomeada para target

O resultado é uma tabela consolidada e preparada para:

- análise exploratória

- construção de features

- modelagem de risco de crédito


Análise Exploratória (EDA)

Arquivo: python/eda_credito.ipynb

Principais análises realizadas:

estatísticas descritivas por variável

detecção de outliers

distribuição do target

análise bivariada entre variáveis categóricas e inadimplência

correlações entre variáveis numéricas

avaliação de importância inicial por informação mútua

Principais insights:

Clientes com histórico ruim apresentam inadimplência significativamente maior.

Baixo tempo de emprego aumenta risco de default.

Valores de crédito mais altos concentram maior variabilidade.

Estado civil e presença de fiador mostraram forte relação com o target.

Modelagem de Risco de Crédito

Arquivo: python/modelagem_credito.ipynb

Etapas executadas:

Divisão treino/teste

Codificação categórica (One-Hot e Ordinal Encoding conforme o modelo)

Normalização de variáveis numéricas

Treinamento de múltiplos modelos:

Regressão Logística

Árvore de Decisão

Random Forest

Gradient Boosting

Avaliação usando:

AUC

KS

Matriz de Confusão

Curva ROC

Modelo Final Selecionado

O modelo escolhido apresentou:

AUC superior a 0.80

KS elevado, adequado para análise de risco

boa separação entre bons e maus pagadores

interpretabilidade através dos coeficientes e gráficos SHAP

Conclusões

O pipeline desenvolvido permite:

integrar e unificar dados de crédito de forma consistente

gerar um dataset de alta qualidade

realizar EDA detalhada

treinar modelos robustos de classificação

identificar padrões relevantes de risco

O trabalho demonstra domínio das etapas essenciais de um projeto de Ciência de Dados aplicado ao segmento financeiro:

Engenharia de Dados

Análise estatística

Modelagem preditiva

Documentação e versionamento

Autor

JEYEST (Jeislan Carlos de Souza)
Projeto desenvolvido como parte da formação "Cientista de Dados - Udemy (Fernando Amaral, 2025)".



