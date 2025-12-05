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
│ ├── pipeline_sql_incremental.sql # Pipeline incremental (views)
│ ├── pipeline_sql_one_shot.sql # Pipeline em join único (materialização)
│ └── .gitkeep
│
├── data/
│ └── .gitkeep # Dados brutos ou extraídos
│
├── python/
│ └── .gitkeep # Scripts de EDA e modelagem (em breve)
│
├── docs/
│ └── .gitkeep # Documentação complementar
│
└── README.md


---

## 🛠 Tecnologias utilizadas

- **PostgreSQL** – modelagem e enriquecimento das tabelas  
- **SQL** – joins, views e materialização de tabelas analíticas  
- **Git + GitHub** – versionamento e estruturação do projeto  
- **Python (em breve)** – EDA, feature engineering e modelagem preditiva  

---

## 🧱 Pipeline Incremental (Views)

O arquivo: sql/pipeline_sql_incremental.sql

Implementa um fluxo incremental clássico:

1. Cada etapa adiciona uma dimensão.
2. O enriquecimento pode ser validado passo a passo.
3. Views permitem auditoria e debugging.

Exemplo do fluxo:

CREDITO
→ vw_credito_1 ( + histórico )
→ vw_credito_2 ( + propósito )
→ vw_credito_3 ( + investimentos )
...
→ vw_credito_9 ( + profissão )


---

## ⚡ Pipeline One-Shot

O arquivo: sql/pipeline_sql_one_shot.sql

Contém uma abordagem otimizada em duas etapas:

1. **TB_CREDITO_BRUTO** — join único com todas as dimensões  
2. **TB_CREDITO** — tabela final renomeada e padronizada (dataset para modelagem)

Ideal para Data Warehouse, Data Lakehouse ou cargas completas.

---

## Tabela Final: TB_CREDITO

A tabela resultante contém variáveis:

- demográficas  
- financeiras  
- socioeconômicas  
- categóricas enriquecidas pelas dimensões  
- variável-alvo: **target** (`Status` 0/1)

Pronta para:

- EDA  
- feature engineering  
- algoritmos de machine learning  

---

## Próximos passos (Python)

1. Carregar `TB_CREDITO` no ambiente Python  
2. Limpeza e tratamento de dados  
3. Análise Exploratória (EDA)
4. Codificação categórica  
5. Balanceamento (se necessário)
6. Modelos:
   - Regressão Logística
   - Árvores
   - Random Forest
   - Gradient Boosting
   - Outros modelos candidatos
7. Avaliação (ROC, KS, Gini, AUC)
8. Explainability (SHAP)

O diretório `/python` será preenchido com notebooks e scripts.

---

## Status Atual do Projeto

✔ Estrutura Git/GitHub criada  
✔ Pipelines SQL incremental e one-shot  
✔ Preparação do ambiente para próxima etapa  

Próximo: iniciar EDA e modelagem em Python.

---

## Contato

*Autor: (Jeislan Carlos de Souza)
Projeto desenvolvido para fins educacionais e demonstração de boas práticas de Engenharia de Dados.


