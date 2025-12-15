# case_risco_credito_wkd

Case completo de Risco de Crédito: Engenharia de Dados, Estatística e Machine Learning

Este repositório apresenta um case completo de Engenharia de Dados e Modelagem de Risco de Crédito utilizando PostgreSQL, Python, Statsmodels e Scikit-Learn.

O projeto cobre todas as etapas de um processo de modelagem utilizado no setor financeiro, desde a construção do dataset até a disponibilização do modelo final para produção. Inclui modelagem estatística com seleção de variáveis por Stepwise Forward e comparação com modelos de machine learning. Ao final, simula o deploy da solução com a escoragem de crédito.

Este case faz parte da formação:
"Formação Cientista de Dados. O Curso Completo - 2025"
Instrutor: Fernando Amaral
Plataforma: Udemy

---
## Estrutura do Repositório

```
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
```

## Objetivo do Projeto

-O objetivo deste case é mostrar de forma integrada:

-A construção de um pipeline de dados completo.

-A criação da tabela analítica final TB_CREDITO.

-A análise exploratória dos dados.

-A aplicação de testes estatísticos.

-A modelagem preditiva usando abordagens estatísticas e de machine learning.

-A preparação do modelo final para deploy.

-O projeto demonstra práticas de Engenharia de Dados, Ciência de Dados e versionamento com Git.

## O projeto demonstra:

- aplicação de boas práticas de Engenharia de Dados  
- versionamento com Git e GitHub  
- SQL estruturado tanto incremental quanto one-shot  
- construção da tabela final `TB_CREDITO`  
- análise exploratória e modelagem em Python

---

## Tecnologias Utilizadas

-PostgreSQL para integração e enriquecimento dos dados.

-SQL para construção de views incrementais e pipelines one-shot.

-Python para análise exploratória, testes estatísticos e modelagem.

-Statsmodels para regressão logística no padrão SAS, SPSS e R.

-Scikit-Learn para machine learning supervisionado.

-Git e GitHub para versionamento e documentação.

---

## Arquitetura SQL

## Pipeline Incremental

Arquivo: sql/pipeline_sql_incremental.sql

Abordagem que utiliza views sucessivas, permitindo acompanhar o enriquecimento do dataset etapa por etapa.

Fluxo principal:

-CREDITO

-vw_credito_1 historico

-vw_credito_2 proposito

-vw_credito_3 investimentos

-vw_credito_4 emprego

-vw_credito_5 estado civil

-vw_credito_6 fiador

-vw_credito_7 habitacao

-vw_credito_8 outros financiamentos

-vw_credito_9 profissao

Essa abordagem facilita auditoria, qualidade dos dados e compreensão da linhagem.

## Pipeline One Shot

Arquivo: sql/pipeline_sql_one_shot.sql

Abordagem otimizada para ambientes produtivos, caso as bases de dados não sejam grandes o suficiente para impactar o desempenho e o ambiente. 
Gera:

TB_CREDITO_BRUTO com todos os joins diretos das dimensões.

TB_CREDITO tabela final normalizada para análise e modelagem.


## Tabela Final TB_CREDITO

A tabela final contém:

-variáveis demográficas

-variáveis financeiras

-características de crédito e risco

-indicadores socioeconômicos

-variável alvo target representando inadimplência

O dataset é entregue pronta para análise estatística e modelagem.

## Análise Exploratória dos Dados

Arquivo: python/eda_credito.ipynb

Principais análises:

-estatísticas descritivas

-distribuição do target

-histogramas, boxplots e análise de outliers

-cruzamento entre variáveis

-análise de proporções de categorias

-identificação de variáveis com relevância univariada

Principais conclusões:

- o histórico de crédito está entre as variáveis com maior relação com inadimplência

- variáveis socioeconômicas possuem menor poder discriminativo

- há desbalanceamento moderado (aprox. 70 por cento bons, 30 por cento maus), padrão aceitável em análise de crédito.

## Modelagem Estatística e Machine Learning

Arquivo: python/modelagem_credito.ipynb

O processo inclui:

-Tratamento de dados e engenharia de variáveis.

-Aplicação de One Hot Encoding e padronização.

-Split treino e teste com estratificação.

-Testes estatísticos para cada tipo de variável.

-Modelagem estatística com regressão logística (statsmodels).

-Seleção de variáveis usando Stepwise Forward por bloco:

-> se uma dummy do bloco é significativa, todo o bloco é mantido.

Modelos de machine learning:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

SVM

MLP Neural Network

## Métricas usadas

AUC

KS

matriz de confusão

precision

recall

F1

curva ROC

## Resultado

modelos apresentam desempenho esperado para bases com variáveis pouco informativas

a regressão logística se mostrou estável e interpretável

o Stepwise por bloco produziu um modelo mais parcimonioso

o modelo final usado para deploy foi a Regressão Logística no Scikit-Learn somente com as variáveis selecionadas pelo Stepwise

## Modelo Final para Produção

O modelo final utiliza apenas as variáveis selecionadas pelo Stepwise Forward por bloco, garantindo:

consistência estatística

interpretabilidade

menor risco de overfitting

facilidade de manutenção

O pipeline completo do Scikit-Learn inclui o pré-processamento e gera o arquivo modelo_logistic_reduzido.pkl para uso em produção.

## Conclusão

Este projeto demonstra um fluxo completo de risco de crédito, abordando:

engenharia de dados

análise exploratória

testes estatísticos

modelagem tradicional e moderna

seleção de variáveis

construção de modelo final pronto para deploy

O case reflete práticas adotadas no setor financeiro e entrega um pipeline reproduzível e documentado de ponta a ponta.

---

## Autor

**JEYEST (Jeislan Carlos de Souza)**  
Projeto desenvolvido como parte da formação "Cientista de Dados - Udemy (Fernando Amaral, 2025)".
