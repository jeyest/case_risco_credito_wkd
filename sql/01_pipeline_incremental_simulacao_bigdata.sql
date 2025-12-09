/* 
Embora o banco utilizado neste case seja pequeno, estou aplicando boas 
práticas de Engenharia de Dados voltadas a cenários de Big Data.

Por isso, os JOINs entre a tabela fato (CREDITO) e as tabelas tipo dimensão 
estão sendo realizados de forma incremental, validando cada etapa antes 
de integrar a próxima dimensão. Para tal, estou utilizando views.
*/

/* Etapa 1: Left join entre a tabela CREDITO e a tabela HISTORICO_CREDITO */

CREATE OR REPLACE VIEW vw_credito_1 AS
SELECT 
      "CREDITO".*, 
	  "HISTORICO_CREDITO"."HISTORICO" 
FROM public."CREDITO"
LEFT JOIN public."HISTORICO_CREDITO"
       ON "CREDITO"."HistoricoCredito" = "HISTORICO_CREDITO"."IDHISTCRED";

/* Exibe as linhas criada da primeira view */
SELECT * FROM vw_credito_1 LIMIT 100;


/* Etapa 2 — Left join entre a vw_credito_1 (já enriquecida no passo 1)
   e a tabela PROPOSITO */

CREATE OR REPLACE VIEW vw_credito_2 AS
SELECT 
      vw_credito_1.*,
	  "PROPOSITO"."PROPOSITO"
FROM vw_credito_1
LEFT JOIN public."PROPOSITO"
       ON vw_credito_1."Proposito" = "PROPOSITO"."IDPROPOSITO";

/* Exibe as linhas criada da segunda view */
SELECT * FROM vw_credito_2 LIMIT 100;

/* Etapa 3 — Left join entre a vw_credito_2 (já enriquecida no passo 2)
   e a tabela INVESTIMENTOS */

CREATE OR REPLACE VIEW vw_credito_3 AS
SELECT 
      vw_credito_2.*,
	  "INVESTIMENTOS"."INVESTIMENTOS"
FROM vw_credito_2
LEFT JOIN public."INVESTIMENTOS"
       ON vw_credito_2."Investimentos" = "INVESTIMENTOS"."IDINVESTIMENTOS";

/* Exibe as linhas criada da terceira view */
SELECT * FROM vw_credito_3 LIMIT 100;

/* Etapa 4 — Left join entre a vw_credito_3 (já enriquecida no passo 3)
   e a tabela EMPREGO */

CREATE OR REPLACE VIEW vw_credito_4 AS
SELECT 
      vw_credito_3.*,
	  "EMPREGO"."EMPREGO"
FROM vw_credito_3
LEFT JOIN public."EMPREGO"
       ON vw_credito_3."Emprego" = "EMPREGO"."IDEMPREGO";

/* Exibe as linhas criada da quarta view */
SELECT * FROM vw_credito_4 LIMIT 100;

/* Etapa 5 — Left join entre a vw_credito_4 (já enriquecida no passo 4)
   e a tabela ESTADOCIVIL*/

CREATE OR REPLACE VIEW vw_credito_5 AS
SELECT 
      vw_credito_4.*,
      "ESTADOCIVIL"."ESTADOCIVIL"
FROM vw_credito_4
LEFT JOIN public."ESTADOCIVIL"
       ON vw_credito_4."EstadoCivil" = "ESTADOCIVIL"."IDESTADOCIVIL";

/* Exibe as linhas criada da quinta view */
SELECT * FROM vw_credito_5 LIMIT 100;

/* Etapa 6 — Left join entre a vw_credito_5 (já enriquecida no passo 5)
   e a tabela FIADOR*/

CREATE OR REPLACE VIEW vw_credito_6 AS 
SELECT 
      vw_credito_5.*,
      "FIADOR"."FIADOR"
FROM  vw_credito_5
LEFT JOIN public."FIADOR"
     ON vw_credito_5."FiadorTerceiros" = "FIADOR"."IDFIADOR";
	 
/* Exibe as linhas criada da sexta view */
SELECT * FROM vw_credito_6 LIMIT 100;

/* Etapa 7 — Left join entre a vw_credito_6 (já enriquecida no passo 6)
   e a tabela HABITACAO*/

CREATE OR REPLACE VIEW vw_credito_7 AS 
SELECT 
      vw_credito_6.*,
      "HABITACAO"."HABITACAO"
FROM vw_credito_6
LEFT JOIN public."HABITACAO"
     ON vw_credito_6."Habitacao" = "HABITACAO"."IDHABITACAO";
	 
/* Exibe as linhas criada da sétima view */
SELECT * FROM vw_credito_7 LIMIT 100;


/* Etapa 8 — Left join entre a vw_credito_7 (já enriquecida no passo 7)
   e a tabela OUTROSFINANC*/

CREATE OR REPLACE VIEW vw_credito_8 AS
SELECT 
      vw_credito_7.*,
      "OUTROSFINANC"."OUTROSFINANCIAMENTOS"
FROM vw_credito_7
LEFT JOIN public."OUTROSFINANC"
       ON vw_credito_7."OutrosFinanciamentos" = "OUTROSFINANC"."IDOUTROSFINANC";
	   
/* Exibe as linhas criada da oitava view */
SELECT * FROM vw_credito_8 LIMIT 100;

/* Etapa 9 — Left join entre a vw_credito_8 (já enriquecida no passo 8)
   e a tabela PROFISSAO*/

CREATE OR REPLACE VIEW vw_credito_9 AS
SELECT 
      vw_credito_8.*, 
      "PROFISSAO"."PROFISSAO"
FROM vw_credito_8
LEFT JOIN public."PROFISSAO"
       ON vw_credito_8."Profissao" = "PROFISSAO"."IDPROFISSAO";

/* Exibe as linhas criada da noba view */
SELECT * FROM vw_credito_9 LIMIT 100;

/* Etapa 10 — Geração de uma tabela final a parti da View gerada na etapa 9*/

DROP TABLE IF EXISTS "TB_CREDITO";
CREATE TABLE "TB_CREDITO" AS 

SELECT 
       "IDCREDITO" as idcredito,
	   "Status" as target,
       "Duracao" as duracao,
	   "Valor" as valor,
	   "TempoParcelamento" as tempo_parcelamento,
       "ResidenciaDesde" as residencia_desde,
	   "Idade" as idade,
	   "EmprestimoExistente" as emprestimo_existente,
	   "Dependentes" as dependentes,
       "SocioEmpresa" as socio_empresa,
	   "Estrangeiro" as estrangeiro,
	   "HISTORICO" as historico_credito,
       "PROPOSITO" as proposito,
	   "INVESTIMENTOS" as faixa_investimento,
	   "EMPREGO" as faixa_tempo_emprego,
	   "ESTADOCIVIL" as estado_civil,
	   "FIADOR" as possui_fiador,
	   "HABITACAO" as habitacao,
	   "OUTROSFINANCIAMENTOS" as outros_financiamentos,
	   "PROFISSAO" as qualificacao_profissional
	   
FROM vw_credito_9;
	
SELECT * FROM "TB_CREDITO" LIMIT 100;   







