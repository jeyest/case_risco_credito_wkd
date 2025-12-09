/* 
Etapa 1 — Criação da tabela intermediária (staging) utilizando
um único comando SQL contendo TODOS os JOINs necessários.
Seria uma versão tipo star schema.

Essa etapa realiza de uma só vez toda a integração entre a 
tabela fato (CREDITO) e as tabelas tipo dimensão, criando um dataset 
bruto completo. Esta abordagem substitui o processo incremental 
via múltiplas views, consolidando todos os relacionamentos em 
um único passo.
*/


DROP TABLE IF EXISTS "TB_CREDITO_BRUTO";

CREATE TABLE "TB_CREDITO_BRUTO" AS
SELECT 
       *
FROM public."CREDITO"

LEFT JOIN public."HISTORICO_CREDITO"
       ON "CREDITO"."HistoricoCredito" = "HISTORICO_CREDITO"."IDHISTCRED"

LEFT JOIN public."PROPOSITO"
       ON "CREDITO"."Proposito" = "PROPOSITO"."IDPROPOSITO"

LEFT JOIN public."INVESTIMENTOS"
       ON "CREDITO"."Investimentos" = "INVESTIMENTOS"."IDINVESTIMENTOS"

LEFT JOIN public."EMPREGO"
       ON "CREDITO"."Emprego" = "EMPREGO"."IDEMPREGO"

LEFT JOIN public."ESTADOCIVIL"
       ON "CREDITO"."EstadoCivil" = "ESTADOCIVIL"."IDESTADOCIVIL"

LEFT JOIN public."FIADOR"
       ON "CREDITO"."FiadorTerceiros" = "FIADOR"."IDFIADOR"

LEFT JOIN public."HABITACAO"
       ON "CREDITO"."Habitacao" = "HABITACAO"."IDHABITACAO"

LEFT JOIN public."OUTROSFINANC"
       ON "CREDITO"."OutrosFinanciamentos" = "OUTROSFINANC"."IDOUTROSFINANC"

LEFT JOIN public."PROFISSAO"
       ON "CREDITO"."Profissao" = "PROFISSAO"."IDPROFISSAO";

/* 
Etapa 2 — Criação da tabela final (curated) a partir do resultado do JOIN único.

Nesta etapa, selecionamos apenas os atributos relevantes e aplicamos 
renomeações consistentes. O objetivo é gerar uma camada analítica 
curada, semelhante à transformação da camada staging para uma camada gold 
em arquiteturas data warehouse baseadas em Star Schema.

Esta tabela representa a visão final consolidada para análise e modelagem.
*/
	   DROP TABLE IF EXISTS "TB_CREDITO";

CREATE TABLE "TB_CREDITO" AS
SELECT
       "IDCREDITO"              AS idcredito,
       "Status"                 AS target,
       "Duracao"                AS duracao,
       "Valor"                  AS valor,
       "TempoParcelamento"      AS tempo_parcelamento,
       "ResidenciaDesde"        AS residencia_desde,
       "Idade"                  AS idade,
       "EmprestimoExistente"    AS emprestimo_existente,
       "Dependentes"            AS dependentes,
       "SocioEmpresa"           AS socio_empresa,
       "Estrangeiro"            AS estrangeiro,

       "HISTORICO"              AS historico_credito,
       "PROPOSITO"              AS proposito,
       "INVESTIMENTOS"          AS faixa_investimento,
       "EMPREGO"                AS faixa_tempo_emprego,
       "ESTADOCIVIL"            AS estado_civil,
       "FIADOR"                 AS possui_fiador,
       "HABITACAO"              AS habitacao,
       "OUTROSFINANCIAMENTOS"   AS outros_financiamentos,
       "PROFISSAO"              AS qualificacao_profissional
FROM "TB_CREDITO_BRUTO";

SELECT * FROM "TB_CREDITO" LIMIT 100;

