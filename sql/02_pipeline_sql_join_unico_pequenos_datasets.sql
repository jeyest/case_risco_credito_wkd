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
       "IDCREDITO"::integer     AS idcredito,
       "Status"::integer        AS target,
       "Duracao"::integer       AS duracao,
       "Valor"::integer         AS valor,
       "TempoParcelamento"::integer AS cd_tempo_parcelamento,
       CASE
            WHEN "TempoParcelamento" = 1 THEN '1-Parcelamento em até 3 meses'
            WHEN "TempoParcelamento" = 2 THEN '2-Parcelamento de 4 a 6 meses'
            WHEN "TempoParcelamento" = 3 THEN '3-Parcelamento de 7 a 12 meses'
            WHEN "TempoParcelamento" = 4 THEN '4-Parcelamento acima de 12 meses'
       END                       AS tempo_parcelamento,

       "ResidenciaDesde"::integer AS cd_residencia_desde,
       CASE
            WHEN "ResidenciaDesde" = 1 THEN '1-Residência fixa'
            WHEN "ResidenciaDesde" = 2 THEN '2-Residência temporária'
            WHEN "ResidenciaDesde" = 3 THEN '3-Residência com familiares'
            WHEN "ResidenciaDesde" = 4 THEN '4-Residência em instituições'
       END                       AS residencia_desde,

       "Idade"::integer          AS idade,

       "EmprestimoExistente"::integer AS cd_emprestimo_existente,
       CASE
            WHEN "EmprestimoExistente" = 1 THEN 'Nenhum'
            WHEN "EmprestimoExistente" = 2 THEN 'Baixo (1-2 empréstimos)'
            WHEN "EmprestimoExistente" = 3 THEN 'Médio (3-4 empréstimos)'
            WHEN "EmprestimoExistente" = 4 THEN 'Alto (5 ou mais empréstimos)'
       END                       AS emprestimo_existente,

       "Dependentes"::integer   AS dependentes,
       "SocioEmpresa"::integer  AS socio_empresa,

       "Estrangeiro"::integer   AS cd_estrangeiro,
       CASE
            WHEN "Estrangeiro" = 1 THEN 'Sim'
            WHEN "Estrangeiro" = 0 THEN 'Não'
       END                       AS estrangeiro,

       "HISTORICO"              AS historico_credito,
       "PROPOSITO"              AS proposito,

       "Investimentos"::integer AS cd_investimentos,
       REPLACE("INVESTIMENTOS", '\\n', '') 
                                AS faixa_investimento,

       "Emprego"::integer       AS cd_emprego,
       "EMPREGO"                AS faixa_tempo_emprego,

       "ESTADOCIVIL"            AS estado_civil,
       "FIADOR"                 AS possui_fiador,
       "HABITACAO"              AS habitacao,

       "Profissao"::integer     AS cd_profissao,
       "OUTROSFINANCIAMENTOS"   AS outros_financiamentos,
       "PROFISSAO"              AS qualificacao_profissional

FROM "TB_CREDITO_BRUTO";

SELECT * FROM "TB_CREDITO" LIMIT 100;



