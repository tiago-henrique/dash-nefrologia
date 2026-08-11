import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from matplotlib.ticker import PercentFormatter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from datetime import datetime
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo
import streamlit as st

# ================= CONFIGURAÇÃO =================
st.set_page_config(
    page_title="Kaplan-Meier – Transplante Renal",
    layout="wide"
)

st.title("Análise de Sobrevida – Transplante Renal")

# ================= LEITURA =================

file_path = st.secrets["CAMINHO"]

response = requests.head(file_path)
last_modified = response.headers.get("Last-Modified")

if last_modified:
    dt_utc = parsedate_to_datetime(last_modified)
    dt_br = dt_utc.astimezone(ZoneInfo("America/Sao_Paulo"))

    st.success(
        f"Dados atualizados em: {dt_br.strftime('%d/%m/%Y %H:%M:%S')}"
    )
else:
    st.warning("Cabeçalho Last-Modified não encontrado.")

# Carregar base
try:
    uploaded_file = pd.read_csv(st.secrets['DATABASE'])
except Exception as e:
    st.error(f"Erro ao carregar o banco de dados: {e}")
    st.stop()
    
df = uploaded_file

#uploaded_file = "indicadores.csv"
#df = pd.read_csv(uploaded_file)
# ================= TRATAMENTO DE DATAS =================
for col in ["data_tx", "data_obito", "data_pe"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df["ano_tx"] = df["data_tx"].dt.year
data_censura = pd.to_datetime(datetime.today().date())

# ================= EVENTOS =================
df["evento_obito"] = df["data_obito"].notna().astype(int)
df["tempo_obito"] = (
    df["data_obito"].fillna(data_censura) - df["data_tx"]
).dt.days

df["evento_pe"] = df["data_pe"].notna().astype(int)
df["tempo_pe"] = (
    df["data_pe"].fillna(data_censura) - df["data_tx"]
).dt.days

df["tempo_obito_anos"] = df["tempo_obito"] / 365.25
df["tempo_pe_anos"] = df["tempo_pe"] / 365.25

anos = sorted(df["ano_tx"].dropna().unique())

# ==========================================================
# 🔹 RESUMO DE EVENTOS POR ANO
# ==========================================================
tabela_resumo = (
    df.groupby("ano_tx")
    .agg(
        total_transplantes=("ano_tx", "count"),
        obitos=("evento_obito", "sum"),
        perda_enxerto=("evento_pe", "sum")
    )
    .reset_index()
    .sort_values("ano_tx")
)

tabela_resumo["taxa_obito_%"] = (
    tabela_resumo["obitos"] / tabela_resumo["total_transplantes"] * 100
).round(1)

tabela_resumo["taxa_perda_enxerto_%"] = (
    tabela_resumo["perda_enxerto"] / tabela_resumo["total_transplantes"] * 100
).round(1)

st.subheader("Resumo de Eventos por Ano do Transplante")
st.dataframe(tabela_resumo, use_container_width=True)

# ==========================================================
# 🔹 COMPARAÇÃO ESTATÍSTICA (LOG-RANK)
# ==========================================================
comparacoes = [(a1, a2) for i, a1 in enumerate(anos)
                for a2 in anos[i+1:]]

st.subheader("Comparação Estatística – Óbito (Log-rank)")
resultados_obito = []

for a1, a2 in comparacoes:
    d1 = df[df["ano_tx"] == a1]
    d2 = df[df["ano_tx"] == a2]

    if len(d1) > 0 and len(d2) > 0:
        res = logrank_test(
            d1["tempo_obito_anos"], d2["tempo_obito_anos"],
            event_observed_A=d1["evento_obito"],
            event_observed_B=d2["evento_obito"]
        )
        resultados_obito.append({
            "Comparação": f"{a1} x {a2}",
            "p-valor": round(res.p_value, 4)
        })

st.dataframe(pd.DataFrame(resultados_obito), use_container_width=True)

st.subheader("Comparação Estatística – Perda de Enxerto (Log-rank)")
resultados_pe = []

for a1, a2 in comparacoes:
    d1 = df[df["ano_tx"] == a1]
    d2 = df[df["ano_tx"] == a2]

    if len(d1) > 0 and len(d2) > 0:
        res = logrank_test(
            d1["tempo_pe_anos"], d2["tempo_pe_anos"],
            event_observed_A=d1["evento_pe"],
            event_observed_B=d2["evento_pe"]
        )
        resultados_pe.append({
            "Comparação": f"{a1} x {a2}",
            "p-valor": round(res.p_value, 4)
        })

st.dataframe(pd.DataFrame(resultados_pe), use_container_width=True)

# ==========================================================
# 🔹 SOBREVIDA EM 1, 2 E 5 ANOS
# ==========================================================
st.subheader("Sobrevida do Paciente em 1, 2 e 5 anos")

linhas = []
for ano in anos:
    dados = df[df["ano_tx"] == ano]
    if len(dados) > 0:
        kmf = KaplanMeierFitter()
        kmf.fit(dados["tempo_obito_anos"], dados["evento_obito"])
        linhas.append({
            "Ano": ano,
            "1 ano (%)": round(kmf.predict(1) * 100, 1),
            "2 anos (%)": round(kmf.predict(2) * 100, 1),
            "5 anos (%)": round(kmf.predict(5) * 100, 1),
        })

st.dataframe(pd.DataFrame(linhas), use_container_width=True)

    # ==========================================================
# 🔹 ANÁLISE GLOBAL
# ==========================================================
total_global = len(df)
total_obitos = df["evento_obito"].sum()
total_pe = df["evento_pe"].sum()

tabela_global = pd.DataFrame({
    "Total Transplantes": [total_global],
    "Óbitos": [total_obitos],
    "Taxa Óbito (%)": [round(total_obitos / total_global * 100, 1)],
    "Perda de Enxerto": [total_pe],
    "Taxa Perda Enxerto (%)": [round(total_pe / total_global * 100, 1)]
})

st.subheader("Análise Global da Coorte")
st.dataframe(tabela_global, use_container_width=True)

# ==========================================================
# 🔹 CURVAS KM (PROBABILIDADE E PORCENTAGEM)
# ==========================================================
kmf_global_obito = KaplanMeierFitter()
kmf_global_obito.fit(df["tempo_obito_anos"], df["evento_obito"], label="Global")

kmf_global_pe = KaplanMeierFitter()
kmf_global_pe.fit(df["tempo_pe_anos"], df["evento_pe"], label="Global")

cores = {ano: cor for ano, cor in zip(anos,
            ["tab:blue", "tab:orange", "tab:green", "tab:red"])}

def eixo_prob(ax, ylabel):
    ax.set_xlabel("Tempo após o transplante (anos)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.5, 1)
    ax.grid(True)

def eixo_percent(ax, ylabel):
    ax.set_xlabel("Tempo após o transplante (anos)")
    ax.set_ylabel(ylabel)
    ax.set_ylim(50, 100)
    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.grid(True)

col1, col2 = st.columns(2)

# ================= PACIENTE =================
with col1:

    # PROBABILIDADE
    st.subheader("Paciente – Probabilidade")
    fig1, ax1 = plt.subplots()
    kmf = KaplanMeierFitter()

    for ano in anos:
        dados = df[df["ano_tx"] == ano]
        kmf.fit(dados["tempo_obito_anos"], dados["evento_obito"], label=str(ano))
        kmf.plot(ax=ax1, ci_show=False, linewidth=2, color=cores.get(ano))

    kmf_global_obito.plot(ax=ax1, ci_show=False,
                            color="black", linestyle="--", linewidth=3)

    eixo_prob(ax1, "Probabilidade de Sobrevida")
    ax1.legend(title="Ano do Transplante")
    st.pyplot(fig1)

    # PORCENTAGEM
    st.subheader("Paciente – Porcentagem")
    fig2, ax2 = plt.subplots()

    for ano in anos:
        dados = df[df["ano_tx"] == ano]
        kmf.fit(dados["tempo_obito_anos"], dados["evento_obito"], label=str(ano))
        ax2.step(kmf.survival_function_.index,
                    kmf.survival_function_[str(ano)] * 100,
                    where="post",
                    linewidth=2,
                    color=cores.get(ano),
                    label=str(ano))

    ax2.step(kmf_global_obito.survival_function_.index,
                kmf_global_obito.survival_function_["Global"] * 100,
                where="post",
                linewidth=3,
                linestyle="--",
                color="black",
                label="Global")

    eixo_percent(ax2, "Sobrevida (%)")
    ax2.legend(title="Ano do Transplante")
    st.pyplot(fig2)

# ================= ENXERTO =================
with col2:

    # PROBABILIDADE
    st.subheader("Enxerto – Probabilidade")
    fig3, ax3 = plt.subplots()
    kmf = KaplanMeierFitter()

    for ano in anos:
        dados = df[df["ano_tx"] == ano]
        kmf.fit(dados["tempo_pe_anos"], dados["evento_pe"], label=str(ano))
        kmf.plot(ax=ax3, ci_show=False, linewidth=2, color=cores.get(ano))

    kmf_global_pe.plot(ax=ax3, ci_show=False,
                        color="black", linestyle="--", linewidth=3)

    eixo_prob(ax3, "Probabilidade de Sobrevida do Enxerto")
    ax3.legend(title="Ano do Transplante")
    st.pyplot(fig3)

    # PORCENTAGEM
    st.subheader("Enxerto – Porcentagem")
    fig4, ax4 = plt.subplots()

    for ano in anos:
        dados = df[df["ano_tx"] == ano]
        kmf.fit(dados["tempo_pe_anos"], dados["evento_pe"], label=str(ano))
        ax4.step(kmf.survival_function_.index,
                    kmf.survival_function_[str(ano)] * 100,
                    where="post",
                    linewidth=2,
                    color=cores.get(ano),
                    label=str(ano))

    ax4.step(kmf_global_pe.survival_function_.index,
                kmf_global_pe.survival_function_["Global"] * 100,
                where="post",
                linewidth=3,
                linestyle="--",
                color="black",
                label="Global")

    eixo_percent(ax4, "Sobrevida do Enxerto (%)")
    ax4.legend(title="Ano do Transplante")
    st.pyplot(fig4)

# ==========================================================
# ==========================================================
# 🔹 NOVAS ANÁLISES: SOBREVIDA POR TIPO DE DOADOR (SCD / ECD / KDPI > 85%)
# ==========================================================
# ==========================================================
#
# ⚠️ EDITAR AQUI — MAPEAMENTO DAS COLUNAS DO DATASET ⚠️
# ----------------------------------------------------------
# Ajuste os três nomes de coluna abaixo para os nomes REAIS que existem
# no seu arquivo de dados (df / CSV). O restante do código funciona
# automaticamente a partir dessas colunas.
#
#   COL_SCD  -> coluna indicadora de doador SCD (Standard Criteria Donor)
#               esperado: 1 = doador SCD, 0 = doador não-SCD
#   COL_ECD  -> coluna indicadora de doador ECD (Expanded Criteria Donor)
#               esperado: 1 = doador ECD, 0 = doador não-ECD
#   COL_KDPI -> coluna numérica do KDPI (0 a 100). A partir dela o
#               script cria automaticamente o grupo "KDPI > 85%".
#
# Se no seu CSV essas colunas tiverem outro nome (ex: "tipo_doador",
# "SCD_ECD", "kdpi_pct" etc.), basta trocar o valor das strings abaixo:

COL_TIPO_DOADOR = "classificacao_automatica"   # <-- EDITAR AQUI: nome da coluna única com texto "SCD"/"ECD"
COL_KDPI = "kdpi"                 # <-- EDITAR AQUI: nome da coluna numérica do KDPI (0-100)

# SCD e ECD estão na MESMA coluna, como texto ("SCD" ou "ECD").
# O código abaixo compara o texto (sem diferenciar maiúsc./minúsc. e
# ignorando espaços nas pontas) para criar as duas colunas binárias
# usadas nas análises. Se os textos no seu dataset forem diferentes
# de "SCD"/"ECD" (ex.: "Padrão"/"Expandido", "Standard"/"Expanded"),
# EDITE os dois valores comparados abaixo (.str.strip().str.upper() == "..."):

# Como SCD e ECD são as DUAS categorias de uma mesma coluna, uma única
# variável binária já compara os dois grupos (SCD x ECD). Não é preciso
# gerar duas seções separadas ("SCD x não-SCD" e "ECD x não-ECD"), pois
# nesse caso elas seriam espelhadas/duplicadas uma da outra.
if COL_TIPO_DOADOR in df.columns:
    _tipo_doador_norm = df[COL_TIPO_DOADOR].astype(str).str.strip().str.upper()
    df["_grupo_tipo_doador"] = pd.NA
    df.loc[_tipo_doador_norm == "SCD", "_grupo_tipo_doador"] = 1
    df.loc[_tipo_doador_norm == "ECD", "_grupo_tipo_doador"] = 0
    # Linhas com texto diferente de "SCD"/"ECD" (ex.: vazio, DCD, outro)
    # ficam como NA e não entram na análise, evitando classificação incorreta.
else:
    df["_grupo_tipo_doador"] = None

# Grupo KDPI > 85% criado automaticamente a partir da coluna COL_KDPI
if COL_KDPI in df.columns:
    df["_grupo_kdpi85"] = (pd.to_numeric(df[COL_KDPI], errors="coerce") > 85).astype("Int64")
else:
    df["_grupo_kdpi85"] = None


# ==========================================================
# 🔹 FUNÇÃO GENÉRICA DE ANÁLISE POR GRUPO (2 CATEGORIAS)
# ==========================================================
def analise_sobrevida_por_grupo(df, coluna_grupo, titulo_secao,
                                    rotulo_grupo1, rotulo_grupo0,
                                    cor_grupo1="tab:red", cor_grupo0="tab:blue"):
    """
    Replica, para uma variável de agrupamento binária (0/1),
    o mesmo padrão de análise já usado no restante do script
    (tabela de sobrevida em 1/2/5 anos, teste log-rank e curvas
    de Kaplan-Meier em probabilidade e porcentagem, para
    paciente e para enxerto).
    """

    st.header(titulo_secao)

    if coluna_grupo not in df.columns or df[coluna_grupo].dropna().empty:
        st.warning(
            f"Coluna '{coluna_grupo}' não encontrada ou vazia no dataset. "
            f"Verifique o mapeamento de colunas (COL_TIPO_DOADOR / COL_KDPI) "
            f"no início desta seção do script."
        )
        return

    dados_validos = df[df[coluna_grupo].isin([0, 1])].copy()

    grupo1 = dados_validos[dados_validos[coluna_grupo] == 1]
    grupo0 = dados_validos[dados_validos[coluna_grupo] == 0]

    if len(grupo1) == 0 or len(grupo0) == 0:
        st.warning(
            f"Não há pacientes suficientes em um dos grupos ({rotulo_grupo1} / "
            f"{rotulo_grupo0}) para gerar a análise."
        )
        return

    # ---------------- SOBREVIDA GLOBAL (referência, linha tracejada) ----------------
    # Calculada sobre todos os pacientes válidos da análise (grupo1 + grupo0 juntos),
    # do mesmo jeito que a curva "Global" já usada nas análises por ano do início do script.
    kmf_global_pac = KaplanMeierFitter()
    kmf_global_pac.fit(dados_validos["tempo_obito_anos"], dados_validos["evento_obito"], label="Global")

    kmf_global_enx = KaplanMeierFitter()
    kmf_global_enx.fit(dados_validos["tempo_pe_anos"], dados_validos["evento_pe"], label="Global")

    # ---------------- TABELA DE SOBREVIDA 1/2/5 ANOS ----------------
    st.subheader(f"Sobrevida do Paciente e do Enxerto em 1, 2 e 5 anos — {titulo_secao}")

    linhas_grupo = []
    for nome_grupo, dados_grupo in [(rotulo_grupo1, grupo1), (rotulo_grupo0, grupo0)]:
        kmf_pac = KaplanMeierFitter()
        kmf_pac.fit(dados_grupo["tempo_obito_anos"], dados_grupo["evento_obito"])

        kmf_enx = KaplanMeierFitter()
        kmf_enx.fit(dados_grupo["tempo_pe_anos"], dados_grupo["evento_pe"])

        linhas_grupo.append({
            "Grupo": nome_grupo,
            "N": len(dados_grupo),
            "Paciente 1 ano (%)": round(kmf_pac.predict(1) * 100, 1),
            "Paciente 2 anos (%)": round(kmf_pac.predict(2) * 100, 1),
            "Paciente 5 anos (%)": round(kmf_pac.predict(5) * 100, 1),
            "Enxerto 1 ano (%)": round(kmf_enx.predict(1) * 100, 1),
            "Enxerto 2 anos (%)": round(kmf_enx.predict(2) * 100, 1),
            "Enxerto 5 anos (%)": round(kmf_enx.predict(5) * 100, 1),
        })

    # Linha "Global" (grupo1 + grupo0 juntos) adicionada ao final da tabela
    linhas_grupo.append({
        "Grupo": "Global",
        "N": len(dados_validos),
        "Paciente 1 ano (%)": round(kmf_global_pac.predict(1) * 100, 1),
        "Paciente 2 anos (%)": round(kmf_global_pac.predict(2) * 100, 1),
        "Paciente 5 anos (%)": round(kmf_global_pac.predict(5) * 100, 1),
        "Enxerto 1 ano (%)": round(kmf_global_enx.predict(1) * 100, 1),
        "Enxerto 2 anos (%)": round(kmf_global_enx.predict(2) * 100, 1),
        "Enxerto 5 anos (%)": round(kmf_global_enx.predict(5) * 100, 1),
    })

    st.dataframe(pd.DataFrame(linhas_grupo), use_container_width=True)

    # ---------------- LOG-RANK (GRUPO1 x GRUPO0) ----------------
    st.subheader(f"Comparação Estatística (Log-rank) — {titulo_secao}")

    res_obito = logrank_test(
        grupo1["tempo_obito_anos"], grupo0["tempo_obito_anos"],
        event_observed_A=grupo1["evento_obito"],
        event_observed_B=grupo0["evento_obito"]
    )
    res_pe = logrank_test(
        grupo1["tempo_pe_anos"], grupo0["tempo_pe_anos"],
        event_observed_A=grupo1["evento_pe"],
        event_observed_B=grupo0["evento_pe"]
    )

    tabela_logrank = pd.DataFrame([
        {"Desfecho": "Óbito (Sobrevida do Paciente)",
            "Comparação": f"{rotulo_grupo1} x {rotulo_grupo0}",
            "p-valor": round(res_obito.p_value, 4)},
        {"Desfecho": "Perda de Enxerto (Sobrevida do Enxerto)",
            "Comparação": f"{rotulo_grupo1} x {rotulo_grupo0}",
            "p-valor": round(res_pe.p_value, 4)},
    ])
    st.dataframe(tabela_logrank, use_container_width=True)

    # ---------------- CURVAS KM ----------------
    col_a, col_b = st.columns(2)

    # ===== PACIENTE =====
    with col_a:
        st.subheader("Paciente – Probabilidade")
        fig_a1, ax_a1 = plt.subplots()
        kmf = KaplanMeierFitter()

        for nome_grupo, dados_grupo, cor in [
            (rotulo_grupo1, grupo1, cor_grupo1),
            (rotulo_grupo0, grupo0, cor_grupo0),
        ]:
            kmf.fit(dados_grupo["tempo_obito_anos"], dados_grupo["evento_obito"], label=nome_grupo)
            kmf.plot(ax=ax_a1, ci_show=False, linewidth=2, color=cor)

        kmf_global_pac.plot(ax=ax_a1, ci_show=False,
                                color="black", linestyle="--", linewidth=3)
        eixo_prob(ax_a1, "Probabilidade de Sobrevida")
        ax_a1.legend(title=titulo_secao)
        st.pyplot(fig_a1)

        st.subheader("Paciente – Porcentagem")
        fig_a2, ax_a2 = plt.subplots()

        for nome_grupo, dados_grupo, cor in [
            (rotulo_grupo1, grupo1, cor_grupo1),
            (rotulo_grupo0, grupo0, cor_grupo0),
        ]:
            kmf.fit(dados_grupo["tempo_obito_anos"], dados_grupo["evento_obito"], label=nome_grupo)
            ax_a2.step(kmf.survival_function_.index,
                        kmf.survival_function_[nome_grupo] * 100,
                        where="post",
                        linewidth=2,
                        color=cor,
                        label=nome_grupo)

        eixo_percent(ax_a2, "Sobrevida (%)")
        ax_a2.step(kmf_global_pac.survival_function_.index,
                    kmf_global_pac.survival_function_["Global"] * 100,
                    where="post",
                    linewidth=3,
                    linestyle="--",
                    color="black",
                    label="Global")
        ax_a2.legend(title=titulo_secao)
        st.pyplot(fig_a2)

    # ===== ENXERTO =====
    with col_b:
        st.subheader("Enxerto – Probabilidade")
        fig_b1, ax_b1 = plt.subplots()
        kmf = KaplanMeierFitter()

        for nome_grupo, dados_grupo, cor in [
            (rotulo_grupo1, grupo1, cor_grupo1),
            (rotulo_grupo0, grupo0, cor_grupo0),
        ]:
            kmf.fit(dados_grupo["tempo_pe_anos"], dados_grupo["evento_pe"], label=nome_grupo)
            kmf.plot(ax=ax_b1, ci_show=False, linewidth=2, color=cor)

        kmf_global_enx.plot(ax=ax_b1, ci_show=False,
                                color="black", linestyle="--", linewidth=3)
        eixo_prob(ax_b1, "Probabilidade de Sobrevida do Enxerto")
        ax_b1.legend(title=titulo_secao)
        st.pyplot(fig_b1)

        st.subheader("Enxerto – Porcentagem")
        fig_b2, ax_b2 = plt.subplots()

        for nome_grupo, dados_grupo, cor in [
            (rotulo_grupo1, grupo1, cor_grupo1),
            (rotulo_grupo0, grupo0, cor_grupo0),
        ]:
            kmf.fit(dados_grupo["tempo_pe_anos"], dados_grupo["evento_pe"], label=nome_grupo)
            ax_b2.step(kmf.survival_function_.index,
                        kmf.survival_function_[nome_grupo] * 100,
                        where="post",
                        linewidth=2,
                        color=cor,
                        label=nome_grupo)

        eixo_percent(ax_b2, "Sobrevida do Enxerto (%)")
        ax_b2.step(kmf_global_enx.survival_function_.index,
                    kmf_global_enx.survival_function_["Global"] * 100,
                    where="post",
                    linewidth=3,
                    linestyle="--",
                    color="black",
                    label="Global")
        ax_b2.legend(title=titulo_secao)
        st.pyplot(fig_b2)


# ==========================================================
# 🔹 1) SOBREVIDA DO PACIENTE / ENXERTO — DOADOR SCD x ECD
# ==========================================================
analise_sobrevida_por_grupo(
    df,
    coluna_grupo="_grupo_tipo_doador",
    titulo_secao="Sobrevida por Tipo de Doador (SCD x ECD)",
    rotulo_grupo1="Doador SCD",
    rotulo_grupo0="Doador ECD",
    cor_grupo1="tab:blue",
    cor_grupo0="tab:red",
)

# ==========================================================
# 🔹 2) SOBREVIDA DO PACIENTE / ENXERTO — KDPI > 85%
# ==========================================================
analise_sobrevida_por_grupo(
    df,
    coluna_grupo="_grupo_kdpi85",
    titulo_secao="Sobrevida por KDPI > 85%",
    rotulo_grupo1="KDPI > 85%",
    rotulo_grupo0="KDPI ≤ 85%",
    cor_grupo1="tab:red",
    cor_grupo0="tab:green",
)
