import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from datetime import datetime

# ================= CONFIGURAÇÃO =================
st.set_page_config(
    page_title="Kaplan-Meier – Transplante Renal",
    layout="wide"
)

st.title("Análise de Sobrevida – Transplante Renal")

uploaded_file = "https://imunogenetica.famerp.br/dash/nefrologia/indicadores.csv"

# ================= LEITURA =================
df = pd.read_csv(uploaded_file)

# ================= TRATAMENTO DE DATAS =================
col_datas = ["data_tx", "data_obito", "data_pe"]
for col in col_datas:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# ================= ANOS =================
df["ano_tx"] = df["data_tx"].dt.year
df["ano_obito"] = df["data_obito"].dt.year
df["ano_pe"] = df["data_pe"].dt.year

data_censura = pd.to_datetime(datetime.today().date())

# ================= EVENTOS E TEMPOS =================
df["evento_obito"] = df["data_obito"].notna().astype(int)
df["tempo_obito"] = (df["data_obito"].fillna(data_censura) - df["data_tx"]).dt.days

df["evento_pe"] = df["data_pe"].notna().astype(int)
df["tempo_pe"] = (df["data_pe"].fillna(data_censura) - df["data_tx"]).dt.days

# ================= ANOS DE ANÁLISE =================
anos_analise = [2022, 2023, 2024, 2025]

# ================= CORES (CIÊNCIA / PUBLICAÇÃO) =================
cores_anos = {
    2022: "#1b9e77",
    2023: "#d95f02",
    2024: "#7570b3",
    2025: "#e7298a",
}

# ================= TEMPOS DE AVALIAÇÃO =================
tempos_sobrevida = {
    "1 ano": 365,
    "3 anos": 3 * 365,
    "5 anos": 5 * 365
}

kmf = KaplanMeierFitter()
kmf_global = KaplanMeierFitter()

# ================= FUNÇÃO IC =================
def extrair_sobrevida_e_ic(kmf, dias):
    surv = kmf.survival_function_at_times(dias).values[0]
    ci = kmf.confidence_interval_
    ci_tempo = ci.loc[ci.index <= dias].iloc[-1]
    return (
        round(surv * 100, 2),
        round(ci_tempo.iloc[0] * 100, 2),
        round(ci_tempo.iloc[1] * 100, 2)
    )

# ======================================================
# TABELA – TOTAL DE ÓBITOS E PERDA DE ENXERTO POR ANO
# ======================================================
tabela_eventos = (
    pd.DataFrame({
        "Óbitos": df.groupby("ano_obito").size(),
        "Perda de Enxerto": df.groupby("ano_pe").size()
    })
    .fillna(0)
    .astype(int)
    .reset_index()
    .rename(columns={"index": "Ano"})
)

tabela_eventos = tabela_eventos[tabela_eventos["Ano"].isin(anos_analise)]

st.subheader("Total de Óbitos e Perda de Enxerto por Ano")
st.dataframe(tabela_eventos, use_container_width=True)

# ======================================================
# SOBREVIDA PACIENTE POR ANO DO ÓBITO
# ======================================================
linhas_paciente = []

for ano in anos_analise:
    dados = df[(df["ano_obito"] == ano) | (df["evento_obito"] == 0)]
    if len(dados) == 0:
        continue

    kmf.fit(dados["tempo_obito"], dados["evento_obito"])
    linha = {"Ano do Óbito": ano}

    for label, dias in tempos_sobrevida.items():
        surv, ic_inf, ic_sup = extrair_sobrevida_e_ic(kmf, dias)
        linha[f"{label} (%)"] = f"{surv} ({ic_inf}–{ic_sup})"

    linhas_paciente.append(linha)

st.subheader("Sobrevida do Paciente por Ano do Óbito")
st.dataframe(pd.DataFrame(linhas_paciente), use_container_width=True)

# ======================================================
# SOBREVIDA ENXERTO POR ANO DA PERDA
# ======================================================
linhas_enxerto = []

for ano in anos_analise:
    dados = df[(df["ano_pe"] == ano) | (df["evento_pe"] == 0)]
    if len(dados) == 0:
        continue

    kmf.fit(dados["tempo_pe"], dados["evento_pe"])
    linha = {"Ano da Perda do Enxerto": ano}

    for label, dias in tempos_sobrevida.items():
        surv, ic_inf, ic_sup = extrair_sobrevida_e_ic(kmf, dias)
        linha[f"{label} (%)"] = f"{surv} ({ic_inf}–{ic_sup})"

    linhas_enxerto.append(linha)

st.subheader("Sobrevida do Enxerto por Ano da Perda")
st.dataframe(pd.DataFrame(linhas_enxerto), use_container_width=True)

# ======================================================
# LOG-RANK GLOBAL
# ======================================================
st.subheader("Teste de Log-Rank Global")

lr_paciente = multivariate_logrank_test(
    df["tempo_obito"], df["ano_obito"], df["evento_obito"]
)
st.markdown(f"**Paciente (óbito por ano do evento):** p = **{lr_paciente.p_value:.4f}**")

lr_enxerto = multivariate_logrank_test(
    df["tempo_pe"], df["ano_pe"], df["evento_pe"]
)
st.markdown(f"**Enxerto (perda por ano do evento):** p = **{lr_enxerto.p_value:.4f}**")

# ======================================================
# GRÁFICOS
# ======================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Kaplan–Meier – Paciente (Ano do Óbito)")
    fig, ax = plt.subplots()

    for ano in anos_analise:
        dados = df[(df["ano_obito"] == ano) | (df["evento_obito"] == 0)]
        kmf.fit(dados["tempo_obito"], dados["evento_obito"], label=str(ano))
        kmf.plot(
            ax=ax,
            ci_show=False,
            color=cores_anos.get(ano, "gray"),
            linewidth=2
        )

    kmf_global.fit(df["tempo_obito"], df["evento_obito"], label="Global")
    kmf_global.plot(
        ax=ax,
        ci_show=False,
        color="black",
        #linestyle="--",
        linewidth=3
    )

    ax.set_xlabel("Dias após o transplante")
    ax.set_ylabel("Probabilidade de Sobrevida")
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.subheader("Kaplan–Meier – Enxerto (Ano da Perda)")
    fig, ax = plt.subplots()

    for ano in anos_analise:
        dados = df[(df["ano_pe"] == ano) | (df["evento_pe"] == 0)]
        kmf.fit(dados["tempo_pe"], dados["evento_pe"], label=str(ano))
        kmf.plot(
            ax=ax,
            ci_show=False,
            color=cores_anos.get(ano, "gray"),
            linewidth=2
        )

    kmf_global.fit(df["tempo_pe"], df["evento_pe"], label="Global")
    kmf_global.plot(
        ax=ax,
        ci_show=False,
        color="black",
        #linestyle="--",
        linewidth=3
    )

    ax.set_xlabel("Dias após o transplante")
    ax.set_ylabel("Probabilidade de Sobrevida do Enxerto")
    ax.grid(True)
    st.pyplot(fig)
