import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from datetime import datetime

# ================= CONFIGURAÇÃO =================
st.set_page_config(
    page_title="Kaplan-Meier – Transplante Renal",
    layout="wide"
)

st.title("Análise de Sobrevida – Transplante Renal")

uploaded_file = st.file_uploader(
    "Envie o arquivo CSV",
    type="csv"
)
if uploaded_file:

    # ================= LEITURA =================
    df = pd.read_csv(uploaded_file)
    # ================= LEITURA DO CSV =================
    # CAMINHO_CSV = "/mnt/data/TRANSPLANTERENALHOSP-SobrevidaPacienteEEn_DATA_2026-01-15_1147.csv"
    # df = pd.read_csv(CAMINHO_CSV)

    # ================= TRATAMENTO DE DATAS =================
    col_datas = ["data_tx", "data_obito", "data_pe"]
    for col in col_datas:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Ano do transplante
    df["ano_tx"] = df["data_tx"].dt.year

    # Data de censura (data atual)
    data_censura = pd.to_datetime(datetime.today().date())

    # ================= SOBREVIDA DO PACIENTE (ÓBITO) =================
    df["evento_obito"] = df["data_obito"].notna().astype(int)
    df["tempo_obito"] = (
        df["data_obito"].fillna(data_censura) - df["data_tx"]
    ).dt.days

    # ================= SOBREVIDA DO ENXERTO (PERDA) =================
    df["evento_pe"] = df["data_pe"].notna().astype(int)
    df["tempo_pe"] = (
        df["data_pe"].fillna(data_censura) - df["data_tx"]
    ).dt.days

    # ================= TABELA RESUMO POR ANO =================
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

    st.subheader("Resumo de Eventos por Ano do Transplante")
    st.dataframe(
        tabela_resumo,
        use_container_width=True
    )

    # ================= GRÁFICOS KAPLAN-MEIER =================
    anos = sorted(df["ano_tx"].dropna().unique())

    col1, col2 = st.columns(2)

    # ===== GRÁFICO 1 – ÓBITO =====
    with col1:
        st.subheader("Kaplan-Meier – Sobrevida do Paciente (Óbito)")

        fig1, ax1 = plt.subplots()
        kmf = KaplanMeierFitter()

        for ano in anos:
            dados = df[df["ano_tx"] == ano]

            kmf.fit(
                durations=dados["tempo_obito"],
                event_observed=dados["evento_obito"],
                label=str(ano)
            )
            kmf.plot(ax=ax1)

        ax1.set_xlabel("Dias após o transplante")
        ax1.set_ylabel("Probabilidade de Sobrevida")
        ax1.grid(True)

        st.pyplot(fig1)

    # ===== GRÁFICO 2 – PERDA DO ENXERTO =====
    with col2:
        st.subheader("Kaplan-Meier – Sobrevida do Enxerto")

        fig2, ax2 = plt.subplots()
        kmf = KaplanMeierFitter()

        for ano in anos:
            dados = df[df["ano_tx"] == ano]

            kmf.fit(
                durations=dados["tempo_pe"],
                event_observed=dados["evento_pe"],
                label=str(ano)
            )
            kmf.plot(ax=ax2)

        ax2.set_xlabel("Dias após o transplante")
        ax2.set_ylabel("Probabilidade de Sobrevida do Enxerto")
        ax2.grid(True)

        st.pyplot(fig2)