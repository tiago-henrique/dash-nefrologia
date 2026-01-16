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
uploaded_file = "https://imunogenetica.famerp.br/nefrologia/indicadores.csv";

uploaded_file = st.file_uploader(
    "Envie o arquivo CSV",
    type="csv"
)

if uploaded_file:

    # ================= LEITURA =================
    df = pd.read_csv(uploaded_file)

    # ================= TRATAMENTO DE DATAS =================
    col_datas = ["data_tx", "data_obito", "data_pe"]
    for col in col_datas:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["ano_tx"] = df["data_tx"].dt.year
    data_censura = pd.to_datetime(datetime.today().date())

    # ================= EVENTOS E TEMPOS =================
    df["evento_obito"] = df["data_obito"].notna().astype(int)
    df["tempo_obito"] = (
        df["data_obito"].fillna(data_censura) - df["data_tx"]
    ).dt.days

    df["evento_pe"] = df["data_pe"].notna().astype(int)
    df["tempo_pe"] = (
        df["data_pe"].fillna(data_censura) - df["data_tx"]
    ).dt.days

    # ================= RESUMO =================
    tabela_resumo = (
        df.groupby("ano_tx")
        .agg(
            total=("ano_tx", "count"),
            obitos=("evento_obito", "sum"),
            perda_enxerto=("evento_pe", "sum")
        )
        .reset_index()
        .sort_values("ano_tx")
    )

    st.subheader("Resumo de Eventos por Ano")
    st.dataframe(tabela_resumo, use_container_width=True)

    # ================= TEMPOS DE AVALIAÇÃO =================
    tempos_sobrevida = {
        "1 ano": 365,
        "3 anos": 3 * 365,
        "5 anos": 5 * 365
    }

    anos = sorted(df["ano_tx"].dropna().unique())
    kmf = KaplanMeierFitter()

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

    # ================= TABELA – PACIENTE =================
    linhas_obito = []

    for ano in anos:
        dados = df[df["ano_tx"] == ano]
        kmf.fit(dados["tempo_obito"], dados["evento_obito"])

        for label, dias in tempos_sobrevida.items():
            surv, ic_inf, ic_sup = extrair_sobrevida_e_ic(kmf, dias)
            linhas_obito.append({
                "Ano TX": int(ano),
                "Tempo": label,
                "Sobrevida (%)": surv,
                "IC 95% Inf (%)": ic_inf,
                "IC 95% Sup (%)": ic_sup
            })

    st.subheader("Sobrevida do Paciente (IC 95%)")
    st.dataframe(pd.DataFrame(linhas_obito), use_container_width=True)

    # ================= TABELA – ENXERTO =================
    linhas_pe = []

    for ano in anos:
        dados = df[df["ano_tx"] == ano]
        kmf.fit(dados["tempo_pe"], dados["evento_pe"])

        for label, dias in tempos_sobrevida.items():
            surv, ic_inf, ic_sup = extrair_sobrevida_e_ic(kmf, dias)
            linhas_pe.append({
                "Ano TX": int(ano),
                "Tempo": label,
                "Sobrevida Enxerto (%)": surv,
                "IC 95% Inf (%)": ic_inf,
                "IC 95% Sup (%)": ic_sup
            })

    st.subheader("Sobrevida do Enxerto (IC 95%)")
    st.dataframe(pd.DataFrame(linhas_pe), use_container_width=True)

    # ================= LOG-RANK =================
    st.subheader("Teste de Log-Rank (Comparação entre Anos)")

    # Paciente
    lr_paciente = multivariate_logrank_test(
        df["tempo_obito"],
        df["ano_tx"],
        df["evento_obito"]
    )

    st.markdown("**Paciente (Óbito):**")
    st.write(f"p-valor = **{lr_paciente.p_value:.4f}**")

    if lr_paciente.p_value < 0.05:
        st.error("Diferença estatisticamente significativa entre os anos (p < 0,05).")
    else:
        st.success("Não há evidência de diferença estatística entre os anos (p ≥ 0,05).")

    # Enxerto
    lr_enxerto = multivariate_logrank_test(
        df["tempo_pe"],
        df["ano_tx"],
        df["evento_pe"]
    )

    st.markdown("**Enxerto:**")
    st.write(f"p-valor = **{lr_enxerto.p_value:.4f}**")

    if lr_enxerto.p_value < 0.05:
        st.error("Diferença estatisticamente significativa entre os anos (p < 0,05).")
    else:
        st.success("Não há evidência de diferença estatística entre os anos (p ≥ 0,05).")

    # ================= GRÁFICOS =================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Kaplan–Meier – Paciente (IC 95%)")
        fig, ax = plt.subplots()

        for ano in anos:
            dados = df[df["ano_tx"] == ano]
            kmf.fit(dados["tempo_obito"], dados["evento_obito"], label=str(int(ano)))
            kmf.plot(ci_show=True, ax=ax)

        ax.set_xlabel("Dias após o transplante")
        ax.set_ylabel("Probabilidade de Sobrevida")
        ax.grid(True)
        st.pyplot(fig)

    with col2:
        st.subheader("Kaplan–Meier – Enxerto (IC 95%)")
        fig, ax = plt.subplots()

        for ano in anos:
            dados = df[df["ano_tx"] == ano]
            kmf.fit(dados["tempo_pe"], dados["evento_pe"], label=str(int(ano)))
            kmf.plot(ci_show=True, ax=ax)

        ax.set_xlabel("Dias após o transplante")
        ax.set_ylabel("Probabilidade de Sobrevida do Enxerto")
        ax.grid(True)
        st.pyplot(fig)
