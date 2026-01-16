import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from datetime import datetime

# ================= CONFIGURAÇÃO =================
st.set_page_config(
    page_title="Kaplan-Meier – Transplante Renal",
    layout="wide"
)

st.title("Análise de Sobrevida – Transplante Renal")

# ================= LEITURA =================
uploaded_file = "https://imunogenetica.famerp.br/dash/nefrologia/indicadores.csv"

if uploaded_file:

    df = pd.read_csv(uploaded_file)

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

    # ================= RESUMO POR ANO =================
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

    # ================= CORES =================
    cores = {
        2022: "tab:blue",
        2023: "tab:orange",
        2024: "tab:green",
        2025: "tab:red"
    }

    anos = sorted(df["ano_tx"].dropna().unique())

    # ================= SOBREVIDA GLOBAL =================
    kmf_global_obito = KaplanMeierFitter()
    kmf_global_obito.fit(df["tempo_obito"], df["evento_obito"], label="Global")

    kmf_global_pe = KaplanMeierFitter()
    kmf_global_pe.fit(df["tempo_pe"], df["evento_pe"], label="Global")

     # ================= COMPARAÇÕES DEFINIDAS =================
    comparacoes = [
        (2022, 2023),
        (2022, 2024),
        (2022, 2025),
        (2023, 2024),
        (2023, 2025),
        (2024, 2025)
    ]

    # ================= LOG-RANK – ÓBITO =================
    st.subheader("Comparação Estatística entre Anos – Óbito (Log-rank)")

    resultados_obito = []

    for a1, a2 in comparacoes:
        d1 = df[df["ano_tx"] == a1]
        d2 = df[df["ano_tx"] == a2]

        if len(d1) > 0 and len(d2) > 0:
            res = logrank_test(
                d1["tempo_obito"], d2["tempo_obito"],
                event_observed_A=d1["evento_obito"],
                event_observed_B=d2["evento_obito"]
            )
            resultados_obito.append({
                "Comparação": f"{a1} x {a2}",
                "p-valor": round(res.p_value, 4)
            })

    st.dataframe(pd.DataFrame(resultados_obito), use_container_width=True)

    # ================= LOG-RANK – PERDA DE ENXERTO =================
    st.subheader("Comparação Estatística entre Anos – Perda de Enxerto (Log-rank)")

    resultados_pe = []

    for a1, a2 in comparacoes:
        d1 = df[df["ano_tx"] == a1]
        d2 = df[df["ano_tx"] == a2]

        if len(d1) > 0 and len(d2) > 0:
            res = logrank_test(
                d1["tempo_pe"], d2["tempo_pe"],
                event_observed_A=d1["evento_pe"],
                event_observed_B=d2["evento_pe"]
            )
            resultados_pe.append({
                "Comparação": f"{a1} x {a2}",
                "p-valor": round(res.p_value, 4)
            })

    st.dataframe(pd.DataFrame(resultados_pe), use_container_width=True)

    st.caption(
        "Teste de Log-rank: p < 0,05 indica diferença estatisticamente significativa entre as curvas."
    )

    # ================= SOBREVIDA 1, 3 E 5 ANOS =================
    st.subheader("Sobrevida do Paciente em 1, 3 e 5 anos")

    linhas = []

    for ano in [2022, 2023, 2024, 2025]:
        dados = df[df["ano_tx"] == ano]
        if len(dados) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(dados["tempo_obito"], dados["evento_obito"])
            linhas.append({
                "Ano": ano,
                "1 ano (%)": round(kmf.predict(365) * 100, 1),
                "3 anos (%)": round(kmf.predict(1095) * 100, 1),
                "5 anos (%)": round(kmf.predict(1825) * 100, 1),
            })

    st.dataframe(pd.DataFrame(linhas), use_container_width=True)


    col1, col2 = st.columns(2)

    # ================= KM – ÓBITO =================
    with col1:
        st.subheader("Kaplan-Meier – Sobrevida do Paciente")

        fig1, ax1 = plt.subplots()
        kmf = KaplanMeierFitter()

        for ano in anos:
            dados = df[df["ano_tx"] == ano]
            kmf.fit(dados["tempo_obito"], dados["evento_obito"], label=str(ano))
            kmf.plot(
                ax=ax1,
                color=cores.get(ano, "gray"),
                linewidth=2,
                ci_show=False
            )

        kmf_global_obito.plot(
            ax=ax1,
            color="black",
            #linestyle="--",
            linewidth=3,
            ci_show=False
        )

        ax1.set_xlabel("Dias após o transplante")
        ax1.set_ylabel("Probabilidade de Sobrevida")
        ax1.grid(True)
        st.pyplot(fig1)

    # ================= KM – ENXERTO =================
    with col2:
        st.subheader("Kaplan-Meier – Sobrevida do Enxerto")

        fig2, ax2 = plt.subplots()
        kmf = KaplanMeierFitter()

        for ano in anos:
            dados = df[df["ano_tx"] == ano]
            kmf.fit(dados["tempo_pe"], dados["evento_pe"], label=str(ano))
            kmf.plot(
                ax=ax2,
                color=cores.get(ano, "gray"),
                linewidth=2,
                ci_show=False
            )

        kmf_global_pe.plot(
            ax=ax2,
            color="black",
            #linestyle="--",
            linewidth=3,
            ci_show=False
        )

        ax2.set_xlabel("Dias após o transplante")
        ax2.set_ylabel("Probabilidade de Sobrevida do Enxerto")
        ax2.grid(True)
        st.pyplot(fig2)
