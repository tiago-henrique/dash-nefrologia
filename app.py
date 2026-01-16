import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, logrank_test
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

df["ano_tx"] = df["data_tx"].dt.year
data_censura = pd.to_datetime(datetime.today().date())

# ================= EVENTOS E TEMPOS =================
df["evento_obito"] = df["data_obito"].notna().astype(int)
df["tempo_obito"] = (df["data_obito"].fillna(data_censura) - df["data_tx"]).dt.days

df["evento_pe"] = df["data_pe"].notna().astype(int)
df["tempo_pe"] = (df["data_pe"].fillna(data_censura) - df["data_tx"]).dt.days

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

# ================= SOBREVIDA PACIENTE AGRUPADA (1, 3, 5 ANOS) =================
linhas_paciente = []

for ano in anos:
    dados = df[df["ano_tx"] == ano]
    if len(dados) == 0:
        continue

    kmf.fit(dados["tempo_obito"], dados["evento_obito"])

    linha = {"Ano TX": int(ano)}

    for label, dias in tempos_sobrevida.items():
        surv, ic_inf, ic_sup = extrair_sobrevida_e_ic(kmf, dias)
        linha[f"{label} (%)"] = f"{surv} ({ic_inf}–{ic_sup})"

    linhas_paciente.append(linha)

st.subheader("Sobrevida do Paciente por Ano de Transplante")
st.dataframe(pd.DataFrame(linhas_paciente), use_container_width=True)

# ================= SOBREVIDA ENXERTO AGRUPADA (1, 3, 5 ANOS) =================
linhas_enxerto = []

for ano in anos:
    dados = df[df["ano_tx"] == ano]
    if len(dados) == 0:
        continue

    kmf.fit(dados["tempo_pe"], dados["evento_pe"])

    linha = {"Ano TX": int(ano)}

    for label, dias in tempos_sobrevida.items():
        surv, ic_inf, ic_sup = extrair_sobrevida_e_ic(kmf, dias)
        linha[f"{label} (%)"] = f"{surv} ({ic_inf}–{ic_sup})"

    linhas_enxerto.append(linha)

st.subheader("Sobrevida do Enxerto por Ano de Transplante")
st.dataframe(pd.DataFrame(linhas_enxerto), use_container_width=True)

# ================= SOBREVIDA GLOBAL =================
linhas_global = []

kmf_global.fit(df["tempo_obito"], df["evento_obito"])
for label, dias in tempos_sobrevida.items():
    surv, ic_inf, ic_sup = extrair_sobrevida_e_ic(kmf_global, dias)
    linhas_global.append({
        "Grupo": "Global",
        "Tipo": "Paciente",
        "Tempo": label,
        "Sobrevida (%)": surv,
        "IC 95% Inf (%)": ic_inf,
        "IC 95% Sup (%)": ic_sup
    })

kmf_global.fit(df["tempo_pe"], df["evento_pe"])
for label, dias in tempos_sobrevida.items():
    surv, ic_inf, ic_sup = extrair_sobrevida_e_ic(kmf_global, dias)
    linhas_global.append({
        "Grupo": "Global",
        "Tipo": "Enxerto",
        "Tempo": label,
        "Sobrevida (%)": surv,
        "IC 95% Inf (%)": ic_inf,
        "IC 95% Sup (%)": ic_sup
    })

st.subheader("Sobrevida Global (IC 95%)")
st.dataframe(pd.DataFrame(linhas_global), use_container_width=True)

# ================= LOG-RANK GLOBAL =================
st.subheader("Teste de Log-Rank Global")

lr_paciente = multivariate_logrank_test(
    df["tempo_obito"], df["ano_tx"], df["evento_obito"]
)
st.markdown(f"**Paciente:** p = **{lr_paciente.p_value:.4f}**")

lr_enxerto = multivariate_logrank_test(
    df["tempo_pe"], df["ano_tx"], df["evento_pe"]
)
st.markdown(f"**Enxerto:** p = **{lr_enxerto.p_value:.4f}**")

# ================= LOG-RANK PAREADO =================
st.subheader("Comparação Pareada entre Anos (Log-Rank)")

def comparar_anos(df, a1, a2, tempo, evento):
    d1 = df[df["ano_tx"] == a1]
    d2 = df[df["ano_tx"] == a2]
    if len(d1) == 0 or len(d2) == 0:
        return None
    return logrank_test(
        d1[tempo], d2[tempo],
        d1[evento], d2[evento]
    ).p_value

comparacoes = [(2022, 2023), (2022, 2024), (2022, 2025), (2023, 2024), (2023, 2025), (2024, 2025)]
linhas_comp = []

for a1, a2 in comparacoes:
    p_obito = comparar_anos(df, a1, a2, "tempo_obito", "evento_obito")
    p_pe = comparar_anos(df, a1, a2, "tempo_pe", "evento_pe")

    linhas_comp.append({
        "Comparação": f"{a1} x {a2}",
        "p-valor Óbito": None if p_obito is None else round(p_obito, 4),
        "p-valor Enxerto": None if p_pe is None else round(p_pe, 4)
    })

st.dataframe(pd.DataFrame(linhas_comp), use_container_width=True)

# ================= GRÁFICOS =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Kaplan–Meier – Paciente")
    fig, ax = plt.subplots()

    for ano in anos:
        dados = df[df["ano_tx"] == ano]
        kmf.fit(dados["tempo_obito"], dados["evento_obito"], label=str(int(ano)))
        kmf.plot(ax=ax, ci_show=False)

    kmf_global.fit(df["tempo_obito"], df["evento_obito"], label="Global")
    kmf_global.plot(ax=ax, ci_show=False, linewidth=3, linestyle="--")

    ax.set_xlabel("Dias após o transplante")
    ax.set_ylabel("Probabilidade de Sobrevida")
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.subheader("Kaplan–Meier – Enxerto")
    fig, ax = plt.subplots()

    for ano in anos:
        dados = df[df["ano_tx"] == ano]
        kmf.fit(dados["tempo_pe"], dados["evento_pe"], label=str(int(ano)))
        kmf.plot(ax=ax, ci_show=False)

    kmf_global.fit(df["tempo_pe"], df["evento_pe"], label="Global")
    kmf_global.plot(ax=ax, ci_show=False, linewidth=3, linestyle="--")

    ax.set_xlabel("Dias após o transplante")
    ax.set_ylabel("Probabilidade de Sobrevida do Enxerto")
    ax.grid(True)
    st.pyplot(fig)
