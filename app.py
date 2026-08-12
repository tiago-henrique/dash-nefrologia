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

# Cores geradas dinamicamente a partir de um colormap, para suportar
# automaticamente qualquer quantidade de anos (inclusive novos anos
# que forem inseridos futuramente no dataset).
def gerar_mapa_cores(lista_valores, colormap_nome="tab10"):
    cmap = plt.get_cmap(colormap_nome if len(lista_valores) <= 10 else "tab20")
    return {valor: cmap(i % cmap.N) for i, valor in enumerate(lista_valores)}

cores = gerar_mapa_cores(anos)

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
#   COL_TIPO_DOADOR -> coluna de texto única indicando "SCD" ou "ECD"
#   COL_KDPI        -> coluna numérica do KDPI (0 a 100). A partir dela o
#                      script cria automaticamente o grupo "KDPI > 85%".
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
    df["_tipo_doador_norm"] = df[COL_TIPO_DOADOR].astype(str).str.strip().str.upper()

    df["_grupo_tipo_doador"] = pd.NA
    df.loc[df["_tipo_doador_norm"] == "SCD", "_grupo_tipo_doador"] = 1
    df.loc[df["_tipo_doador_norm"] == "ECD", "_grupo_tipo_doador"] = 0
    # Linhas com texto diferente de "SCD"/"ECD" (ex.: vazio, DCD, outro)
    # ficam como NA e não entram na análise, evitando classificação incorreta.
else:
    df["_tipo_doador_norm"] = None
    df["_grupo_tipo_doador"] = None

# Grupo KDPI > 85% criado automaticamente a partir da coluna COL_KDPI
if COL_KDPI in df.columns:
    df["_grupo_kdpi85"] = (pd.to_numeric(df[COL_KDPI], errors="coerce") > 85).astype("Int64")
else:
    df["_grupo_kdpi85"] = None


# ==========================================================
# ==========================================================
# 🔹 3) SOBREVIDA POR ANO DO TRANSPLANTE, DENTRO DE CADA TIPO
#        DE DOADOR (SCD E ECD SEPARADOS)
# ==========================================================
# ==========================================================
# Diferente da seção acima (que compara SCD x ECD dentro de cada ano),
# aqui a lógica é invertida: dentro do grupo SCD, comparam-se os anos
# entre si; e o mesmo é feito, separadamente, dentro do grupo ECD.
#
# A quantidade de curvas (anos) é detectada automaticamente a partir
# dos dados presentes no dataset (df["ano_tx"].unique()), então quando
# novos anos/registros forem inseridos na base, os gráficos passam a
# incluí-los sem precisar editar o código.

def analise_por_ano_do_subgrupo(dados_tipo, titulo_secao, nome_subgrupo="este grupo"):
    """
    Recebe um dataframe já filtrado para um subgrupo (ex.: só SCD, só ECD,
    ou só KDPI > 85%) e gera:
    - tabela de sobrevida (paciente e enxerto) em 1/2/5 anos, por ano do transplante
    - curvas de Kaplan-Meier (probabilidade e porcentagem) por ano do transplante,
      com a curva "Global" (todos os anos, dentro do subgrupo) sobreposta

    As cores e a quantidade de curvas se ajustam automaticamente à
    quantidade de anos presentes nos dados filtrados — não é necessário
    editar nada quando novos anos forem incluídos no dataset.
    """

    st.header(titulo_secao)

    if dados_tipo.empty:
        st.warning(f"Não há pacientes em '{nome_subgrupo}' no dataset.")
        return

    dados_tipo = dados_tipo.copy()
    anos_tipo = sorted(dados_tipo["ano_tx"].dropna().unique())

    if len(anos_tipo) == 0:
        st.warning(f"Não há anos de transplante válidos para '{nome_subgrupo}'.")
        return

    cores_tipo = gerar_mapa_cores(anos_tipo)

    # ---------------- CURVAS GLOBAIS DO TIPO DE DOADOR (todos os anos juntos) ----------------
    kmf_global_pac = KaplanMeierFitter()
    kmf_global_pac.fit(dados_tipo["tempo_obito_anos"], dados_tipo["evento_obito"], label="Global")

    kmf_global_enx = KaplanMeierFitter()
    kmf_global_enx.fit(dados_tipo["tempo_pe_anos"], dados_tipo["evento_pe"], label="Global")

    # ---------------- TABELA DE SOBREVIDA 1/2/5 ANOS, POR ANO DO TRANSPLANTE ----------------
    st.subheader(f"Sobrevida do Paciente e do Enxerto em 1, 2 e 5 anos — {titulo_secao}")

    linhas_ano = []
    for ano in anos_tipo:
        dados_ano = dados_tipo[dados_tipo["ano_tx"] == ano]
        if len(dados_ano) == 0:
            continue

        kmf_pac = KaplanMeierFitter()
        kmf_pac.fit(dados_ano["tempo_obito_anos"], dados_ano["evento_obito"])

        kmf_enx = KaplanMeierFitter()
        kmf_enx.fit(dados_ano["tempo_pe_anos"], dados_ano["evento_pe"])

        linhas_ano.append({
            "Ano": int(ano),
            "N": len(dados_ano),
            "Paciente 1 ano (%)": round(kmf_pac.predict(1) * 100, 1),
            "Paciente 2 anos (%)": round(kmf_pac.predict(2) * 100, 1),
            "Paciente 5 anos (%)": round(kmf_pac.predict(5) * 100, 1),
            "Enxerto 1 ano (%)": round(kmf_enx.predict(1) * 100, 1),
            "Enxerto 2 anos (%)": round(kmf_enx.predict(2) * 100, 1),
            "Enxerto 5 anos (%)": round(kmf_enx.predict(5) * 100, 1),
        })

    linhas_ano.append({
        "Ano": "Global",
        "N": len(dados_tipo),
        "Paciente 1 ano (%)": round(kmf_global_pac.predict(1) * 100, 1),
        "Paciente 2 anos (%)": round(kmf_global_pac.predict(2) * 100, 1),
        "Paciente 5 anos (%)": round(kmf_global_pac.predict(5) * 100, 1),
        "Enxerto 1 ano (%)": round(kmf_global_enx.predict(1) * 100, 1),
        "Enxerto 2 anos (%)": round(kmf_global_enx.predict(2) * 100, 1),
        "Enxerto 5 anos (%)": round(kmf_global_enx.predict(5) * 100, 1),
    })

    st.dataframe(pd.DataFrame(linhas_ano), use_container_width=True)

    # ---------------- LOG-RANK ENTRE ANOS (dentro do tipo de doador) ----------------
    comparacoes_tipo = [(a1, a2) for i, a1 in enumerate(anos_tipo) for a2 in anos_tipo[i + 1:]]

    if comparacoes_tipo:
        st.subheader(f"Comparação Estatística entre Anos (Log-rank) — {titulo_secao}")

        resultados_obito_tipo = []
        resultados_pe_tipo = []

        for a1, a2 in comparacoes_tipo:
            d1 = dados_tipo[dados_tipo["ano_tx"] == a1]
            d2 = dados_tipo[dados_tipo["ano_tx"] == a2]

            if len(d1) > 0 and len(d2) > 0:
                res_obito = logrank_test(
                    d1["tempo_obito_anos"], d2["tempo_obito_anos"],
                    event_observed_A=d1["evento_obito"],
                    event_observed_B=d2["evento_obito"]
                )
                resultados_obito_tipo.append({
                    "Comparação": f"{int(a1)} x {int(a2)}",
                    "p-valor": round(res_obito.p_value, 4)
                })

                res_pe = logrank_test(
                    d1["tempo_pe_anos"], d2["tempo_pe_anos"],
                    event_observed_A=d1["evento_pe"],
                    event_observed_B=d2["evento_pe"]
                )
                resultados_pe_tipo.append({
                    "Comparação": f"{int(a1)} x {int(a2)}",
                    "p-valor": round(res_pe.p_value, 4)
                })

        st.write("Óbito (Sobrevida do Paciente)")
        st.dataframe(pd.DataFrame(resultados_obito_tipo), use_container_width=True)

        st.write("Perda de Enxerto (Sobrevida do Enxerto)")
        st.dataframe(pd.DataFrame(resultados_pe_tipo), use_container_width=True)

    # ---------------- CURVAS KM POR ANO ----------------
    col_a, col_b = st.columns(2)

    # ===== PACIENTE =====
    with col_a:
        st.subheader("Paciente – Probabilidade")
        fig1, ax1 = plt.subplots()
        kmf = KaplanMeierFitter()

        # ---- Curvas individuais por ano do transplante (COMENTADO) ----
        # Mantida apenas a curva "Global" do subgrupo (SCD / ECD / KDPI > 85%).
        # for ano in anos_tipo:
        #     dados_ano = dados_tipo[dados_tipo["ano_tx"] == ano]
        #     if len(dados_ano) == 0:
        #         continue
        #     kmf.fit(dados_ano["tempo_obito_anos"], dados_ano["evento_obito"], label=str(int(ano)))
        #     kmf.plot(ax=ax1, ci_show=False, linewidth=2, color=cores_tipo.get(ano))

        kmf_global_pac.plot(ax=ax1, ci_show=False, color="black", linestyle="--", linewidth=3)
        eixo_prob(ax1, "Probabilidade de Sobrevida")
        ax1.legend(title="Ano do Transplante")
        st.pyplot(fig1)

        st.subheader("Paciente – Porcentagem")
        fig2, ax2 = plt.subplots()

        # ---- Curvas individuais por ano do transplante (COMENTADO) ----
        # Mantida apenas a curva "Global" do subgrupo (SCD / ECD / KDPI > 85%).
        # for ano in anos_tipo:
        #     dados_ano = dados_tipo[dados_tipo["ano_tx"] == ano]
        #     if len(dados_ano) == 0:
        #         continue
        #     kmf.fit(dados_ano["tempo_obito_anos"], dados_ano["evento_obito"], label=str(int(ano)))
        #     ax2.step(kmf.survival_function_.index,
        #              kmf.survival_function_[str(int(ano))] * 100,
        #              where="post", linewidth=2, color=cores_tipo.get(ano), label=str(int(ano)))

        ax2.step(kmf_global_pac.survival_function_.index,
                 kmf_global_pac.survival_function_["Global"] * 100,
                 where="post", linewidth=3, linestyle="--", color="black", label="Global")
        eixo_percent(ax2, "Sobrevida (%)")
        ax2.legend(title="Ano do Transplante")
        st.pyplot(fig2)

    # ===== ENXERTO =====
    with col_b:
        st.subheader("Enxerto – Probabilidade")
        fig3, ax3 = plt.subplots()
        kmf = KaplanMeierFitter()

        for ano in anos_tipo:
            dados_ano = dados_tipo[dados_tipo["ano_tx"] == ano]
            if len(dados_ano) == 0:
                continue
            kmf.fit(dados_ano["tempo_pe_anos"], dados_ano["evento_pe"], label=str(int(ano)))
            kmf.plot(ax=ax3, ci_show=False, linewidth=2, color=cores_tipo.get(ano))

        kmf_global_enx.plot(ax=ax3, ci_show=False, color="black", linestyle="--", linewidth=3)
        eixo_prob(ax3, "Probabilidade de Sobrevida do Enxerto")
        ax3.legend(title="Ano do Transplante")
        st.pyplot(fig3)

        st.subheader("Enxerto – Porcentagem")
        fig4, ax4 = plt.subplots()

        #for ano in anos_tipo:
        #    dados_ano = dados_tipo[dados_tipo["ano_tx"] == ano]
        #    if len(dados_ano) == 0:
        #        continue
        #    kmf.fit(dados_ano["tempo_pe_anos"], dados_ano["evento_pe"], label=str(int(ano)))
        #    ax4.step(kmf.survival_function_.index,
        #             kmf.survival_function_[str(int(ano))] * 100,
        #             where="post", linewidth=2, color=cores_tipo.get(ano), label=str(int(ano)))

        ax4.step(kmf_global_enx.survival_function_.index,
                 kmf_global_enx.survival_function_["Global"] * 100,
                 where="post", linewidth=3, linestyle="--", color="black", label="Global")
        eixo_percent(ax4, "Sobrevida do Enxerto (%)")
        ax4.legend(title="Ano do Transplante")
        st.pyplot(fig4)


# ---- Chamadas: um bloco de gráficos por ano para SCD, ECD e KDPI > 85% ----
if "_tipo_doador_norm" in df.columns:
    analise_por_ano_do_subgrupo(
        df[df["_tipo_doador_norm"] == "SCD"],
        titulo_secao="Sobrevida por Ano do Transplante — Doadores SCD",
        nome_subgrupo="Doadores SCD",
    )

    analise_por_ano_do_subgrupo(
        df[df["_tipo_doador_norm"] == "ECD"],
        titulo_secao="Sobrevida por Ano do Transplante — Doadores ECD",
        nome_subgrupo="Doadores ECD",
    )
else:
    st.warning(
        "Coluna de tipo de doador não encontrada. "
        "Verifique o mapeamento COL_TIPO_DOADOR no início da seção anterior do script."
    )

if "_grupo_kdpi85" in df.columns:
    analise_por_ano_do_subgrupo(
        df[df["_grupo_kdpi85"] == 1],
        titulo_secao="Sobrevida por Ano do Transplante — KDPI > 85%",
        nome_subgrupo="KDPI > 85%",
    )
else:
    st.warning(
        "Coluna de KDPI não encontrada. "
        "Verifique o mapeamento COL_KDPI no início da seção anterior do script."
    )
