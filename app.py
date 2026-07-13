import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from PIL import Image
import os
import json

# ---------------------------------------------
# Configuração da página
# ---------------------------------------------
# Tenta carregar ícone, mas não quebra se não existir
try:
    icon = Image.open("violencia-icon.png")
    st.set_page_config(
        page_title="VulneraMapa — Violência Doméstica",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
except Exception:
    st.set_page_config(
        page_title="VulneraMapa — Violência Doméstica",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ---------------------------------------------
# Configuração de acesso ao backend
# Credenciais vêm de st.secrets (Streamlit Cloud → App settings → Secrets),
# com fallback para variáveis de ambiente na execução local.
# ---------------------------------------------
def _cfg(key: str, default: str = "") -> str:
    """Lê uma chave de st.secrets; se ausente, cai para variável de ambiente."""
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


API_URL = _cfg(
    "API_URL", os.getenv("BACKEND_URL", "https://truth-muppet-granddad.ngrok-free.dev")
).rstrip("/")
API_USER = _cfg("API_USER", "admin")
API_PASSWORD = _cfg("API_PASSWORD", "")
# Só envia Basic auth quando há senha configurada (o backend pode rodar aberto localmente).
AUTH = (API_USER, API_PASSWORD) if API_PASSWORD else None

PREDICT_ENDPOINT = f"{API_URL}/predict"
COMPARE_ENDPOINT = f"{API_URL}/predict/compare"
WHATIF_ENDPOINT = f"{API_URL}/predict/what-if"
BATCH_ENDPOINT = f"{API_URL}/predict/batch"
IMPORTANCE_ENDPOINT = f"{API_URL}/model/importance"
FEATURES_ENDPOINT = f"{API_URL}/model/features"
CLUSTERS_ENDPOINT = f"{API_URL}/clusters"
MUNICIPIOS_ENDPOINT = f"{API_URL}/municipios"

MODELOS = ["logistica", "random_forest", "xgboost", "lightgbm"]
MODELO_LABEL = {
    "logistica": "Regressão logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

# ---------------------------------------------
# CSS
# ---------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp { background-color: #f7f5f2; color: #2e2a26; }

section[data-testid="stSidebar"] {
    background-color: #edeae5;
    border-right: 1px solid #d8d3cc;
}

h1, h2, h3, h4 { color: #2e2a26 !important; }

.main-title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #2e2a26;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}
.main-subtitle {
    font-size: 0.8rem;
    color: #8a8278;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

.bloco-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #3d7a8a;
    margin-bottom: 0.4rem;
    margin-top: 1.2rem;
}

.result-placeholder {
    background: #ffffff;
    border: 1px dashed #c8c2ba;
    border-radius: 10px;
    padding: 2.2rem 1.5rem;
    text-align: center;
    color: #9a9288;
    font-size: 0.85rem;
    line-height: 1.7;
}

/* Card de resultado */
.result-card {
    background: #ffffff;
    border: 1px solid #d8d3cc;
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
}

/* Barra de probabilidade */
.prob-bar-wrapper {
    background: #edeae5;
    border-radius: 8px;
    height: 14px;
    width: 100%;
    margin: 0.6rem 0 0.3rem 0;
    overflow: hidden;
}
.prob-bar-fill {
    height: 14px;
    border-radius: 8px;
    transition: width 0.4s ease;
}

/* SHAP placeholder */
.shap-placeholder {
    background: #f0f4f5;
    border: 1px dashed #3d7a8a44;
    border-radius: 10px;
    padding: 1.6rem 1rem;
    text-align: center;
    color: #3d7a8a;
    font-size: 0.82rem;
    line-height: 1.7;
    margin-top: 0.8rem;
}

.nota-metodologica {
    background: #eef5f7;
    border-left: 3px solid #3d7a8a;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.78rem;
    color: #3a5f68;
    margin-top: 1.4rem;
    line-height: 1.6;
}

.stButton > button {
    background: #3d7a8a !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.5rem !important;
    width: 100%;
    margin-top: 0.5rem;
    transition: background 0.2s !important;
}
.stButton > button:hover { background: #2e6070 !important; }

hr { border-color: #d8d3cc !important; }

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border-color: #c8c2ba !important;
    color: #2e2a26 !important;
}

.fonte-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #3a3a5a;
    margin-top: 0.3rem;
}

/* Risco alto / médio / baixo */
.risco-alto   { color: #c0392b; font-weight: 700; }
.risco-medio  { color: #d35400; font-weight: 700; }
.risco-baixo  { color: #27ae60; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------
# Constantes / enumerações
# ---------------------------------------------
UFS = sorted([
    "AC","AL","AM","AP","BA","CE","DF","ES","GO",
    "MA","MG","MS","MT","PA","PB","PE","PI","PR",
    "RJ","RN","RO","RR","RS","SC","SE","SP","TO",
])

UF_PARA_REGIAO = {
    "AC":"Norte","AL":"Nordeste","AM":"Norte","AP":"Norte","BA":"Nordeste",
    "CE":"Nordeste","DF":"Centro-Oeste","ES":"Sudeste","GO":"Centro-Oeste",
    "MA":"Nordeste","MG":"Sudeste","MS":"Centro-Oeste","MT":"Centro-Oeste",
    "PA":"Norte","PB":"Nordeste","PE":"Nordeste","PI":"Nordeste","PR":"Sul",
    "RJ":"Sudeste","RN":"Nordeste","RO":"Norte","RR":"Norte","RS":"Sul",
    "SC":"Sul","SE":"Nordeste","SP":"Sudeste","TO":"Norte",
}

# UF <-> código IBGE de 2 dígitos (o parquet de clusters usa o código, não a sigla)
UF_TO_IBGE = {
    "RO":"11","AC":"12","AM":"13","RR":"14","PA":"15","AP":"16","TO":"17",
    "MA":"21","PI":"22","CE":"23","RN":"24","PB":"25","PE":"26","AL":"27",
    "SE":"28","BA":"29","MG":"31","ES":"32","RJ":"33","SP":"35","PR":"41",
    "SC":"42","RS":"43","MS":"50","MT":"51","GO":"52","DF":"53",
}
IBGE_TO_UF = {v: k for k, v in UF_TO_IBGE.items()}

IDHM_POR_REGIAO = {
    "Norte":"Médio (0.60–0.69)",
    "Nordeste":"Baixo (0.50–0.59)",
    "Centro-Oeste":"Alto (0.70–0.79)",
    "Sudeste":"Alto (0.70–0.79)",
    "Sul":"Alto (0.70–0.79)",
}

# Mapeamento faixa IDHM → valor numérico central
IDHM_VALOR = {
    "Muito baixo (< 0.50)": 0.45,
    "Baixo (0.50–0.59)":    0.55,
    "Médio (0.60–0.69)":    0.65,
    "Alto (0.70–0.79)":     0.75,
    "Muito alto (≥ 0.80)":  0.85,
}
IDHM_OPTIONS = list(IDHM_VALOR.keys())

FAIXAS_ETARIAS = ["menor_18","18_24","25_34","35_44","45_54","55_64","65_mais"]
FAIXAS_DISPLAY = {
    "menor_18":"Menor de 18 anos","18_24":"18–24 anos","25_34":"25–34 anos",
    "35_44":"35–44 anos","45_54":"45–54 anos","55_64":"55–64 anos","65_mais":"65 anos ou mais",
}
# Idade representativa por faixa (para enviar ao backend)
FAIXA_PARA_IDADE = {
    "menor_18": 16, "18_24": 21, "25_34": 29,
    "35_44": 39, "45_54": 49, "55_64": 59, "65_mais": 70,
}

RACAS = ["Branca","Preta","Amarela","Parda","Indigena","Ignorado"]

ESCOLARIDADES = [
    "Sem instrucao","Fundamental incompleto","Fundamental completo",
    "Medio incompleto","Medio completo","Superior incompleto",
    "Superior completo","Ignorado",
]

SIT_CONJUGAL = [
    "Solteira","Casada/Uniao estavel","Viuva","Separada","Nao se aplica","Ignorado",
]

RELACAO_AGRESSOR = [
    "Conjuge","Ex_conjuge","Namorado","Ex_namorado","Pai","Filho",
    "Irmao","Conhecido","Desconhecido","Outros","Ignorado",
]

# Relações consideradas "parceiro íntimo"
PARCEIRO_INTIMO_SET = {"Conjuge","Ex_conjuge","Namorado","Ex_namorado"}

TIPOS_VIOLENCIA = {
    "viol_fisica":      "Física",
    "viol_psico":       "Psicológica",
    "viol_sexual":      "Sexual",
    "viol_financeira":  "Financeira",
    "viol_negligencia": "Negligência",
}

# Variáveis numéricas que podem ser varridas na aba What-if (início, fim, rótulo)
SWEEP_CONFIG = {
    "idade":                  (18.0, 80.0, "Idade (anos)"),
    "idhm":                   (0.40, 0.90, "IDHM municipal"),
    "renda_pc":               (0.0, 5000.0, "Renda per capita (R$)"),
    "prop_pobreza":           (0.0, 1.0, "Proporção em pobreza (0–1)"),
    "indice_gini":            (0.30, 0.70, "Índice de Gini"),
    "indice_vulnerabilidade": (0.0, 1.0, "Índice de vulnerabilidade"),
    "n_tipos_violencia":      (0.0, 5.0, "Nº de tipos de violência"),
    "n_encaminhamentos":      (0.0, 5.0, "Nº de encaminhamentos"),
}

REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]

# Variáveis categóricas varridas na aba What-if (feature → lista de valores conhecidos)
WHATIF_CAT = {
    "relacao_agressor": RELACAO_AGRESSOR,
    "sit_conjugal":     SIT_CONJUGAL,
    "raca":             RACAS,
    "escolaridade":     ESCOLARIDADES,
    "nome_regiao":      REGIOES,
    "uf":               UFS,
}

# Variáveis binárias varridas na aba What-if (sempre 0/1)
WHATIF_BIN = [
    "parceiro_intimo", "viol_fisica", "viol_psico", "viol_sexual",
    "viol_financeira", "viol_negligencia", "enc_delegacia", "enc_deam",
    "enc_creas", "enc_casa_mulher", "cons_suicidio", "cons_saude_mental",
]

# Rótulos amigáveis de toda variável varrível (numéricas reaproveitam SWEEP_CONFIG)
WHATIF_LABEL = {
    **{f: cfg[2] for f, cfg in SWEEP_CONFIG.items()},
    "relacao_agressor": "Relação com o agressor",
    "sit_conjugal":     "Situação conjugal",
    "raca":             "Raça / cor",
    "escolaridade":     "Escolaridade",
    "nome_regiao":      "Macroregião",
    "uf":               "UF de residência",
    "parceiro_intimo":  "Parceiro íntimo (0/1)",
    "viol_fisica":      "Violência física (0/1)",
    "viol_psico":       "Violência psicológica (0/1)",
    "viol_sexual":      "Violência sexual (0/1)",
    "viol_financeira":  "Violência financeira (0/1)",
    "viol_negligencia": "Negligência (0/1)",
    "enc_delegacia":    "Encaminhamento: Delegacia (0/1)",
    "enc_deam":         "Encaminhamento: DEAM (0/1)",
    "enc_creas":        "Encaminhamento: CREAS (0/1)",
    "enc_casa_mulher":  "Encaminhamento: Casa da Mulher (0/1)",
    "cons_suicidio":    "Suspeita de suicídio (0/1)",
    "cons_saude_mental": "Consulta saúde mental (0/1)",
}

# Ordem exibida no seletor de variável da aba What-if
WHATIF_FEATURES = list(SWEEP_CONFIG.keys()) + list(WHATIF_CAT.keys()) + WHATIF_BIN

# Campos aceitos por /predict - usados para filtrar colunas de CSV no lote,
# evitando 422 causado por colunas extras (o backend usa extra="forbid").
PREDICT_FIELDS = {
    "ano_num", "cons_saude_mental", "cons_suicidio", "enc_casa_mulher", "enc_creas",
    "enc_deam", "enc_delegacia", "escolaridade", "idade", "idhm", "indice_gini",
    "indice_vulnerabilidade", "n_encaminhamentos", "n_tipos_violencia", "nome_regiao",
    "parceiro_intimo", "prop_pobreza", "raca", "relacao_agressor", "renda_pc",
    "sit_conjugal", "uf", "viol_fisica", "viol_psico", "viol_sexual", "viol_financeira",
    "viol_negligencia", "faixa_etaria", "raca_enc", "escolaridade_enc", "sit_conjugal_enc",
    "relacao_agressor_enc", "uf_enc", "faixa_etaria_enc", "nome_regiao_enc",
    "features", "model",
}

# ---------------------------------------------
# Níveis de risco - alinhados aos cutoffs do backend (0.33 / 0.66)
# ---------------------------------------------
NIVEL_CORES = {"baixo": "#27ae60", "moderado": "#d35400", "alto": "#c0392b"}
NIVEL_LABEL = {"baixo": "Baixo", "moderado": "Moderado", "alto": "Alto"}
NIVEL_CLASSE = {"baixo": "risco-baixo", "moderado": "risco-medio", "alto": "risco-alto"}


def cor_por_prob(prob: float):
    """Deriva o nível a partir da probabilidade (cutoffs do backend: 0.33 / 0.66)."""
    if prob >= 0.66:
        nivel = "alto"
    elif prob >= 0.33:
        nivel = "moderado"
    else:
        nivel = "baixo"
    return NIVEL_CORES[nivel], NIVEL_LABEL[nivel], NIVEL_CLASSE[nivel], nivel


def cor_por_nivel(risk_level):
    """Cor/rótulo/classe a partir do campo risk_level retornado pelo backend."""
    nivel = (risk_level or "").lower()
    if nivel not in NIVEL_CORES:
        return "#8a8278", (risk_level or "—"), "", nivel
    return NIVEL_CORES[nivel], NIVEL_LABEL[nivel], NIVEL_CLASSE[nivel], nivel


# Cluster municipal (0/1/2) -> nível de risco, para reusar as cores/rótulos de risco.
CLUSTER_NIVEL = {0: "baixo", 1: "moderado", 2: "alto"}


def cor_cluster(cluster_id, cluster_label: str = "") -> str:
    """Cor do nível de risco de um cluster municipal (pelo id 0/1/2 ou pelo rótulo)."""
    nivel = CLUSTER_NIVEL.get(cluster_id)
    if nivel is None:
        lab = (cluster_label or "").casefold()
        nivel = "baixo" if "baixo" in lab else "alto" if "alto" in lab else "moderado"
    return NIVEL_CORES.get(nivel, "#8a8278")


def faixa_from_idhm(idhm: float) -> str:
    """Mapeia um IDHM numérico para o rótulo de faixa mais próximo (IDHM_OPTIONS)."""
    if idhm < 0.50:
        return "Muito baixo (< 0.50)"
    if idhm < 0.60:
        return "Baixo (0.50–0.59)"
    if idhm < 0.70:
        return "Médio (0.60–0.69)"
    if idhm < 0.80:
        return "Alto (0.70–0.79)"
    return "Muito alto (≥ 0.80)"


# ---------------------------------------------
# Chamadas HTTP autenticadas ao backend
# ---------------------------------------------
def _post(url: str, payload: dict, timeout: int = 30) -> dict:
    """POST autenticado; traduz os códigos de erro documentados da API."""
    resp = requests.post(url, json=payload, auth=AUTH, timeout=timeout)
    if resp.status_code == 401:
        raise PermissionError(
            "Credenciais ausentes ou incorretas (401). Verifique API_USER e "
            "API_PASSWORD nos secrets do Streamlit."
        )
    if resp.status_code == 422:
        raise ValueError(f"Dados inválidos para o modelo (422): {resp.text}")
    if resp.status_code == 500:
        raise RuntimeError(f"Falha ao executar o modelo/scaler no backend (500): {resp.text}")
    if resp.status_code == 503:
        raise RuntimeError(f"Artefato do modelo indisponível no backend (503): {resp.text}")
    resp.raise_for_status()
    return resp.json()


def chamar_predict(payload: dict) -> dict:
    """POST /predict — score de um caso pelo modelo escolhido."""
    return _post(PREDICT_ENDPOINT, payload)


def chamar_compare(payload: dict) -> dict:
    """POST /predict/compare — mesmo caso avaliado pelos 4 modelos + consenso."""
    return _post(COMPARE_ENDPOINT, payload)


def chamar_what_if(payload: dict) -> dict:
    """POST /predict/what-if — varre uma feature e devolve a curva de risco."""
    return _post(WHATIF_ENDPOINT, payload)


def chamar_batch(payload: dict) -> dict:
    """POST /predict/batch — pontua vários casos (até 500) de uma vez."""
    return _post(BATCH_ENDPOINT, payload, timeout=120)


@st.cache_data(ttl=3600, show_spinner=False)
def chamar_model_features() -> dict:
    """GET /model/features — metadados do modelo (cacheado por 1h)."""
    resp = requests.get(FEATURES_ENDPOINT, auth=AUTH, timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600, show_spinner=False)
def chamar_importance() -> dict:
    """GET /model/importance — importância global por modelo (cacheado por 1h)."""
    resp = requests.get(IMPORTANCE_ENDPOINT, auth=AUTH, timeout=30)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600, show_spinner=False)
def chamar_clusters() -> dict:
    """GET /clusters — tipologia dos clusters municipais (cacheado por 1h)."""
    resp = requests.get(CLUSTERS_ENDPOINT, auth=AUTH, timeout=30)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600, show_spinner=False)
def chamar_municipio(cod: str):
    """GET /municipios/{cod} — cluster e perfil de um município; None quando 404."""
    resp = requests.get(f"{MUNICIPIOS_ENDPOINT}/{cod}", auth=AUTH, timeout=15)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def modelos_disponiveis() -> list:
    """Modelos com artefato carregado no servidor; cai para todos em caso de erro."""
    try:
        av = chamar_model_features().get("available_models", {}) or {}
        disp = [m for m in MODELOS if av.get(m, {}).get("available")]
        return disp or MODELOS
    except Exception:
        return MODELOS


def mostrar_erro(e: Exception) -> None:
    """Renderiza uma mensagem de erro amigável para as exceções esperadas."""
    if isinstance(e, PermissionError):
        st.error(str(e))
    elif isinstance(e, requests.exceptions.ConnectionError):
        st.error(
            "Não foi possível conectar ao backend. "
            f"Verifique se o servidor está em execução em `{API_URL}`."
        )
    elif isinstance(e, requests.exceptions.Timeout):
        st.error("O backend demorou demais para responder. Tente novamente.")
    elif isinstance(e, (ValueError, RuntimeError)):
        st.error(str(e))
    elif isinstance(e, requests.exceptions.HTTPError):
        st.error(f"Erro HTTP {e.response.status_code}: {e.response.text}")
    elif isinstance(e, (KeyError, TypeError)):
        st.error(f"Resposta do backend em formato inesperado (campo ausente: {e}).")
    else:
        st.error(f"Erro inesperado: {e}")


def render_prediction(resultado: dict, payload: dict) -> None:
    """Renderiza o resultado de um POST /predict (card, distribuição, SHAP, detalhes)."""
    prob_f = float(resultado["probability_reincidencia"])
    prediction_label = resultado.get("prediction_label", "—")
    probs_dict = resultado.get("probabilities", {}) or {}
    shap_values = resultado.get("shap_values", {}) or {}
    shap_base_value = resultado.get("shap_base_value")
    scaler_used = resultado.get("scaler_used")
    input_mode = resultado.get("input_mode")
    modelo_resp = resultado.get("model", "—")
    risk_level = resultado.get("risk_level")
    preproc_warn = resultado.get("preprocessing_warning")

    # Nível de risco: usa o risk_level do backend; se ausente, deriva da probabilidade.
    if risk_level:
        cor, nivel, cls, _ = cor_por_nivel(risk_level)
    else:
        cor, nivel, cls, _ = cor_por_prob(prob_f)
    pct = round(prob_f * 100, 1)

    st.markdown(
        f"""
        <div class="result-card">
            <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;
                        text-transform:uppercase;color:#8a8278;margin-bottom:0.4rem;">
                Probabilidade estimada de reincidência
            </div>
            <div style="font-size:2.6rem;font-weight:700;color:{cor};
                        line-height:1;margin-bottom:0.2rem;">
                {pct}%
            </div>
            <div class="prob-bar-wrapper">
                <div class="prob-bar-fill"
                     style="width:{pct}%;background:{cor};"></div>
            </div>
            <div style="font-size:0.82rem;color:#5b5451;margin-top:0.4rem;">
                Nível de risco: <span class="{cls}">{nivel}</span>
                &nbsp;·&nbsp; Classe prevista: <b>{prediction_label}</b>
                &nbsp;·&nbsp; Modelo: <b>{MODELO_LABEL.get(modelo_resp, modelo_resp)}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if preproc_warn:
        st.caption(f"Aviso: {preproc_warn}")

    # Contexto do cluster municipal, quando um cod_mun6 conhecido foi enviado
    mc = resultado.get("municipal_cluster")
    if mc:
        clabel = mc.get("cluster_label", "—")
        cor_mc = cor_cluster(mc.get("cluster"), clabel)
        tx = mc.get("taxa_reincidencia")
        tx_txt = f"{tx * 100:.1f}%" if tx is not None else "—"
        st.markdown(
            f"<div style='margin-top:0.6rem;font-size:0.82rem;color:#5b5451;'>"
            f"Município <b>{mc.get('cod_mun6', '—')}</b> &nbsp;·&nbsp; "
            f"Cluster municipal: <b style='color:{cor_mc}'>{clabel}</b> &nbsp;·&nbsp; "
            f"reincidência média local <b>{tx_txt}</b></div>",
            unsafe_allow_html=True,
        )
        auto = mc.get("autofilled_fields") or []
        if auto:
            st.caption(
                "Campos preenchidos pelo backend a partir do IBGE: " + ", ".join(auto) + "."
            )

    # Distribuição por classe
    if probs_dict:
        st.markdown("<div style='margin-top:0.9rem;font-weight:600'>Distribuição por classe</div>", unsafe_allow_html=True)
        for label, v in probs_dict.items():
            try:
                pf = float(v)
            except Exception:
                continue
            p_pct = round(pf * 100, 1)
            color, _, _, _ = cor_por_prob(pf)
            st.markdown(
                f"""
                <div style='margin-top:0.55rem;'>
                  <div style='font-size:0.82rem;margin-bottom:0.15rem;color:#444'>{label}</div>
                  <div class='prob-bar-wrapper'>
                    <div class='prob-bar-fill' style='width:{p_pct}%;background:{color};'></div>
                  </div>
                  <div style='font-size:0.78rem;color:#666;margin-top:0.2rem'>{p_pct}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Gráfico SHAP (contribuição de cada feature, em log-odds)
    if shap_values:
        st.markdown("<div style='margin-top:1.4rem;font-weight:600'>Fatores de maior peso na predição (SHAP)</div>", unsafe_allow_html=True)

        itens = list(shap_values.items())[:15]
        itens = itens[::-1]  # maior impacto no topo do gráfico horizontal
        nomes = [k for k, _ in itens]
        valores = [float(v) for _, v in itens]
        cores = ["#c0392b" if v > 0 else "#27ae60" for v in valores]

        fig = go.Figure(
            go.Bar(
                x=valores,
                y=nomes,
                orientation="h",
                marker_color=cores,
                text=[f"{v:+.3f}" for v in valores],
                textposition="outside",
            )
        )
        fig.update_layout(
            height=max(320, 26 * len(nomes)),
            margin=dict(l=10, r=30, t=10, b=10),
            xaxis_title="Contribuição em log-odds (→ Reincidência | ← Primeiro episódio)",
            yaxis_title=None,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Sora, sans-serif", size=12, color="#2e2a26"),
        )
        fig.add_vline(x=0, line_width=1, line_color="#8a8278")
        st.plotly_chart(fig, width="stretch")

        if shap_base_value is not None:
            st.markdown(
                f"<small style='color:#8a8278'>Valor base do modelo (intercepto): "
                f"{shap_base_value:.4f} log-odds. Positivo empurra para <b>Reincidência</b>, "
                f"negativo empurra para <b>Primeiro episódio</b>.</small>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="shap-placeholder">O backend não retornou <code>shap_values</code> para esta predição.</div>',
            unsafe_allow_html=True,
        )
        if modelo_resp == "random_forest":
            st.caption(
                "O Random Forest não possui suporte nativo a SHAP. Use a Regressão "
                "logística, o XGBoost ou o LightGBM para ver as contribuições por feature."
            )

    # Detalhes técnicos da resposta
    with st.expander("Detalhes técnicos da resposta do modelo"):
        st.markdown(
            f"- **Modelo:** `{modelo_resp}`\n"
            f"- **Modo de entrada:** `{input_mode}`\n"
            f"- **Scaler aplicado:** `{scaler_used}`\n"
            f"- **Predição (classe numérica):** `{resultado.get('prediction')}`"
        )
        st.json(resultado)

    st.markdown("---")
    st.markdown("**Dados enviados ao modelo:**")
    st.dataframe(
        pd.DataFrame(payload.items(), columns=["Campo", "Valor"]),
        width="stretch",
        hide_index=True,
        height=480,
    )


def render_comparison(comp: dict) -> None:
    """Renderiza a resposta de POST /predict/compare (tabela + barras + consenso)."""
    results = comp.get("results", {}) or {}
    consensus = comp.get("consensus", {}) or {}

    # Tabela por modelo
    rows = []
    for m in MODELOS:
        r = results.get(m)
        if not r:
            continue
        prob = r.get("probability_reincidencia")
        nivel = (r.get("risk_level") or "").lower()
        rows.append({
            "Modelo": MODELO_LABEL.get(m, m),
            "Disponível": "sim" if r.get("available") else "não",
            "Prob. reincidência": f"{prob * 100:.1f}%" if prob is not None else "—",
            "Nível": NIVEL_LABEL.get(nivel, r.get("risk_level") or "—"),
            "Classe prevista": r.get("prediction_label") or (r.get("error") or "—"),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Barras horizontais de probabilidade por modelo (colorido pelo nível)
    disp = [
        (m, results[m])
        for m in MODELOS
        if results.get(m) and results[m].get("probability_reincidencia") is not None
    ]
    if disp:
        nomes = [MODELO_LABEL.get(m, m) for m, _ in disp][::-1]
        valores = [r["probability_reincidencia"] * 100 for _, r in disp][::-1]
        cores = [
            NIVEL_CORES.get((r.get("risk_level") or "").lower(), "#8a8278")
            for _, r in disp
        ][::-1]
        figc = go.Figure(
            go.Bar(
                x=valores, y=nomes, orientation="h", marker_color=cores,
                text=[f"{v:.1f}%" for v in valores], textposition="outside",
            )
        )
        figc.update_layout(
            height=max(240, 62 * len(nomes)),
            margin=dict(l=10, r=44, t=10, b=10),
            xaxis_title="Probabilidade de reincidência (%)",
            xaxis_range=[0, 100],
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Sora, sans-serif", size=12, color="#2e2a26"),
        )
        st.plotly_chart(figc, width="stretch")

    # Card de consenso
    if consensus:
        c_nivel = (consensus.get("risk_level") or "").lower()
        c_cor = NIVEL_CORES.get(c_nivel, "#8a8278")
        c_lbl = NIVEL_LABEL.get(c_nivel, consensus.get("risk_level") or "—")
        mean_p = consensus.get("mean_probability_reincidencia")
        mean_txt = f"{mean_p * 100:.1f}%" if mean_p is not None else "—"
        votes = consensus.get("votes", {}) or {}
        votes_txt = " · ".join(f"{k}: {v}" for k, v in votes.items()) or "—"
        unanime = "sim" if consensus.get("unanimous") else "não"
        st.markdown(
            f"""
            <div class="result-card" style="margin-top:0.4rem;">
                <div style="font-size:0.72rem;font-weight:600;letter-spacing:0.1em;
                            text-transform:uppercase;color:#8a8278;margin-bottom:0.4rem;">
                    Consenso ({consensus.get('models_compared', 0)} modelos)
                </div>
                <div style="font-size:1.5rem;font-weight:700;color:{c_cor};line-height:1.1;">
                    {consensus.get('prediction_label', '—')}
                </div>
                <div style="font-size:0.85rem;color:#5b5451;margin-top:0.5rem;">
                    Prob. média de reincidência: <b>{mean_txt}</b>
                    &nbsp;·&nbsp; Nível: <span class="{NIVEL_CLASSE.get(c_nivel, '')}">{c_lbl}</span><br>
                    Votos — {votes_txt}<br>
                    Unânime: <b>{unanime}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("Resposta completa da comparação"):
        st.json(comp)


def whatif_kind(feat: str) -> str:
    """Classifica a variável a varrer: numérica, categórica ou binária."""
    if feat in SWEEP_CONFIG:
        return "num"
    if feat in WHATIF_CAT:
        return "cat"
    return "bin"


def render_whatif(feature: str, kind: str, wi: dict) -> None:
    """Renderiza a resposta de POST /predict/what-if (linha p/ numérica, barras p/ categórica)."""
    points = wi.get("points", []) or []
    feat_label = WHATIF_LABEL.get(feature, feature)
    base_value = wi.get("base_value")
    base_prob = wi.get("base_probability")
    base_risk = wi.get("base_risk_level")

    ys = [(p.get("probability_reincidencia") or 0) * 100 for p in points]
    figw = go.Figure()
    # Faixas de risco (mesmos cutoffs do backend)
    figw.add_hrect(y0=0, y1=33, fillcolor=NIVEL_CORES["baixo"], opacity=0.08, line_width=0)
    figw.add_hrect(y0=33, y1=66, fillcolor=NIVEL_CORES["moderado"], opacity=0.08, line_width=0)
    figw.add_hrect(y0=66, y1=100, fillcolor=NIVEL_CORES["alto"], opacity=0.08, line_width=0)

    if kind == "num":
        xs = [p["value"] for p in points]
        figw.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers", name="Risco",
                line=dict(color="#3d7a8a", width=2.5), marker=dict(size=6),
            )
        )
        if base_value is not None and base_prob is not None:
            figw.add_trace(
                go.Scatter(
                    x=[base_value], y=[base_prob * 100], mode="markers", name="Perfil base",
                    marker=dict(size=13, color="#2e2a26", symbol="diamond"),
                )
            )
        figw.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
    else:
        if kind == "bin":
            xs = ["Não (0)" if str(p["value"]) in ("0", "0.0", "False") else "Sim (1)" for p in points]
        else:
            xs = [str(p["value"]) for p in points]
        cores = [NIVEL_CORES.get((p.get("risk_level") or "").lower(), "#8a8278") for p in points]
        figw.add_trace(
            go.Bar(
                x=xs, y=ys, marker_color=cores,
                text=[f"{v:.1f}%" for v in ys], textposition="outside",
            )
        )
        figw.update_layout(xaxis_type="category")

    figw.update_layout(
        height=430,
        margin=dict(l=10, r=20, t=10, b=10),
        xaxis_title=feat_label,
        yaxis_title="Prob. de reincidência (%)",
        yaxis_range=[0, 100],
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Sora, sans-serif", size=12, color="#2e2a26"),
    )
    st.plotly_chart(figw, width="stretch")

    if base_prob is not None:
        if base_risk:
            _, blbl, _, _ = cor_por_nivel(base_risk)
        else:
            _, blbl, _, _ = cor_por_prob(base_prob)
        st.caption(f"Perfil base ({feat_label} = {base_value}): {base_prob * 100:.1f}% · nível {blbl}.")

    with st.expander("Pontos da curva"):
        st.dataframe(pd.DataFrame(points), width="stretch", hide_index=True)


def render_batch(batch: dict) -> None:
    """Renderiza a resposta de POST /predict/batch (métricas, distribuições, tabela, download)."""
    summary = batch.get("summary", {}) or {}
    items = batch.get("items", []) or []

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", summary.get("total", 0))
    m2.metric("Pontuados", summary.get("succeeded", 0))
    m3.metric("Falhas", summary.get("failed", 0))
    meanp = summary.get("mean_probability_reincidencia")
    m4.metric("Prob. média", f"{meanp * 100:.1f}%" if meanp is not None else "—")

    warn = batch.get("preprocessing_warning")
    if warn:
        st.caption(f"Aviso: {warn}")

    # Distribuição por nível de risco
    rc = summary.get("risk_counts", {}) or {}
    if rc:
        ordem = ["baixo", "moderado", "alto"]
        xr = [NIVEL_LABEL[n] for n in ordem if n in rc]
        yr = [rc[n] for n in ordem if n in rc]
        cr = [NIVEL_CORES[n] for n in ordem if n in rc]
        figr = go.Figure(go.Bar(x=xr, y=yr, marker_color=cr, text=yr, textposition="outside"))
        figr.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Nível de risco", yaxis_title="Casos",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Sora, sans-serif", size=12, color="#2e2a26"),
        )
        st.plotly_chart(figr, width="stretch")

    # Distribuição por classe prevista
    lc = summary.get("label_counts", {}) or {}
    if lc:
        st.markdown("<b>Distribuição por classe prevista</b>", unsafe_allow_html=True)
        st.dataframe(
            pd.DataFrame([{"Classe": k, "Casos": v} for k, v in lc.items()]),
            width="stretch", hide_index=True,
        )

    # Tabela de resultados por linha + download
    df_items = pd.DataFrame(items)
    if not df_items.empty:
        view = df_items.copy()
        if "probability_reincidencia" in view.columns:
            view["probability_reincidencia"] = view["probability_reincidencia"].apply(
                lambda v: f"{v * 100:.1f}%" if pd.notna(v) else "—"
            )
        cols = [c for c in ["index", "prediction_label", "probability_reincidencia", "risk_level", "error"] if c in view.columns]
        st.dataframe(view[cols], width="stretch", hide_index=True)
        csv = df_items.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar resultados (CSV)", data=csv,
            file_name="resultados_lote.csv", mime="text/csv", key="dl_batch",
        )


def render_importance(entry: dict, model: str, note) -> None:
    """Renderiza a importância global de um modelo (GET /model/importance)."""
    if not entry.get("available"):
        st.error(f"Modelo indisponível: {entry.get('error') or 'artefato não carregado'}")
        return

    st.caption(f"Método: {entry.get('method', '—')}")

    importances = entry.get("importances") or {}
    if importances:
        itens = list(importances.items())[:15][::-1]
        nomes = [k for k, _ in itens]
        vals = [v for _, v in itens]
        figi = go.Figure(
            go.Bar(
                x=vals, y=nomes, orientation="h", marker_color="#3d7a8a",
                text=[f"{v:.3f}" for v in vals], textposition="outside",
            )
        )
        figi.update_layout(
            height=max(320, 26 * len(nomes)),
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis_title="Importância normalizada (soma = 1)",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Sora, sans-serif", size=12, color="#2e2a26"),
        )
        st.plotly_chart(figi, width="stretch")

    # Regressão logística: coeficientes assinados + razões de chance
    odds = entry.get("odds_ratios")
    coefs = entry.get("coefficients")
    if odds and coefs:
        st.markdown(
            "<div style='margin-top:1rem;font-weight:600'>Coeficientes (log-odds) — "
            "regressão logística</div>",
            unsafe_allow_html=True,
        )
        citens = list(coefs.items())[:15][::-1]
        cn = [k for k, _ in citens]
        cv = [v for _, v in citens]
        cores = ["#c0392b" if v > 0 else "#27ae60" for v in cv]
        figco = go.Figure(
            go.Bar(
                x=cv, y=cn, orientation="h", marker_color=cores,
                text=[f"{v:+.3f}" for v in cv], textposition="outside",
            )
        )
        figco.add_vline(x=0, line_width=1, line_color="#8a8278")
        figco.update_layout(
            height=max(320, 26 * len(cn)),
            margin=dict(l=10, r=40, t=10, b=10),
            xaxis_title="Coeficiente (log-odds; + empurra p/ Reincidência)",
            plot_bgcolor="white", paper_bgcolor="white",
            font=dict(family="Sora, sans-serif", size=12, color="#2e2a26"),
        )
        st.plotly_chart(figco, width="stretch")

        dfo = pd.DataFrame(
            [
                {"Feature": k, "Coeficiente": round(coefs.get(k, 0.0), 4), "Odds ratio": round(v, 4)}
                for k, v in odds.items()
            ]
        )
        st.dataframe(dfo, width="stretch", hide_index=True)
        st.caption(
            "Odds ratio > 1: cada +1 desvio-padrão da feature multiplica as chances de "
            "reincidência por esse fator; < 1 reduz."
        )
        inter = entry.get("intercept")
        if inter is not None:
            st.caption(f"Intercepto (log-odds): {inter:.4f}")

    if note:
        st.caption(note)


def render_municipio_badge(ctx) -> None:
    """Mostra o resultado da busca por município (badge do cluster, 404 ou erro)."""
    if not ctx:
        return
    if ctx.get("error") is not None:
        mostrar_erro(ctx["error"])
        return
    if ctx.get("not_found"):
        st.info(
            f"Município {ctx['not_found']} não está na base de clusters "
            "(cobertura de ~4.248 dos 5.570 municípios). Preencha os campos manualmente."
        )
        return
    row = ctx.get("row") or {}
    clabel = row.get("cluster_label", "—")
    cor = cor_cluster(row.get("cluster"), clabel)
    uf_sigla = IBGE_TO_UF.get(str(row.get("uf") or ""), str(row.get("uf") or "—"))
    tx = row.get("taxa_reincidencia")
    tx_txt = f"{tx * 100:.1f}%" if tx is not None else "—"
    st.markdown(
        f"""
        <div class="result-card" style="padding:0.9rem 1.1rem;margin:0.2rem 0 1rem;">
            <small style="color:#8a8278;text-transform:uppercase;letter-spacing:0.08em;
                          font-size:0.66rem;font-weight:600;">
                Município {row.get('cod_mun6', '—')} · {uf_sigla} · {row.get('nome_regiao', '—')}
            </small>
            <div style="margin-top:0.35rem;font-size:1.05rem;font-weight:700;color:{cor};">
                Cluster: {clabel}
            </div>
            <div style="margin-top:0.3rem;font-size:0.8rem;color:#5b5451;">
                Reincidência média municipal: <b>{tx_txt}</b> &nbsp;·&nbsp;
                IDHM/renda/Gini/pobreza preenchidos a partir do IBGE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Variáveis do perfil médio exibidas nos cards e no radar da tipologia
TIPOLOGIA_CAMPOS = [
    ("idhm", "IDHM", lambda v: f"{v:.3f}"),
    ("renda_pc", "Renda per capita", lambda v: f"R$ {v:,.0f}"),
    ("indice_gini", "Gini", lambda v: f"{v:.3f}"),
    ("prop_pobreza", "Pobreza", lambda v: f"{v:.1f}%"),
    ("taxa_analfabetismo", "Analfabetismo", lambda v: f"{v:.1f}%"),
    ("taxa_reincidencia", "Reincidência", lambda v: f"{v * 100:.1f}%"),
    ("n_notificacoes", "Notificações (méd.)", lambda v: f"{v:,.0f}"),
]


def render_tipologia(data: dict) -> None:
    """Renderiza a tipologia de clusters (cards de perfil médio + radar comparativo)."""
    clusters = sorted(data.get("clusters", []) or [], key=lambda c: c.get("cluster", 0))
    if not clusters:
        st.info("Nenhum cluster retornado pelo backend.")
        return

    # Cards: um por cluster, com contagem e perfil médio
    for col, c in zip(st.columns(len(clusters)), clusters):
        clabel = c.get("cluster_label", "—")
        cor = cor_cluster(c.get("cluster"), clabel)
        means = c.get("means", {}) or {}
        linhas = "".join(
            f"<div style='display:flex;justify-content:space-between;font-size:0.8rem;"
            f"color:#5b5451;margin-top:0.28rem;'><span>{lbl}</span>"
            f"<b>{fmt(means[k])}</b></div>"
            for k, lbl, fmt in TIPOLOGIA_CAMPOS
            if means.get(k) is not None
        )
        col.markdown(
            f"""
            <div class="result-card" style="padding:1rem 1.1rem;">
                <div style="font-size:0.66rem;text-transform:uppercase;letter-spacing:0.08em;
                            font-weight:600;color:#8a8278;">Cluster {c.get('cluster')}</div>
                <div style="font-size:1.1rem;font-weight:700;color:{cor};margin:0.1rem 0 0.15rem;">
                    {clabel}</div>
                <div style="font-size:0.76rem;color:#8a8278;margin-bottom:0.5rem;">
                    {c.get('count', 0)} municípios</div>
                {linhas}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Radar comparativo: cada variável normalizada 0-1 entre os clusters
    feats = [k for k, _, _ in TIPOLOGIA_CAMPOS]
    labels = {k: lbl for k, lbl, _ in TIPOLOGIA_CAMPOS}
    faixas = {}
    for k in feats:
        vs = [
            (c.get("means", {}) or {}).get(k)
            for c in clusters
            if (c.get("means", {}) or {}).get(k) is not None
        ]
        faixas[k] = (min(vs), max(vs)) if vs else (0.0, 1.0)

    figt = go.Figure()
    theta = [labels[k] for k in feats] + [labels[feats[0]]]
    for c in clusters:
        means = c.get("means", {}) or {}
        r, raw = [], []
        for k in feats:
            v = means.get(k)
            lo, hi = faixas[k]
            r.append(0.5 if v is None or hi == lo else (v - lo) / (hi - lo))
            raw.append(v if v is not None else float("nan"))
        figt.add_trace(
            go.Scatterpolar(
                r=r + [r[0]], theta=theta, customdata=raw + [raw[0]],
                fill="toself", name=c.get("cluster_label", "—"),
                line=dict(color=cor_cluster(c.get("cluster"), c.get("cluster_label", ""))),
                hovertemplate="%{theta}: %{customdata:.2f}<extra>"
                + str(c.get("cluster_label", "")) + "</extra>",
            )
        )
    figt.update_layout(
        height=460, margin=dict(l=40, r=40, t=50, b=40),
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], showticklabels=False)),
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="center", x=0.5),
        paper_bgcolor="white",
        font=dict(family="Sora, sans-serif", size=12, color="#2e2a26"),
    )
    st.plotly_chart(figt, width="stretch")
    st.caption(
        "Radar normalizado (0–1 por variável entre os clusters; passe o mouse para o valor "
        f"bruto). Médias sobre {data.get('total', '—')} municípios "
        "(linha-agregado '000000' excluída)."
    )


def _buscar_municipio() -> None:
    """Callback do botão 'Buscar município': consulta o IBGE e preenche o perfil."""
    cod = (st.session_state.get("cod_mun6_input") or "").strip()
    if not cod:
        st.session_state["muni_ctx"] = None
        return
    try:
        row = chamar_municipio(cod)
    except Exception as e:
        st.session_state["muni_ctx"] = {"error": e}
        return
    if row is None:
        st.session_state["muni_ctx"] = {"not_found": cod}
        return
    uf_sigla = IBGE_TO_UF.get(str(row.get("uf") or ""))
    if uf_sigla:
        st.session_state["uf_select"] = uf_sigla
        st.session_state["last_uf"] = uf_sigla  # evita o reset automático da faixa de IDHM
    if row.get("idhm") is not None:
        st.session_state["faixa_idhm"] = faixa_from_idhm(float(row["idhm"]))
    if row.get("renda_pc") is not None:
        st.session_state["renda_pc"] = round(float(row["renda_pc"]), 2)
    if row.get("indice_gini") is not None:
        st.session_state["indice_gini"] = round(float(row["indice_gini"]), 2)
    if row.get("prop_pobreza") is not None:
        st.session_state["prop_pobreza"] = round(float(row["prop_pobreza"]) * 2) / 2
    st.session_state["muni_ctx"] = {"row": row}


# ---------------------------------------------
# Sidebar
# ---------------------------------------------
with st.sidebar:
    st.markdown("### VulneraMapa")
    st.markdown(
        "<small style='color:#555'>Modelo preditivo de reincidência<br>"
        "Violência doméstica · UFRA 2026</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("""
    <p style='font-size:0.72rem;color:#444466;line-height:1.7'>
    <b>Fontes de dados</b><br>
    SINAN/DataSUS 2010–2024<br>
    PNAD Contínua 2019<br>
    IPEA — IDH municipal 2010<br>
    FBSP — Anuário 2024
    </p>
    <p style='font-size:0.72rem;color:#444466;line-height:1.7;margin-top:0.5rem'>
    Análise sempre sobre <b>perfis de grupo</b>,<br>nunca indivíduos identificados.<br>
    Alinhado à LGPD · Lei Maria da Penha · ODS 5
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p class='bloco-label' style='margin-top:0'>Status do backend</p>", unsafe_allow_html=True)
    try:
        _features_meta = chamar_model_features()
        _scaler_ok = _features_meta.get("scaler_available")
        _encoders_ok = _features_meta.get("label_encoders_available")
        _status_cor = "#27ae60" if (_scaler_ok and _encoders_ok) else "#d4a017"
        st.markdown(
            f"<small style='color:#444'><span style='color:{_status_cor}'>&#9679;</span> Modelo online — "
            f"{len(_features_meta.get('feature_order', []))} features<br>"
            f"Scaler: {'ok' if _scaler_ok else 'ausente'} · "
            f"Encoders: {'ok' if _encoders_ok else 'ausente'}</small>",
            unsafe_allow_html=True,
        )
        _av = _features_meta.get("available_models", {}) or {}
        _faltantes = [MODELO_LABEL[m] for m in MODELOS if not _av.get(m, {}).get("available")]
        if _faltantes:
            st.markdown(
                "<small style='color:#b9770e'>Indisponível no servidor: "
                f"{', '.join(_faltantes)}</small>",
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            "<small style='color:#c0392b'>&#9679; Não foi possível consultar "
            f"<code>{FEATURES_ENDPOINT}</code>.<br>Verifique a URL e as credenciais.</small>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("""
    <small style='color:#555'><b>Desenvolvedores</b><br>
    Gabriel Junichiro Soares Inada<br>
    Natalia Vanessa Lopes Macedo<br>
    Wallace Luan da Cruz Guimarães<br>

    """, unsafe_allow_html=True)

# ---------------------------------------------
# Cabeçalho
# ---------------------------------------------
st.markdown(
    '<p class="main-subtitle">Uma ferramenta de apoio para profissionais — estimativa por perfil de grupo</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h1 class="main-title">Vulnerabilidade à Violência Doméstica</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="fonte-tag">Estimativa de risco de reincidência • dados agregados</p>',
    unsafe_allow_html=True,
)
st.markdown("---")


# =============================================
# Perfil compartilhado (base para todas as abas)
# =============================================
st.markdown("#### Simulador de Perfil Sociodemográfico")
st.markdown(
    "<small style='color:#55556a'>Defina as características do perfil <b>uma única vez</b>. "
    "As abas abaixo reutilizam este mesmo perfil para estimar o risco, comparar os modelos "
    "e analisar a sensibilidade (<i>what-if</i>) da probabilidade de <b>reincidência</b>.</small>",
    unsafe_allow_html=True,
)
st.markdown("")

# - UF e região ---------------------------
st.session_state.setdefault("uf_select", "PA")
uf = st.selectbox(
    "UF de residência",
    options=UFS,
    key="uf_select",
)

regiao_auto = UF_PARA_REGIAO.get(uf, "Norte")
idhm_auto   = IDHM_POR_REGIAO.get(regiao_auto, "Médio (0.60–0.69)")

if st.session_state.get("last_uf") != uf:
    st.session_state["last_uf"] = uf
    st.session_state["faixa_idhm"] = idhm_auto

st.markdown('<p class="bloco-label">Contexto municipal (IDH)</p>', unsafe_allow_html=True)

mcod1, mcod2 = st.columns([3, 1])
with mcod1:
    st.text_input(
        "Código IBGE do município (6 dígitos, opcional)",
        key="cod_mun6_input",
        placeholder="ex.: 150140 (Belém)",
        help="Preenche IDHM, renda, Gini e pobreza a partir da base municipal e mostra o cluster de risco.",
    )
with mcod2:
    st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    st.button(
        "Buscar município", key="btn_buscar_muni",
        on_click=_buscar_municipio, width="stretch",
    )

render_municipio_badge(st.session_state.get("muni_ctx"))

faixa_idhm = st.select_slider(
    "Faixa de IDHM municipal",
    options=IDHM_OPTIONS,
    key="faixa_idhm",
    help="Valor inicial baseado na média da macroregião da UF selecionada",
)

st.markdown(
    f"""
    <div style="background:#ffffff;border:1px solid #d8d3cc;border-radius:8px;
                padding:0.85rem 1rem;margin-top:1.2rem;margin-bottom:1.7rem;">
        <small style="color:#8a8278;text-transform:uppercase;letter-spacing:0.08em;
                       font-size:0.68rem;font-weight:600;">Macroregião</small>
        <div style="margin-top:0.35rem;font-size:1rem;font-weight:600;color:#2e2a26;">
            {regiao_auto}
        </div>
        <div style="margin-top:0.45rem;font-size:0.82rem;color:#5b5451;">
            IDHM de referência: <b>{idhm_auto}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# - Bloco 1: Perfil da vítima ---------------
st.markdown('<p class="bloco-label">Perfil da vítima</p>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    faixa_key = st.selectbox(
        "Faixa etária",
        options=FAIXAS_ETARIAS,
        format_func=lambda x: FAIXAS_DISPLAY[x],
        index=2,
        key="faixa_key",
    )
with c2:
    raca = st.selectbox("Raça / cor", options=RACAS, index=3, key="raca")
with c3:
    escolaridade = st.selectbox("Escolaridade", options=ESCOLARIDADES, index=4, key="escolaridade")

c4, c5, c6 = st.columns(3)
with c4:
    sit_conjugal = st.selectbox("Situação conjugal", options=SIT_CONJUGAL, index=1, key="sit_conjugal")
with c5:
    relacao = st.selectbox("Relação com o agressor", options=RELACAO_AGRESSOR, index=0, key="relacao")
with c6:
    autor_alcool = st.selectbox(
        "Agressor sob efeito de álcool",
        options=["Sim", "Não", "Ignorado"],
        index=2,
        key="autor_alcool",
    )

# - Bloco 2: Tipo de violência --------------
st.markdown('<p class="bloco-label">Tipo(s) de violência relatados</p>', unsafe_allow_html=True)
cv = st.columns(len(TIPOS_VIOLENCIA))
sel_violencias = {}
for i, (col_key, label) in enumerate(TIPOS_VIOLENCIA.items()):
    with cv[i]:
        sel_violencias[col_key] = st.checkbox(label, value=(col_key == "viol_fisica"), key=f"chk_{col_key}")

# - Bloco 3: Encaminhamentos ----------------
st.markdown('<p class="bloco-label">Encaminhamentos realizados</p>', unsafe_allow_html=True)
ce1, ce2, ce3, ce4, ce5 = st.columns(5)
with ce1:
    enc_delegacia      = st.checkbox("Delegacia", key="enc_delegacia")
with ce2:
    enc_deam           = st.checkbox("DEAM", key="enc_deam")
with ce3:
    enc_creas          = st.checkbox("CREAS", key="enc_creas")
with ce4:
    enc_casa_mulher    = st.checkbox("Casa da Mulher", key="enc_casa_mulher")
with ce5:
    enc_saude_mental   = st.checkbox("Saúde mental", key="enc_saude_mental")

# - Bloco 4: Contexto socioeconômico --------
st.markdown('<p class="bloco-label">Contexto socioeconômico</p>', unsafe_allow_html=True)
cs1, cs2, cs3 = st.columns(3)
with cs1:
    st.session_state.setdefault("renda_pc", 1200.0)
    renda_pc = st.number_input(
        "Renda per capita (R$)",
        min_value=0.0, max_value=50000.0,
        step=50.0,
        key="renda_pc",
        help="Renda mensal per capita estimada do município/grupo",
    )
with cs2:
    st.session_state.setdefault("prop_pobreza", 20.0)
    prop_pobreza = st.slider(
        "Proporção em pobreza (%)",
        min_value=0.0, max_value=100.0,
        step=0.5,
        key="prop_pobreza",
        help="% da população abaixo da linha de pobreza",
    )
with cs3:
    st.session_state.setdefault("indice_gini", 0.50)
    indice_gini = st.slider(
        "Índice de Gini",
        min_value=0.0, max_value=1.0,
        step=0.01,
        key="indice_gini",
        help="Desigualdade de renda (0 = perfeita igualdade, 1 = máxima desigualdade)",
    )

cs4, cs5 = st.columns(2)
with cs4:
    indice_vulnerabilidade = st.slider(
        "Índice de vulnerabilidade (PCA)",
        min_value=0.0, max_value=1.0,
        value=0.30, step=0.01,
        key="indice_vulnerabilidade",
        help="PC1 normalizado gerado pelo notebook de limpeza",
    )
with cs5:
    cons_suicidio = st.checkbox(
        "Suspeita de tentativa de suicídio",
        key="cons_suicidio",
        help="Indicador de risco adicional registrado no boletim",
    )


def montar_payload() -> dict:
    """Monta o corpo de campos nomeados (sem 'model'/'features') a partir do perfil atual."""
    viol_values = {k: (1 if v else 0) for k, v in sel_violencias.items()}
    n_tipos = sum(viol_values.values())
    enc_list = [enc_delegacia, enc_deam, enc_creas, enc_casa_mulher, enc_saude_mental]
    n_enc = sum(1 for e in enc_list if e)

    return {
        "ano_num": 2024,
        "cons_saude_mental": 1 if enc_saude_mental else 0,
        "cons_suicidio": 1 if cons_suicidio else 0,
        "enc_casa_mulher": 1 if enc_casa_mulher else 0,
        "enc_creas": 1 if enc_creas else 0,
        "enc_deam": 1 if enc_deam else 0,
        "enc_delegacia": 1 if enc_delegacia else 0,
        "escolaridade": escolaridade,
        "idade": FAIXA_PARA_IDADE[faixa_key],
        "idhm": IDHM_VALOR[faixa_idhm],
        "indice_gini": round(indice_gini, 2),
        "indice_vulnerabilidade": round(indice_vulnerabilidade, 2),
        "n_encaminhamentos": n_enc,
        "n_tipos_violencia": n_tipos,
        "nome_regiao": regiao_auto,
        "parceiro_intimo": 1 if relacao in PARCEIRO_INTIMO_SET else 0,
        "prop_pobreza": round(prop_pobreza / 100, 4),
        "raca": raca,
        "relacao_agressor": relacao,
        "renda_pc": round(renda_pc, 2),
        "sit_conjugal": sit_conjugal,
        "uf": uf,
        **viol_values,
    }


st.markdown("---")

# =============================================
# Abas: Simulador · Comparar modelos · What-if
# =============================================
tab_sim, tab_cmp, tab_wi, tab_lote, tab_imp, tab_tip = st.tabs(
    ["Simulador", "Comparar modelos", "What-if", "Lote (CSV)", "Importância",
     "Tipologia municipal"]
)

# -- Aba 1: Simulador (POST /predict) ----------
with tab_sim:
    st.markdown("##### Estimativa para o perfil selecionado")
    modelo_sim = st.selectbox(
        "Modelo",
        options=modelos_disponiveis(),
        format_func=lambda m: MODELO_LABEL[m],
        key="modelo_sim",
        help="Regressão logística usa features escaladas; os modelos de árvore usam valores brutos.",
    )
    estimar = st.button(
        "Estimar Probabilidade de Reincidência", key="btn_predict", width="stretch"
    )
    st.markdown("---")

    if estimar:
        payload = {**montar_payload(), "model": modelo_sim}
        _cod_mun = (st.session_state.get("cod_mun6_input") or "").strip()
        if _cod_mun:
            payload["cod_mun6"] = _cod_mun
        with st.spinner("Consultando o modelo..."):
            try:
                st.session_state["sim_result"] = (payload, chamar_predict(payload))
            except Exception as e:
                st.session_state["sim_result"] = None
                mostrar_erro(e)

    if st.session_state.get("sim_result"):
        _payload, _resultado = st.session_state["sim_result"]
        render_prediction(_resultado, _payload)
    elif not estimar:
        st.markdown(
            '<div class="result-placeholder">'
            'Ajuste o perfil acima e clique em <b>Estimar Probabilidade de Reincidência</b>.'
            '</div>',
            unsafe_allow_html=True,
        )

# -- Aba 2: Comparar modelos (POST /predict/compare) --
with tab_cmp:
    st.markdown("##### Comparação entre os 4 modelos")
    st.markdown(
        "<small style='color:#55556a'>O mesmo perfil é avaliado por todos os modelos "
        "disponíveis, com um voto de <b>consenso</b> por maioria.</small>",
        unsafe_allow_html=True,
    )
    comparar = st.button("Comparar os 4 modelos", key="btn_compare", width="stretch")
    st.markdown("---")

    if comparar:
        with st.spinner("Consultando os modelos..."):
            try:
                st.session_state["cmp_result"] = chamar_compare(montar_payload())
            except Exception as e:
                st.session_state["cmp_result"] = None
                mostrar_erro(e)

    if st.session_state.get("cmp_result"):
        render_comparison(st.session_state["cmp_result"])
    elif not comparar:
        st.markdown(
            '<div class="result-placeholder">'
            'Clique em <b>Comparar os 4 modelos</b> para avaliar o perfil atual em todos eles.'
            '</div>',
            unsafe_allow_html=True,
        )

# -- Aba 3: What-if (POST /predict/what-if) ----
with tab_wi:
    st.markdown("##### Análise de sensibilidade (what-if)")
    st.markdown(
        "<small style='color:#55556a'>Varia <b>uma única variável</b> mantendo o restante "
        "do perfil fixo e traça a curva de probabilidade de reincidência.</small>",
        unsafe_allow_html=True,
    )

    wc1, wc2 = st.columns(2)
    with wc1:
        modelo_wi = st.selectbox(
            "Modelo", options=modelos_disponiveis(),
            format_func=lambda m: MODELO_LABEL[m], key="modelo_wi",
        )
    with wc2:
        feature_wi = st.selectbox(
            "Variável a analisar",
            options=WHATIF_FEATURES,
            format_func=lambda f: WHATIF_LABEL.get(f, f),
            key="feature_wi",
        )

    kind = whatif_kind(feature_wi)
    payload = {"model": modelo_wi, "feature": feature_wi, "base": None}
    valores_sel = None

    if kind == "num":
        dflt_start, dflt_stop, _ = SWEEP_CONFIG[feature_wi]
        wc3, wc4, wc5 = st.columns(3)
        with wc3:
            start = st.number_input("Início", value=float(dflt_start), key=f"wi_start_{feature_wi}")
        with wc4:
            stop = st.number_input("Fim", value=float(dflt_stop), key=f"wi_stop_{feature_wi}")
        with wc5:
            steps = st.slider("Pontos", min_value=2, max_value=200, value=20, key="wi_steps")
        payload.update(start=float(start), stop=float(stop), steps=int(steps))
    elif kind == "cat":
        valores_sel = st.multiselect(
            "Valores a testar",
            options=WHATIF_CAT[feature_wi],
            default=list(WHATIF_CAT[feature_wi]),
            key=f"wi_vals_{feature_wi}",
        )
        payload["values"] = valores_sel
    else:  # bin
        st.caption("Variável binária: serão testados os valores 0 (não) e 1 (sim).")
        payload["values"] = [0, 1]

    gerar = st.button("Gerar curva de risco", key="btn_whatif", width="stretch")
    st.markdown("---")

    if gerar:
        if kind == "cat" and len(valores_sel or []) < 2:
            st.warning("Selecione ao menos 2 valores para comparar.")
        else:
            payload["base"] = montar_payload()
            with st.spinner("Calculando curva..."):
                try:
                    st.session_state["wi_result"] = (
                        feature_wi, kind, chamar_what_if(payload)
                    )
                except Exception as e:
                    st.session_state["wi_result"] = None
                    mostrar_erro(e)

    if st.session_state.get("wi_result"):
        _f, _k, _wi = st.session_state["wi_result"]
        render_whatif(_f, _k, _wi)
    elif not gerar:
        st.markdown(
            '<div class="result-placeholder">'
            'Escolha uma variável e clique em <b>Gerar curva de risco</b> para ver a sensibilidade.'
            '</div>',
            unsafe_allow_html=True,
        )

# -- Aba 4: Lote (CSV) (POST /predict/batch) ---
with tab_lote:
    st.markdown("##### Pontuação em lote (CSV)")
    st.markdown(
        "<small style='color:#55556a'>Envie um CSV onde cada linha é um caso com os "
        "campos nomeados do modelo. Até <b>500 casos</b> por envio.</small>",
        unsafe_allow_html=True,
    )

    lc1, lc2 = st.columns(2)
    with lc1:
        modelo_lote = st.selectbox(
            "Modelo", options=modelos_disponiveis(),
            format_func=lambda m: MODELO_LABEL[m], key="modelo_lote",
        )
    with lc2:
        incluir_shap = st.checkbox(
            "Incluir SHAP por linha", value=False, key="lote_shap",
            help="Anexa a explicação SHAP de cada linha (dobra o tamanho da resposta).",
        )

    # Template com uma linha de exemplo (perfil atual) para o usuário preencher
    _template_df = pd.DataFrame([montar_payload()])
    st.download_button(
        "Baixar modelo de CSV (1 exemplo)",
        data=_template_df.to_csv(index=False).encode("utf-8"),
        file_name="modelo_lote.csv", mime="text/csv", key="dl_template",
    )

    up = st.file_uploader("CSV de casos", type=["csv"], key="lote_csv")
    df_lote = None
    if up is not None:
        try:
            df_lote = pd.read_csv(up)
        except Exception as e:
            st.error(f"Não foi possível ler o CSV: {e}")

    pontuar = False
    if df_lote is not None:
        descartadas = [c for c in df_lote.columns if c not in PREDICT_FIELDS]
        if descartadas:
            st.warning(f"Colunas ignoradas (não reconhecidas pelo modelo): {', '.join(descartadas)}")
        df_lote = df_lote[[c for c in df_lote.columns if c in PREDICT_FIELDS]]
        if len(df_lote) > 500:
            st.warning(f"{len(df_lote)} linhas recebidas; apenas as 500 primeiras serão pontuadas.")
            df_lote = df_lote.head(500)
        st.markdown(
            f"<small style='color:#666'>Pré-visualização ({len(df_lote)} linhas):</small>",
            unsafe_allow_html=True,
        )
        st.dataframe(df_lote.head(), width="stretch", hide_index=True)
        pontuar = st.button("Pontuar lote", key="btn_batch", width="stretch")

    st.markdown("---")

    if pontuar and df_lote is not None:
        # to_json converte NaN→null e emite tipos nativos (evita erro de serialização numpy)
        items = json.loads(df_lote.to_json(orient="records"))
        payload = {"model": modelo_lote, "include_shap": incluir_shap, "items": items}
        with st.spinner("Pontuando o lote..."):
            try:
                st.session_state["batch_result"] = chamar_batch(payload)
            except Exception as e:
                st.session_state["batch_result"] = None
                mostrar_erro(e)

    if st.session_state.get("batch_result"):
        render_batch(st.session_state["batch_result"])
    elif not pontuar:
        st.markdown(
            '<div class="result-placeholder">'
            'Baixe o modelo de CSV, preencha os casos e faça o upload para pontuar em lote.'
            '</div>',
            unsafe_allow_html=True,
        )

# -- Aba 5: Importância global (GET /model/importance) --
with tab_imp:
    st.markdown("##### Importância global das features")
    st.markdown(
        "<small style='color:#55556a'>Quais variáveis mais pesam em cada modelo, no geral "
        "(complementa o SHAP, que explica um caso por vez).</small>",
        unsafe_allow_html=True,
    )
    modelo_imp = st.selectbox(
        "Modelo", options=modelos_disponiveis(),
        format_func=lambda m: MODELO_LABEL[m], key="modelo_imp",
    )
    st.markdown("---")
    try:
        _imp_data = chamar_importance()
        _entry = (_imp_data.get("models", {}) or {}).get(modelo_imp, {})
        render_importance(_entry, modelo_imp, _imp_data.get("note"))
    except Exception as e:
        mostrar_erro(e)

# -- Aba 6: Tipologia municipal (GET /clusters) --
with tab_tip:
    st.markdown("##### Tipologia de risco municipal")
    st.markdown(
        "<small style='color:#55556a'>Os municípios da base foram agrupados em "
        "<b>3 clusters</b> de risco por perfil socioeconômico. Veja o perfil médio de cada "
        "grupo e um radar comparativo.</small>",
        unsafe_allow_html=True,
    )
    with st.expander("Como ler esta aba"):
        st.markdown(
            "- **O que é:** um algoritmo de agrupamento (*clustering*) juntou os ~4,2 mil "
            "municípios da base em **3 grupos** com perfil socioeconômico e histórico de "
            "reincidência parecidos. Cada grupo recebeu um rótulo: **Baixo**, **Médio** e "
            "**Alto risco**.\n"
            "- **Os cards:** mostram o *município médio* de cada grupo — quantos municípios "
            "ele reúne e a média de cada indicador (IDHM, renda, Gini, pobreza, "
            "analfabetismo, reincidência e nº de notificações).\n"
            "- **O radar:** compara os mesmos indicadores, mas **normalizados de 0 a 1** "
            "entre os grupos (o menor valor vira 0 e o maior vira 1), porque as escalas são "
            "muito diferentes (renda em centenas de reais vs. IDHM perto de 0,6). Passe o "
            "mouse sobre o gráfico para ver o **valor bruto**.\n"
            "- **Atenção ao rótulo:** *risco* aqui significa **reincidência observada**, não "
            "grau de pobreza. O grupo de **Alto risco** tende a reunir municípios maiores / "
            "urbanos, com muito mais **notificações** (mais denúncia registrada → mais "
            "reincidência captada). Já grupos mais pobres podem ter menos notificações por "
            "**subnotificação** — o mesmo viés citado na nota metodológica no rodapé."
        )
    st.markdown("---")
    try:
        render_tipologia(chamar_clusters())
    except Exception as e:
        mostrar_erro(e)


st.markdown(
    '<div class="nota-metodologica">'
    '<b>Nota metodológica:</b> Este resultado representa uma estimativa estatística para um '
    '<b>perfil de grupo sociodemográfico</b>, não um diagnóstico individual. '
    'A variável-alvo é a <b>reincidência</b> (campo OUT_VEZES do SINAN). '
    'Grupos com alta subnotificação (mulheres negras, rurais, sem escolaridade formal) '
    'recebem peso maior no treino para mitigar viés estrutural. '
    'Taxa de subnotificação estimada em até 90% (FBSP 2024).'
    '</div>',
    unsafe_allow_html=True,
)
