import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from PIL import Image
import os

# ─────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────
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
        page_icon="🔴",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ─────────────────────────────────────────────
# Configuração de acesso ao backend
# Credenciais vêm de st.secrets (Streamlit Cloud → App settings → Secrets),
# com fallback para variáveis de ambiente na execução local.
# ─────────────────────────────────────────────
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
FEATURES_ENDPOINT = f"{API_URL}/model/features"

MODELOS = ["logistica", "random_forest", "xgboost", "lightgbm"]
MODELO_LABEL = {
    "logistica": "Regressão logística",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Constantes / enumerações
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# Níveis de risco — alinhados aos cutoffs do backend (0.33 / 0.66)
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# Chamadas HTTP autenticadas ao backend
# ─────────────────────────────────────────────
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


@st.cache_data(ttl=3600, show_spinner=False)
def chamar_model_features() -> dict:
    """GET /model/features — metadados do modelo (cacheado por 1h)."""
    resp = requests.get(FEATURES_ENDPOINT, auth=AUTH, timeout=15)
    resp.raise_for_status()
    return resp.json()


def mostrar_erro(e: Exception) -> None:
    """Renderiza uma mensagem de erro amigável para as exceções esperadas."""
    if isinstance(e, PermissionError):
        st.error(f"🔒 {e}")
    elif isinstance(e, requests.exceptions.ConnectionError):
        st.error(
            "❌ Não foi possível conectar ao backend. "
            f"Verifique se o servidor está em execução em `{API_URL}`."
        )
    elif isinstance(e, requests.exceptions.Timeout):
        st.error("⏱️ O backend demorou demais para responder. Tente novamente.")
    elif isinstance(e, (ValueError, RuntimeError)):
        st.error(f"⚠️ {e}")
    elif isinstance(e, requests.exceptions.HTTPError):
        st.error(f"⚠️ Erro HTTP {e.response.status_code}: {e.response.text}")
    elif isinstance(e, (KeyError, TypeError)):
        st.error(f"⚠️ Resposta do backend em formato inesperado (campo ausente: {e}).")
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
        st.caption(f"⚠️ {preproc_warn}")

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
        st.plotly_chart(fig, use_container_width=True)

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
        use_container_width=True,
        hide_index=True,
        height=480,
    )


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
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
        _status_icone = "🟢" if (_scaler_ok and _encoders_ok) else "🟡"
        st.markdown(
            f"<small style='color:#444'>{_status_icone} Modelo online — "
            f"{len(_features_meta.get('feature_order', []))} features<br>"
            f"Scaler: {'ok' if _scaler_ok else 'ausente'} · "
            f"Encoders: {'ok' if _encoders_ok else 'ausente'}</small>",
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            "<small style='color:#c0392b'>🔴 Não foi possível consultar "
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

# ─────────────────────────────────────────────
# Cabeçalho
# ─────────────────────────────────────────────
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


# ═════════════════════════════════════════════
# Perfil compartilhado (base para todas as abas)
# ═════════════════════════════════════════════
st.markdown("#### Simulador de Perfil Sociodemográfico")
st.markdown(
    "<small style='color:#55556a'>Defina as características do perfil <b>uma única vez</b>. "
    "As abas abaixo reutilizam este mesmo perfil para estimar o risco, comparar os modelos "
    "e analisar a sensibilidade (<i>what-if</i>) da probabilidade de <b>reincidência</b>.</small>",
    unsafe_allow_html=True,
)
st.markdown("")

# — UF e região ———————————————————————————
uf = st.selectbox(
    "UF de residência",
    options=UFS,
    index=UFS.index("PA"),
    key="uf_select",
)

regiao_auto = UF_PARA_REGIAO.get(uf, "Norte")
idhm_auto   = IDHM_POR_REGIAO.get(regiao_auto, "Médio (0.60–0.69)")

if st.session_state.get("last_uf") != uf:
    st.session_state["last_uf"] = uf
    st.session_state["faixa_idhm"] = idhm_auto

st.markdown('<p class="bloco-label">Contexto municipal (IDH)</p>', unsafe_allow_html=True)

faixa_idhm = st.select_slider(
    "Faixa de IDHM municipal",
    options=IDHM_OPTIONS,
    value=st.session_state.get("faixa_idhm", idhm_auto),
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

# — Bloco 1: Perfil da vítima ———————————————
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

# — Bloco 2: Tipo de violência ——————————————
st.markdown('<p class="bloco-label">Tipo(s) de violência relatados</p>', unsafe_allow_html=True)
cv = st.columns(len(TIPOS_VIOLENCIA))
sel_violencias = {}
for i, (col_key, label) in enumerate(TIPOS_VIOLENCIA.items()):
    with cv[i]:
        sel_violencias[col_key] = st.checkbox(label, value=(col_key == "viol_fisica"), key=f"chk_{col_key}")

# — Bloco 3: Encaminhamentos ————————————————
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

# — Bloco 4: Contexto socioeconômico ————————
st.markdown('<p class="bloco-label">Contexto socioeconômico</p>', unsafe_allow_html=True)
cs1, cs2, cs3 = st.columns(3)
with cs1:
    renda_pc = st.number_input(
        "Renda per capita (R$)",
        min_value=0.0, max_value=50000.0,
        value=1200.0, step=50.0,
        key="renda_pc",
        help="Renda mensal per capita estimada do município/grupo",
    )
with cs2:
    prop_pobreza = st.slider(
        "Proporção em pobreza (%)",
        min_value=0.0, max_value=100.0,
        value=20.0, step=0.5,
        key="prop_pobreza",
        help="% da população abaixo da linha de pobreza",
    )
with cs3:
    indice_gini = st.slider(
        "Índice de Gini",
        min_value=0.0, max_value=1.0,
        value=0.50, step=0.01,
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

# ═════════════════════════════════════════════
# Abas: Simulador · Comparar modelos · What-if
# ═════════════════════════════════════════════
tab_sim, tab_cmp, tab_wi = st.tabs(
    ["🎯 Simulador", "⚖️ Comparar modelos", "📈 What-if"]
)

# ── Aba 1: Simulador (POST /predict) ──────────
with tab_sim:
    st.markdown("##### Estimativa para o perfil selecionado")
    modelo_sim = st.selectbox(
        "Modelo",
        options=MODELOS,
        format_func=lambda m: MODELO_LABEL[m],
        key="modelo_sim",
        help="Regressão logística usa features escaladas; os modelos de árvore usam valores brutos.",
    )
    estimar = st.button(
        "Estimar Probabilidade de Reincidência", key="btn_predict", use_container_width=True
    )
    st.markdown("---")

    if estimar:
        payload = {**montar_payload(), "model": modelo_sim}
        with st.spinner("Consultando o modelo…"):
            try:
                resultado = chamar_predict(payload)
                render_prediction(resultado, payload)
            except Exception as e:
                mostrar_erro(e)
    else:
        st.markdown(
            '<div class="result-placeholder">'
            'Ajuste o perfil acima e clique em <b>Estimar Probabilidade de Reincidência</b>.'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Aba 2: Comparar modelos (POST /predict/compare) ──
with tab_cmp:
    st.markdown("##### Comparação entre os 4 modelos")
    st.markdown(
        "<small style='color:#55556a'>O mesmo perfil é avaliado por todos os modelos "
        "disponíveis, com um voto de <b>consenso</b> por maioria.</small>",
        unsafe_allow_html=True,
    )
    comparar = st.button("Comparar os 4 modelos", key="btn_compare", use_container_width=True)
    st.markdown("---")

    if comparar:
        with st.spinner("Consultando os modelos…"):
            try:
                comp = chamar_compare(montar_payload())
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
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

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
                    st.plotly_chart(figc, use_container_width=True)

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
            except Exception as e:
                mostrar_erro(e)
    else:
        st.markdown(
            '<div class="result-placeholder">'
            'Clique em <b>Comparar os 4 modelos</b> para avaliar o perfil atual em todos eles.'
            '</div>',
            unsafe_allow_html=True,
        )

# ── Aba 3: What-if (POST /predict/what-if) ────
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
            "Modelo", options=MODELOS, format_func=lambda m: MODELO_LABEL[m], key="modelo_wi"
        )
    with wc2:
        feature_wi = st.selectbox(
            "Variável a analisar",
            options=list(SWEEP_CONFIG.keys()),
            format_func=lambda f: SWEEP_CONFIG[f][2],
            key="feature_wi",
        )

    dflt_start, dflt_stop, feat_label = SWEEP_CONFIG[feature_wi]
    wc3, wc4, wc5 = st.columns(3)
    with wc3:
        start = st.number_input("Início", value=float(dflt_start), key=f"wi_start_{feature_wi}")
    with wc4:
        stop = st.number_input("Fim", value=float(dflt_stop), key=f"wi_stop_{feature_wi}")
    with wc5:
        steps = st.slider("Pontos", min_value=2, max_value=200, value=20, key="wi_steps")

    gerar = st.button("Gerar curva de risco", key="btn_whatif", use_container_width=True)
    st.markdown("---")

    if gerar:
        payload = {
            "model": modelo_wi,
            "feature": feature_wi,
            "base": montar_payload(),
            "start": float(start),
            "stop": float(stop),
            "steps": int(steps),
        }
        with st.spinner("Calculando curva…"):
            try:
                wi = chamar_what_if(payload)
                points = wi.get("points", []) or []
                xs = [p["value"] for p in points]
                ys = [(p.get("probability_reincidencia") or 0) * 100 for p in points]

                figw = go.Figure()
                # Faixas de risco (mesmos cutoffs do backend)
                figw.add_hrect(y0=0, y1=33, fillcolor=NIVEL_CORES["baixo"], opacity=0.08, line_width=0)
                figw.add_hrect(y0=33, y1=66, fillcolor=NIVEL_CORES["moderado"], opacity=0.08, line_width=0)
                figw.add_hrect(y0=66, y1=100, fillcolor=NIVEL_CORES["alto"], opacity=0.08, line_width=0)
                figw.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines+markers", name="Risco",
                        line=dict(color="#3d7a8a", width=2.5), marker=dict(size=6),
                    )
                )
                base_value = wi.get("base_value")
                base_prob = wi.get("base_probability")
                if base_value is not None and base_prob is not None:
                    figw.add_trace(
                        go.Scatter(
                            x=[base_value], y=[base_prob * 100], mode="markers", name="Perfil base",
                            marker=dict(size=13, color="#2e2a26", symbol="diamond"),
                        )
                    )
                figw.update_layout(
                    height=430,
                    margin=dict(l=10, r=20, t=10, b=10),
                    xaxis_title=feat_label,
                    yaxis_title="Prob. de reincidência (%)",
                    yaxis_range=[0, 100],
                    plot_bgcolor="white", paper_bgcolor="white",
                    font=dict(family="Sora, sans-serif", size=12, color="#2e2a26"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(figw, use_container_width=True)

                if base_prob is not None:
                    base_risk = wi.get("base_risk_level")
                    if base_risk:
                        _, blbl, _, _ = cor_por_nivel(base_risk)
                    else:
                        _, blbl, _, _ = cor_por_prob(base_prob)
                    st.caption(
                        f"Perfil base ({feat_label} = {base_value}): "
                        f"{base_prob * 100:.1f}% · nível {blbl}."
                    )

                with st.expander("Pontos da curva"):
                    st.dataframe(pd.DataFrame(points), use_container_width=True, hide_index=True)
            except Exception as e:
                mostrar_erro(e)
    else:
        st.markdown(
            '<div class="result-placeholder">'
            'Escolha uma variável e clique em <b>Gerar curva de risco</b> para ver a sensibilidade.'
            '</div>',
            unsafe_allow_html=True,
        )


st.markdown(
    '<div class="nota-metodologica">'
    '⚠️ <b>Nota metodológica:</b> Este resultado representa uma estimativa estatística para um '
    '<b>perfil de grupo sociodemográfico</b>, não um diagnóstico individual. '
    'A variável-alvo é a <b>reincidência</b> (campo OUT_VEZES do SINAN). '
    'Grupos com alta subnotificação (mulheres negras, rurais, sem escolaridade formal) '
    'recebem peso maior no treino para mitigar viés estrutural. '
    'Taxa de subnotificação estimada em até 90% (FBSP 2024).'
    '</div>',
    unsafe_allow_html=True,
)
