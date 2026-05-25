import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VulneraMapa — Violência Doméstica",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp { background-color: #0f0f14; color: #e8e4df; }

section[data-testid="stSidebar"] {
    background-color: #13131b;
    border-right: 1px solid #222230;
}

h1, h2, h3, h4 { color: #f0ebe4 !important; }

.main-title {
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #f0ebe4;
    line-height: 1.2;
    margin-bottom: 0.2rem;
}
.main-subtitle {
    font-size: 0.8rem;
    color: #666688;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}

/* blocos de campos */
.bloco-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #c0392b;
    margin-bottom: 0.4rem;
    margin-top: 1.2rem;
}

/* placeholder de resultado */
.result-placeholder {
    background: #12121a;
    border: 1px dashed #2a2a3a;
    border-radius: 10px;
    padding: 2.2rem 1.5rem;
    text-align: center;
    color: #44445a;
    font-size: 0.85rem;
    line-height: 1.7;
}

/* nota metodológica */
.nota-metodologica {
    background: #17170e;
    border-left: 3px solid #c8a84b;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    font-size: 0.78rem;
    color: #a09870;
    margin-top: 1.4rem;
    line-height: 1.6;
}

/* badge mock */
.badge-mock {
    display: inline-block;
    background: #1e1228;
    border: 1px solid #5a2a8a;
    color: #a070d8;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    padding: 2px 8px;
    border-radius: 4px;
    margin-left: 6px;
    vertical-align: middle;
}

/* botão */
.stButton > button {
    background: #c0392b !important;
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
.stButton > button:hover { background: #e74c3c !important; }

hr { border-color: #222232 !important; }

/* selectbox */
div[data-baseweb="select"] > div {
    background-color: #16161f !important;
    border-color: #2a2a3a !important;
    color: #e8e4df !important;
}

/* upload box */
[data-testid="stFileUploader"] {
    background: #14141c;
    border: 1px dashed #2a2a3a;
    border-radius: 10px;
    padding: 0.5rem;
}

.fonte-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #3a3a5a;
    margin-top: 0.3rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Valores reais extraídos do notebook de limpeza
# ─────────────────────────────────────────────

UFS = sorted([
    "AC","AL","AM","AP","BA","CE","DF","ES","GO",
    "MA","MG","MS","MT","PA","PB","PE","PI","PR",
    "RJ","RN","RO","RR","RS","SC","SE","SP","TO",
])

# faixas criadas pelo pd.cut no notebook (bins=[0,17,24,34,44,54,64,120])
FAIXAS_ETARIAS = [
    "menor_18",
    "18_24",
    "25_34",
    "35_44",
    "45_54",
    "55_64",
    "65_mais",
]
FAIXAS_DISPLAY = {
    "menor_18": "Menor de 18 anos",
    "18_24":    "18–24 anos",
    "25_34":    "25–34 anos",
    "35_44":    "35–44 anos",
    "45_54":    "45–54 anos",
    "55_64":    "55–64 anos",
    "65_mais":  "65 anos ou mais",
}

# mapa_raca do notebook
RACAS = ["Branca", "Preta", "Amarela", "Parda", "Indigena", "Ignorado"]

# mapa_escol do notebook
ESCOLARIDADES = [
    "Sem instrucao",
    "Fundamental incompleto",
    "Fundamental completo",
    "Medio incompleto",
    "Medio completo",
    "Superior incompleto",
    "Superior completo",
    "Ignorado",
]

# mapa_conjug do notebook
SIT_CONJUGAL = [
    "Solteira",
    "Casada/Uniao estavel",
    "Viuva",
    "Separada",
    "Nao se aplica",
    "Ignorado",
]

# relacao_agressor derivada no notebook
RELACAO_AGRESSOR = [
    "Conjuge",
    "Ex_conjuge",
    "Namorado",
    "Ex_namorado",
    "Pai",
    "Filho",
    "Irmao",
    "Conhecido",
    "Desconhecido",
    "Outros",
    "Ignorado",
]

# tipos de violência (colunas binárias do df_final)
TIPOS_VIOLENCIA = {
    "viol_fisica":      "Física",
    "viol_psico":       "Psicológica",
    "viol_sexual":      "Sexual",
    "viol_financeira":  "Financeira",
    "viol_negligencia": "Negligência",
}

REGIOES = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔴 VulneraMapa")
    st.markdown(
        "<small style='color:#555'>Modelo preditivo de reincidência<br>"
        "Violência doméstica · UFRA 2025</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    modo = st.radio(
        "**Modo de uso**",
        options=["🧩 Simulador de Perfil", "📂 Análise em Lote"],
        index=0,
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


# ─────────────────────────────────────────────
# Cabeçalho
# ─────────────────────────────────────────────
st.markdown(
    '<p class="main-subtitle">Ciência de Dados II · UFRA · Bacharelado em Sistemas de Informação</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h1 class="main-title">Modelo Preditivo de Vulnerabilidade<br>à Violência Doméstica contra a Mulher</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="fonte-tag">Variável-alvo: reincidência (OUT_VEZES) · Modelo: RandomForestClassifier · Dados: SINAN 2010–2024</p>',
    unsafe_allow_html=True,
)
st.markdown("---")


# ═════════════════════════════════════════════
# TELA 1 — Simulador de Perfil
# ═════════════════════════════════════════════
if modo == "🧩 Simulador de Perfil":

    st.markdown(
        "#### Simulador de Perfil Sociodemográfico"
        '<span class="badge-mock">API pendente</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small style='color:#55556a'>Selecione as características do perfil para estimar a "
        "probabilidade de <b>reincidência</b> de violência doméstica naquele grupo.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    with st.form("form_simulador"):

        # — Bloco 1: Perfil da vítima ——————————————————————
        st.markdown('<p class="bloco-label">Perfil da vítima</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            uf = st.selectbox(
                "UF de residência",
                options=UFS,
                index=UFS.index("PA"),
                help="Coluna 'uf' do df_final (SG_UF original do SINAN)",
            )
        with c2:
            faixa_key = st.selectbox(
                "Faixa etária",
                options=FAIXAS_ETARIAS,
                format_func=lambda x: FAIXAS_DISPLAY[x],
                index=2,
                help="Derivada de NU_IDADE_N com pd.cut (bins 0,17,24,34,44,54,64,120)",
            )
        with c3:
            raca = st.selectbox(
                "Raça / cor",
                options=RACAS,
                index=3,
                help="mapa_raca do notebook — códigos 1–5 + 9=Ignorado",
            )

        c4, c5, c6 = st.columns(3)
        with c4:
            escolaridade = st.selectbox(
                "Escolaridade",
                options=ESCOLARIDADES,
                index=2,
                help="mapa_escol do notebook — CS_ESCOL_N",
            )
        with c5:
            sit_conjugal = st.selectbox(
                "Situação conjugal",
                options=SIT_CONJUGAL,
                index=1,
                help="mapa_conjug do notebook — SIT_CONJUG",
            )
        with c6:
            relacao = st.selectbox(
                "Relação com o agressor",
                options=RELACAO_AGRESSOR,
                index=0,
                help="Derivada das colunas REL_* binárias do SINAN",
            )

        # — Bloco 2: Tipo de violência ——————————————————————
        st.markdown('<p class="bloco-label">Tipo(s) de violência relatados</p>', unsafe_allow_html=True)
        cv = st.columns(len(TIPOS_VIOLENCIA))
        sel_violencias = {}
        for i, (col_key, label) in enumerate(TIPOS_VIOLENCIA.items()):
            with cv[i]:
                sel_violencias[col_key] = st.checkbox(label, value=(col_key == "viol_fisica"))

        # — Bloco 3: Contexto municipal ——————————————————————
        st.markdown('<p class="bloco-label">Contexto municipal (IDH)</p>', unsafe_allow_html=True)
        cm1, cm2, cm3 = st.columns(3)
        with cm1:
            faixa_idhm = st.select_slider(
                "Faixa de IDHM municipal",
                options=["Muito baixo (< 0.50)", "Baixo (0.50–0.59)",
                         "Médio (0.60–0.69)", "Alto (0.70–0.79)", "Muito alto (≥ 0.80)"],
                value="Médio (0.60–0.69)",
                help="idhm do IPEA · Atlas Brasil 2010",
            )
        with cm2:
            regiao = st.selectbox("Macrorregião", options=REGIOES, index=0,
                                  help="nome_regiao do df_final")
        with cm3:
            autor_alcool = st.selectbox(
                "Agressor sob efeito de álcool",
                options=["Sim", "Não", "Ignorado"],
                index=2,
                help="AUTOR_ALCO do SINAN",
            )

        submitted = st.form_submit_button(
            "▶  Estimar Probabilidade de Reincidência", use_container_width=True
        )

    # Área de resultado
    st.markdown("---")
    if submitted:
        st.markdown("##### Resultado da Estimativa")

        # resumo do perfil selecionado (sem chamar API ainda)
        viols_sel = [v for k, v in TIPOS_VIOLENCIA.items() if sel_violencias[k]]
        resumo = {
            "UF": uf,
            "Faixa etária": FAIXAS_DISPLAY[faixa_key],
            "Raça": raca,
            "Escolaridade": escolaridade,
            "Sit. conjugal": sit_conjugal,
            "Relação agressor": relacao,
            "Tipos de violência": ", ".join(viols_sel) if viols_sel else "Nenhum selecionado",
            "IDHM": faixa_idhm,
            "Macrorregião": regiao,
            "Agressor c/ álcool": autor_alcool,
        }

        col_res, col_info = st.columns([1, 1])
        with col_res:
            st.markdown(
                '<div class="result-placeholder">'
                '🔌 <b>Aguardando integração com a API do backend (Gabriel)</b><br><br>'
                'Aqui serão exibidos:<br>'
                '• Probabilidade estimada de reincidência (%) com intervalo de confiança<br>'
                '• Comparativo com média nacional e por UF<br>'
                '• Gráfico SHAP interativo dos fatores de maior peso<br>'
                '• Indicador de parceiro_intimo e n_tipos_violencia<br>'
                '• Índice de vulnerabilidade sociodemográfica (PCA)'
                '</div>',
                unsafe_allow_html=True,
            )
        with col_info:
            st.markdown("**Perfil enviado para estimativa:**")
            st.dataframe(
                pd.DataFrame(resumo.items(), columns=["Campo", "Valor"]),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.markdown(
            '<div class="result-placeholder">'
            'Preencha o perfil acima e clique em <b>Estimar Probabilidade de Reincidência</b>.'
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


# ═════════════════════════════════════════════
# TELA 2 — Análise em Lote
# ═════════════════════════════════════════════
elif modo == "📂 Análise em Lote":

    st.markdown(
        "#### Análise em Lote por Dataset Municipal"
        '<span class="badge-mock">API pendente</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<small style='color:#55556a'>Faça upload de microdados agregados no mesmo formato do "
        "<code>df_final.parquet</code> gerado pelo notebook de limpeza. "
        "O sistema retornará ranking de vulnerabilidade e mapa choropleth por UF.</small>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # formato esperado
    with st.expander("📋 Colunas esperadas no arquivo (df_final)"):
        colunas_esperadas = [
            ("ano", "Arquivo de origem — ex: VIOLBR22"),
            ("uf", "Sigla da UF (SG_UF original)"),
            ("cod_mun6", "Código IBGE 6 dígitos do município"),
            ("nome_regiao", "Norte / Nordeste / Centro-Oeste / Sudeste / Sul"),
            ("faixa_etaria", "menor_18 / 18_24 / 25_34 / ... / 65_mais"),
            ("raca", "Branca / Preta / Amarela / Parda / Indigena / Ignorado"),
            ("escolaridade", "Sem instrucao / Fundamental incompleto / ..."),
            ("sit_conjugal", "Solteira / Casada/Uniao estavel / ..."),
            ("relacao_agressor", "Conjuge / Ex_conjuge / Namorado / ..."),
            ("parceiro_intimo", "0 ou 1 (flag derivada)"),
            ("viol_fisica … viol_negligencia", "Binárias 0/1"),
            ("reincidencia", "Sim / Nao / Ignorado"),
            ("alvo_reincidencia", "1=Sim / 0=Não / NaN=Ignorado"),
            ("agressor_alcool", "Sim / Nao / Ignorado"),
            ("idhm", "IDH municipal (IPEA 2010)"),
            ("indice_vulnerabilidade", "PC1 normalizado 0–1 (PCA do notebook)"),
            ("n_tipos_violencia", "Contagem 0–5"),
            ("n_encaminhamentos", "Contagem 0–4"),
        ]
        st.dataframe(
            pd.DataFrame(colunas_esperadas, columns=["Coluna", "Descrição"]),
            use_container_width=True,
            hide_index=True,
        )

    uploaded_file = st.file_uploader(
        "📁 Carregar arquivo CSV ou Parquet com microdados municipais",
        type=["csv", "parquet"],
        help="Formato gerado pelo notebook limpeza_dataframe.ipynb (df_final)",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".parquet"):
                df_up = pd.read_parquet(uploaded_file)
            else:
                df_up = pd.read_csv(uploaded_file)

            st.success(
                f"✅ **{uploaded_file.name}** carregado — "
                f"{len(df_up):,} registros · {len(df_up.columns)} colunas"
            )

            tab1, tab2 = st.tabs(["Pré-visualização", "Resumo"])
            with tab1:
                st.dataframe(df_up.head(10), use_container_width=True)
            with tab2:
                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    if "uf" in df_up.columns:
                        st.markdown("**Registros por UF:**")
                        st.dataframe(
                            df_up["uf"].value_counts().reset_index()
                            .rename(columns={"uf": "UF", "count": "Registros"}),
                            use_container_width=True, hide_index=True,
                        )
                with col_s2:
                    if "raca" in df_up.columns:
                        st.markdown("**Distribuição por raça:**")
                        st.dataframe(
                            df_up["raca"].value_counts().reset_index()
                            .rename(columns={"raca": "Raça", "count": "Registros"}),
                            use_container_width=True, hide_index=True,
                        )

            st.markdown("---")
            st.markdown("##### Resultados da Análise em Lote")
            st.markdown(
                '<div class="result-placeholder">'
                '🔌 <b>Aguardando integração com a API do backend (Gabriel)</b><br><br>'
                'Aqui serão exibidos:<br>'
                '• Ranking de vulnerabilidade por perfil (UF × raça × escolaridade)<br>'
                '• Mapa choropleth interativo por UF (Plotly)<br>'
                '• Clusters de risco municipal (K-Means — alta / média / baixa vulnerabilidade)<br>'
                '• Comparativo de grupos com maior alvo_reincidencia<br>'
                '• Distribuição do índice_vulnerabilidade (PCA) por região'
                '</div>',
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Erro ao carregar arquivo: {e}")

    else:
        st.markdown(
            '<div class="result-placeholder">'
            '📂 Faça upload de um arquivo <code>CSV</code> ou <code>Parquet</code> '
            'no formato do <code>df_final</code> gerado pelo notebook de limpeza.'
            '</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="nota-metodologica">'
        '⚠️ <b>Nota metodológica (LGPD):</b> Os resultados representam estimativas para '
        '<b>perfis de grupo por município</b>. Nenhum dado pessoal identificável é processado. '
        'O sistema opera sobre distribuições agregadas — hipótese de pesquisa (Art. 7º, IV) '
        'e interesse público (Art. 7º, III) da LGPD. '
        'O upload não é armazenado após o encerramento da sessão.'
        '</div>',
        unsafe_allow_html=True,
    )