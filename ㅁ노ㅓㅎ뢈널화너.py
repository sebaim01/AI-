import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from transformers import pipeline, AutoTokenizer
import torch
from google import genai

# -----------------------------
# 1. 페이지 설정 및 디자인 (CSS)
# -----------------------------
st.set_page_config(
    page_title="동북고 AI 텍스트 판독기 v1.7",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
    html, body, [data-testid="stSidebar"] {
        font-family: 'Noto Sans KR', sans-serif;
    }
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00E5FF, #0083B0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 1rem;
        color: #888;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #1e2430;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        border: 1px solid #2d3748;
    }
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
        border-left: 4px solid #00E5FF;
        padding-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 2. 모델 및 토크나이저 로드 (캐싱)
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "roberta-base-openai-detector"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=tokenizer,
        truncation=True,
        max_length=512,
        device=0 if torch.cuda.is_available() else -1
    )
    return model, tokenizer

detector, tokenizer = load_model()

def split_by_tokens(text, max_tokens=350):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

# -----------------------------
# 3. Gemini 기반 문장 인간화 함수
# -----------------------------
def humanize_text_gemini(api_key, text_to_rewrite):
    try:
        client = genai.Client(api_key=api_key)
        system_instruction = (
            "너는 AI가 작성한 글을 자연스러운 사람이 쓴 글로 바꾸는 전문가야. "
            "아래 규칙을 무조건 지켜줘:\n"
            "1. 너무 격식 차리거나 기계적인 문투(예: ~적, ~에 관하여, 첫째로)를 줄이고 구어체와 문어체를 자연스럽게 섞어라.\n"
            "2. 문장의 길이를 다양하게 해라. (어떤 문장은 길게, 어떤 문장은 짧게)\n"
            "3. 전체적인 맥락과 핵심 의미는 절대 훼손하지 마라.\n"
            "4. 번역기 말투를 없애고 고등학생이나 일반 성인이 직접 고민해서 쓴 듯한 자연스러운 한국어로 작성해라."
        )
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"다음 텍스트를 사람이 쓴 것처럼 자연스럽게 재작성해줘:\n\n{text_to_rewrite}",
            config={
                'system_instruction': system_instruction,
                'temperature': 0.8
            }
        )
        return response.text
    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"

if "rewritten_chunks" not in st.session_state:
    st.session_state.rewritten_chunks = {}

# -----------------------------
# 4. 사이드바 설정
# -----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluent/96/000000/artificial-intelligence.png", width=60)
    st.markdown("### ⚙️ 시스템 설정")
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="AI Studio 키 입력")
    
    if gemini_key:
        st.success("⚡ Gemini 연결 완료")
    else:
        st.caption("⚠️ 문장 인간화 기능을 쓰려면 API 키를 입력하세요.")

    st.divider()
    st.caption("• RoBERTa-base AI Detector")
    st.caption("• Gemini 2.5 Flash 실시간 튜닝")
    st.info("🏫 **동북고등학교**\n\nAI 텍스트 검증기 v1.7")

# -----------------------------
# 5. 🌟 전면 개편된 좌우 분할 레이아웃 🌟
# -----------------------------
# 메인 헤더
st.markdown('<div class="main-title">🏫 동북고 AI 텍스트 판독기</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">RoBERTa 분석 디텍터 & Gemini AI 문장 인간화 시스템</div>', unsafe_allow_html=True)

# 화면을 왼쪽(입력단), 오른쪽(결과 대시보드단)으로 쪼갭니다.
left_col, right_col = st.columns([2, 3], gap="large")

# --- [왼쪽 컬럼: 텍스트 입력 및 제어] ---
with left_col:
    st.markdown('<div class="section-title">📝 분석할 텍스트 입력</div>', unsafe_allow_html=True)
    text = st.text_area(
        "분석 본문 입력창",
        height=450,  # 넉넉하게 입력할 수 있도록 높이 확보
        placeholder="여기에 본문을 붙여넣은 후 아래 '분석 시작' 버튼을 누르세요...",
        label_visibility="collapsed"
    )
    
    submit_button = st.button("🔍 분석 시작", use_container_width=True, type="primary")

    # 분석 트리거 로직
    if submit_button:
        if not text.strip():
            st.warning("텍스트를 입력하세요.")
            st.stop()

        chunks = split_by_tokens(text)

        with st.spinner("🚀 AI가 텍스트를 정밀 분석 중..."):
            results = detector(chunks)
            df = pd.DataFrame([
                {
                    "chunk": i + 1,
                    "text": chunks[i],
                    "label": r["label"],
                    "score": r["score"]
                }
                for i, r in enumerate(results)
            ])
            
        st.session_state.df = df
        st.session_state.chunks = chunks

# --- [오른쪽 컬럼: 결과 출력 대시보드] ---
with right_col:
    if "df" in st.session_state:
        df = st.session_state.df
        chunks = st.session_state.chunks

        avg_score = df["score"].mean()
        max_score = df["score"].max()

        if avg_score > 0.7:
            status_color = "#e74c3c"
            final_status = "🤖 AI 생성 가능성 높음"
        elif avg_score > 0.4:
            status_color = "#f1c40f"
            final_status = "⚠️ 사람이 쓴 글과 AI 글 혼합됨"
        else:
            status_color = "#2ecc71"
            final_status = "👤 사람 작성 가능성 높음"

        st.markdown('<div class="section-title">📊 분석 결과 대시보드</div>', unsafe_allow_html=True)
        
        # 미니 메트릭 & 게이지 가로 배치
        m_col1, m_col2 = st.columns([4, 5])
        with m_col1:
            st.markdown(f"""
            <div class="card" style="height: 140px; display: flex; flex-direction: column; justify-content: center;">
                <span style='color:#aaa; font-size:0.9rem;'>종합 판정</span>
                <b style='color:{status_color}; font-size: 1.25rem; margin-top:5px;'>{final_status}</b>
            </div>
            """, unsafe_allow_html=True)
            
            sub_c1, sub_c2 = st.columns(2)
            sub_c1.metric("평균 AI 점수", f"{avg_score*100:.1f}%")
            sub_c2.metric("최고 위험 점수", f"{max_score*100:.1f}%")
            
        with m_col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_score * 100,
                number={'suffix': "%", 'font': {'size': 20, 'color': "#ffffff"}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#888", 'tickfont': {'size': 10}},
                    'bar': {'color': "#00E5FF"},
                    'bgcolor': "rgba(0,0,0,0)",
                    'steps': [
                        {"range": [0, 40], "color": "rgba(46, 204, 113, 0.15)"},
                        {"range": [40, 70], "color": "rgba(241, 196, 113, 0.15)"},
                        {"range": [70, 100], "color": "rgba(231, 76, 60, 0.15)"}
                    ]
                }
            ))
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=140, 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        st.divider()

        # 추이 그래프 및 데이터프레임 가로 배치
        g_col1, g_col2 = st.columns([1, 1])
        with g_col1:
            st.markdown("💾 **청크별 위험도 추이**")
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=df["chunk"],
                    y=df["score"] * 100,
                    marker_color='#00E5FF',
                    hovertemplate="청크 %{x}<br>위험도: %{y:.1f}%<extra></extra>"
                )
            ])
            fig_bar.update_layout(
                margin=dict(l=5, r=5, t=5, b=5),
                height=150,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title=dict(text="청크", font=dict(size=10, color='#888')), tickmode='linear', tickfont=dict(size=9)),
                yaxis=dict(title=dict(text="(%)", font=dict(size=10, color='#888')), range=[0, 100], tickfont=dict(size=9)),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

        with g_col2:
            st.markdown("📋 **데이터 요약**")
            df_display = df.copy()
            df_display["score"] = (df_display["score"] * 100).round(1).astype(str) + "%"
            df_display["text"] = df_display["text"].str[:15] + "..."
            st.dataframe(
                df_display[["chunk", "score", "text"]].rename(columns={"chunk": "청크", "score": "위험도", "text": "요약"}), 
                use_container_width=True, 
                height=150
            )

        st.divider()

        # 위험 구간 및 Gemini 수정 레이아웃
        st.markdown('<div class="section-title">🚨 AI 의심 구간 튜닝 (Gemini)</div>', unsafe_allow_html=True)
        danger_chunks = df[df["score"] > 0.75]
        
        if danger_chunks.empty:
            st.success("🎉 AI 의심 구간이 없습니다. 안전합니다!")
        else:
            for idx, row in danger_chunks.iterrows():
                with st.container():
                    st.markdown(f"""
                    <div style="background-color: rgba(231, 76, 60, 0.08); border-left: 4px solid #e74c3c; padding: 10px; border-radius: 4px; margin-bottom: 10px;">
                        <span style="color:#e74c3c; font-weight:bold; font-size:0.9rem;">[구간 {row['chunk']}] 위험도 {row['score']*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    c_src, c_dst = st.columns(2)
                    with c_src:
                        st.caption("📋 원문")
                        st.info(row["text"])
                        
                        if not gemini_key:
                            st.warning("🔒 API Key를 등록하세요.")
                        else:
                            if st.button(f"✨ 인간화 적용", key=f"btn_{row['chunk']}", type="secondary", use_container_width=True):
                                with st.spinner("재작성 중..."):
                                    rewritten = humanize_text_gemini(gemini_key, row["text"])
                                    st.session_state.rewritten_chunks[row['chunk']] = rewritten
                                    st.rerun()
                                    
                    with c_dst:
                        st.caption("🪄 Gemini 수정본")
                        if row['chunk'] in st.session_state.rewritten_chunks:
                            st.success(st.session_state.rewritten_chunks[row['chunk']])
                        else:
                            st.text_area("대기 중", placeholder="버튼을 누르면 수정본이 출력됩니다.", height=100, disabled=True, label_visibility="collapsed", key=f"p_{row['chunk']}")
    else:
        # 최초 실행 시 가이드 메시지
        st.info("👈 왼쪽창에 텍스트를 입력하고 '분석 시작' 버튼을 누르면 이 자리에 모니터링 대시보드가 표시됩니다.")