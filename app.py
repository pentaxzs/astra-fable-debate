import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import streamlit as st
from openai import OpenAI
from anthropic import Anthropic


st.set_page_config(
    page_title="Astra × Fable Debate",
    page_icon="⚔️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    .small-note {color:#777; font-size:0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_TOPIC = """AI 시대에는 프로덕트 디자이너보다
프로덕트 매니저의 역할이 더 크게 축소될 것이다."""

OPENAI_MODELS = [
    "gpt-6-astra",
    "o3-pro",
    "o3",
    "o4-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "gpt-4o-mini",
]

ANTHROPIC_MODELS = [
    "claude-fable-5",
    "claude-opus-4-6-20250904",
    "claude-sonnet-4-6-20250514",
    "claude-haiku-4-5-20251001",
    "claude-3-5-sonnet-20241022",
]


def get_api_key(ui_value: str, env_name: str) -> str:
    return (ui_value or "").strip() or os.getenv(env_name, "").strip()


def call_astra(client: OpenAI, prompt: str, model: str, effort: str) -> str:
    response = client.responses.create(
        model=model,
        reasoning={"effort": effort},
        input=prompt,
    )
    return response.output_text


def call_fable(client: Anthropic, prompt: str, model: str, max_tokens: int, effort: str) -> str:
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    # Fable 5 supports output_config.effort. If an older SDK rejects it,
    # retry once without output_config so the app remains usable.
    try:
        response = client.messages.create(
            **kwargs,
            output_config={"effort": effort},
        )
    except TypeError:
        response = client.messages.create(**kwargs)

    texts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    return "\n".join(texts).strip()


ROUND_LABELS = {
    1: "독립 주장",
    2: "상대 주장 비판",
    3: "최종 반론",
}


def build_transcript(topic: str, pro_label: str, con_label: str, results: dict, num_rounds: int) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# Astra × Fable Debate\n",
        f"- 생성 시각: {ts}",
        f"- 토론 주제: {topic}",
        f"- Astra 역할: {pro_label}",
        f"- Fable 역할: {con_label}",
        f"- 라운드 수: {num_rounds}",
        "",
    ]
    for r in range(1, num_rounds + 1):
        label = ROUND_LABELS[r]
        lines.append(f"## ROUND {r} — {label}\n")
        lines.append(f"### Astra\n{results[f'astra_{r}']}\n")
        lines.append(f"### Fable\n{results[f'fable_{r}']}\n")
    return "\n".join(lines)


def _round1_prompts(topic, pro_label, con_label):
    base = "상대방의 의견은 아직 보지 않았다.\n독립적으로 가장 강력한 논거 3개를 제시하라.\n약한 주장이나 수사는 제외하고, 구체적인 인과관계와 전제를 명확히 하라.\n한국어로 답하라."
    astra = f"토론 주제:\n{topic}\n\n당신은 {pro_label}이다.\n{base}"
    fable = f"토론 주제:\n{topic}\n\n당신은 {con_label}이다.\n{base}"
    return astra, fable


def _round2_prompts(topic, pro_label, con_label, results):
    critique = "상대 주장을 다음 구조로 비판하라.\n1. 가장 강한 주장\n2. 가장 약한 주장\n3. 논리적 허점 또는 숨은 전제\n4. 가장 강력한 반론\n\n상대의 주장을 공정하게 재구성한 뒤 비판하고, 허수아비 논증을 피하라.\n한국어로 답하라."
    astra = f"토론 주제:\n{topic}\n\n당신은 {pro_label}이다.\n\nClaude Fable의 주장:\n---\n{results['fable_1']}\n---\n\n{critique}"
    fable = f"토론 주제:\n{topic}\n\n당신은 {con_label}이다.\n\nGPT-6 Astra의 주장:\n---\n{results['astra_1']}\n---\n\n{critique}"
    return astra, fable


def _round3_prompts(topic, pro_label, con_label, results):
    closing = "이제 새로운 주장을 무작정 늘어놓지 말고 상대의 핵심 반론에 직접 답하면서 최종 입장을 제시하라.\n마지막에는 반드시 다음 세 항목을 포함하라.\n- 내가 여전히 맞다고 보는 이유\n- 상대방에게 인정하는 부분\n- 최종 결론\n\n한국어로 답하라."
    astra = f"토론 주제:\n{topic}\n\n당신은 {pro_label}이다.\n\nClaude의 최초 주장:\n---\n{results['fable_1']}\n---\n\nClaude의 반론:\n---\n{results['fable_2']}\n---\n\n{closing}"
    fable = f"토론 주제:\n{topic}\n\n당신은 {con_label}이다.\n\nAstra의 최초 주장:\n---\n{results['astra_1']}\n---\n\nAstra의 반론:\n---\n{results['astra_2']}\n---\n\n{closing}"
    return astra, fable


ROUND_PROMPT_BUILDERS = {
    1: _round1_prompts,
    2: _round2_prompts,
    3: _round3_prompts,
}


def run_debate(topic: str, pro_label: str, con_label: str,
               openai_key: str, anthropic_key: str,
               astra_model: str, fable_model: str,
               astra_effort: str, fable_effort: str,
               max_tokens: int, num_rounds: int) -> dict:
    openai_client = OpenAI(api_key=openai_key)
    anthropic_client = Anthropic(api_key=anthropic_key)

    results = {}

    for r in range(1, num_rounds + 1):
        builder = ROUND_PROMPT_BUILDERS[r]
        if r == 1:
            astra_prompt, fable_prompt = builder(topic, pro_label, con_label)
        else:
            astra_prompt, fable_prompt = builder(topic, pro_label, con_label, results)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(call_astra, openai_client, astra_prompt, astra_model, astra_effort)
            ff = pool.submit(call_fable, anthropic_client, fable_prompt, fable_model, max_tokens, fable_effort)
            results[f"astra_{r}"] = fa.result()
            results[f"fable_{r}"] = ff.result()

    return results


st.title("⚔️ Astra × Fable Debate")
st.caption("GPT-6 Astra와 Claude Fable 5를 3라운드로 토론시키는 로컬 웹앱")

with st.sidebar:
    st.header("API 설정")
    openai_key_ui = st.text_input("OpenAI API Key", type="password", help="비워두면 OPENAI_API_KEY 환경변수를 사용합니다.")
    anthropic_key_ui = st.text_input("Anthropic API Key", type="password", help="비워두면 ANTHROPIC_API_KEY 환경변수를 사용합니다.")

    st.divider()
    st.header("모델 설정")
    astra_model = st.selectbox("OpenAI 모델", options=OPENAI_MODELS, index=0)
    fable_model = st.selectbox("Anthropic 모델", options=ANTHROPIC_MODELS, index=0)
    astra_effort = st.select_slider("Astra reasoning effort", options=["low", "medium", "high", "xhigh", "max"], value="high")
    fable_effort = st.select_slider("Fable effort", options=["low", "medium", "high"], value="high")
    max_tokens = st.number_input("Fable max_tokens", min_value=512, max_value=16000, value=3000, step=256)
    num_rounds = st.radio("토론 라운드 수", options=[1, 2, 3], index=2, horizontal=True,
                          help="1라운드=독립 주장만, 2라운드=+비판, 3라운드=+최종 반론")

    st.caption("API Key는 앱 내부에 저장하지 않습니다.")

left, right = st.columns([2, 1])
with left:
    topic = st.text_area("토론 주제", value=DEFAULT_TOPIC, height=140)
with right:
    pro_label = st.text_input("Astra 역할", value="찬성 측")
    con_label = st.text_input("Fable 역할", value="반대 측")

start = st.button("토론 시작", type="primary", use_container_width=True)

if start:
    openai_key = get_api_key(openai_key_ui, "OPENAI_API_KEY")
    anthropic_key = get_api_key(anthropic_key_ui, "ANTHROPIC_API_KEY")

    if not topic.strip():
        st.error("토론 주제를 입력해주세요.")
        st.stop()
    if not openai_key:
        st.error("OpenAI API Key가 필요합니다. 사이드바에 입력하거나 OPENAI_API_KEY 환경변수를 설정해주세요.")
        st.stop()
    if not anthropic_key:
        st.error("Anthropic API Key가 필요합니다. 사이드바에 입력하거나 ANTHROPIC_API_KEY 환경변수를 설정해주세요.")
        st.stop()

    try:
        with st.status("토론을 진행하고 있습니다…", expanded=True) as status:
            for r in range(1, num_rounds + 1):
                st.write(f"ROUND {r} · {ROUND_LABELS[r]}")
            results = run_debate(
                topic=topic.strip(),
                pro_label=pro_label.strip() or "찬성 측",
                con_label=con_label.strip() or "반대 측",
                openai_key=openai_key,
                anthropic_key=anthropic_key,
                astra_model=astra_model.strip(),
                fable_model=fable_model.strip(),
                astra_effort=astra_effort,
                fable_effort=fable_effort,
                max_tokens=int(max_tokens),
                num_rounds=num_rounds,
            )
            status.update(label="토론 완료", state="complete", expanded=False)

        st.success(f"토론이 완료되었습니다. (총 {num_rounds}라운드, API 호출 {num_rounds * 2}회)")

        tab_names = [f"ROUND {r} · {ROUND_LABELS[r]}" for r in range(1, num_rounds + 1)]
        tabs = st.tabs(tab_names)

        for i, tab in enumerate(tabs):
            r = i + 1
            with tab:
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("GPT-6 Astra" + (" · 최종" if r == 3 else ""))
                    st.markdown(results[f"astra_{r}"])
                with c2:
                    st.subheader("Claude Fable 5" + (" · 최종" if r == 3 else ""))
                    st.markdown(results[f"fable_{r}"])

        transcript = build_transcript(topic.strip(), pro_label, con_label, results, num_rounds)
        st.download_button(
            "토론 결과 Markdown으로 저장",
            data=transcript,
            file_name="astra_fable_debate.md",
            mime="text/markdown",
            use_container_width=True,
        )

    except Exception as exc:
        st.error("API 호출 중 오류가 발생했습니다.")
        st.exception(exc)
