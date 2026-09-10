import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import markdown
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from openai import OpenAI
from anthropic import Anthropic


def md_to_html(text: str) -> str:
    """Convert markdown text to HTML."""
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


st.set_page_config(
    page_title="Astra × Fable Debate",
    page_icon="⚔️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 900px; padding-top: 2rem; padding-bottom: 3rem;}
    .small-note {color:#777; font-size:0.9rem;}

    .debate-card {
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        line-height: 1.7;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .debate-card h1, .debate-card h2, .debate-card h3,
    .debate-card h4, .debate-card h5, .debate-card h6 {
        margin-top: 0.8rem; margin-bottom: 0.4rem;
    }
    .debate-card p { margin-bottom: 0.6rem; }

    .card-astra {
        background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
        border-left: 5px solid #43a047;
        margin-right: 3rem;
    }
    .card-fable {
        background: linear-gradient(135deg, #ede7f6 0%, #e8eaf6 100%);
        border-right: 5px solid #5e35b1;
        border-left: none;
        margin-left: 3rem;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
        margin-bottom: 0.6rem;
        letter-spacing: 0.02em;
    }
    .badge-astra { background: #43a047; color: #fff; }
    .badge-fable { background: #5e35b1; color: #fff; }

    .round-divider {
        text-align: center;
        margin: 2.5rem 0 1.5rem 0;
        position: relative;
    }
    .round-divider::before {
        content: "";
        position: absolute;
        top: 50%;
        left: 0; right: 0;
        border-top: 2px dashed #bbb;
    }
    .round-divider span {
        background: #fff;
        padding: 0.3rem 1.2rem;
        font-weight: 700;
        font-size: 1rem;
        color: #555;
        position: relative;
        border-radius: 20px;
        border: 2px solid #bbb;
    }

    .arrow-down {
        text-align: center;
        font-size: 1.4rem;
        color: #999;
        margin: 0.4rem 0;
    }

    .card-judge {
        background: linear-gradient(135deg, #fff8e1 0%, #fff3e0 100%);
        border-left: 5px solid #f9a825;
        border-right: 5px solid #f9a825;
    }
    .badge-judge { background: #f9a825; color: #fff; }

    .verdict-box {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        padding: 1rem;
        margin: 0.8rem 0;
        border-radius: 8px;
    }
    .verdict-astra { background: #43a047; color: #fff; }
    .verdict-fable { background: #5e35b1; color: #fff; }
    .verdict-draw { background: #757575; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)



def render_debate_card_html(model: str, side: str, content_md: str):
    """Render a full debate card, converting markdown content to HTML."""
    badge_map = {
        "astra": ("badge-astra", "card-astra"),
        "fable": ("badge-fable", "card-fable"),
        "judge": ("badge-judge", "card-judge"),
    }
    badge_cls, card_cls = badge_map.get(side, ("badge-judge", "card-judge"))
    content_html = md_to_html(content_md)
    st.markdown(
        f'<div class="debate-card {card_cls}">'
        f'<span class="badge {badge_cls}">{model}</span>'
        f"{content_html}"
        f"</div>",
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
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-20250115",
]


def get_api_key(ui_value: str, env_name: str) -> str:
    return (ui_value or "").strip() or os.getenv(env_name, "").strip()


REASONING_MODELS = {"o3-pro", "o3", "o4-mini"}


def call_astra(client: OpenAI, prompt: str, model: str, effort: str) -> str:
    kwargs = {"model": model, "input": prompt}
    if model in REASONING_MODELS:
        kwargs["reasoning"] = {"effort": effort}
    response = client.responses.create(**kwargs)
    return response.output_text


def call_fable(client: Anthropic, prompt: str, model: str, max_tokens: int, effort: str) -> str:
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }

    # output_config.effort is only supported by some models.
    # Fall back gracefully if the API or SDK rejects it.
    try:
        response = client.messages.create(
            **kwargs,
            output_config={"effort": effort},
        )
    except (TypeError, Exception) as exc:
        if "effort" in str(exc).lower() or isinstance(exc, TypeError):
            response = client.messages.create(**kwargs)
        else:
            raise

    texts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    return "\n".join(texts).strip()


JUDGE_PROVIDERS = ["OpenAI", "Anthropic"]


def build_judge_prompt(topic: str, pro_label: str, con_label: str,
                       results: dict, num_rounds: int) -> str:
    transcript_parts = []
    for r in range(1, num_rounds + 1):
        label = ROUND_LABELS_MAP.get(r, f"라운드 {r}")
        transcript_parts.append(f"## ROUND {r} — {label}")
        transcript_parts.append(f"### Astra ({pro_label})\n{results[f'astra_{r}']}")
        transcript_parts.append(f"### Fable ({con_label})\n{results[f'fable_{r}']}")
    transcript = "\n\n".join(transcript_parts)

    return f"""당신은 공정한 제3자 AI 심판(Judge)이다.
아래는 "{topic}"에 대해 Astra({pro_label})와 Fable({con_label})이 {num_rounds}라운드에 걸쳐 진행한 토론 기록이다.

{transcript}

---

위 토론을 분석하여 아래 세 항목을 반드시 포함하여 한국어로 답하라.

### 1. 판정
"Astra 승", "Fable 승", "무승부" 중 하나를 명확히 선언하라.

### 2. 판정 이유
- 각 측의 가장 강력했던 논거와 가장 약했던 논거를 짚어라.
- 논리적 일관성, 근거의 구체성, 반론 대응력을 기준으로 평가하라.
- 왜 해당 측이 이겼는지 (또는 무승부인지) 핵심 근거를 설명하라.
{f"- 이 토론은 {num_rounds}라운드만 진행되었으므로 제한된 정보 하에서의 판정임을 밝혀라." if num_rounds < 3 else ""}

### 3. 의사결정 권고
이 토론 주제에 대해 실무 의사결정자에게 어떤 입장을 취하면 좋을지 구체적으로 권고하라.
양측의 타당한 논거를 통합하여 현실적인 행동 방안을 제시하라.
"""


def call_judge(provider: str, model: str, prompt: str,
               openai_key: str, anthropic_key: str, max_tokens: int) -> str:
    if provider == "OpenAI":
        client = OpenAI(api_key=openai_key)
        kwargs = {"model": model, "input": prompt}
        if model in REASONING_MODELS:
            kwargs["reasoning"] = {"effort": "high"}
        response = client.responses.create(**kwargs)
        return response.output_text
    else:
        client = Anthropic(api_key=anthropic_key)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        texts = [
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text" and getattr(block, "text", None)
        ]
        return "\n".join(texts).strip()


def parse_verdict(judge_text: str) -> str:
    """Extract verdict keyword from judge response."""
    lower = judge_text.lower()
    # Search in the first ~500 chars where the verdict declaration usually is
    verdict_section = lower[:500]
    if "astra 승" in verdict_section or "astra가 승" in verdict_section or "astra의 승" in verdict_section:
        return "astra"
    if "fable 승" in verdict_section or "fable가 승" in verdict_section or "fable의 승" in verdict_section:
        return "fable"
    if "무승부" in verdict_section:
        return "draw"
    # Fallback: search entire text
    if "astra 승" in lower:
        return "astra"
    if "fable 승" in lower:
        return "fable"
    return "draw"


ROUND_LABELS_MAP = {
    1: "독립 주장",
    2: "상대 주장 비판",
    3: "최종 반론",
}

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

# --- Load saved API keys from browser localStorage ---
if "keys_loaded" not in st.session_state:
    loaded = streamlit_js_eval(
        js_expressions='JSON.stringify({o:localStorage.getItem("afd_openai_key")||"",a:localStorage.getItem("afd_anthropic_key")||""})',
        key="load_keys",
    )
    if loaded is not None and loaded != 0:
        data = json.loads(loaded)
        st.session_state.saved_openai_key = data["o"]
        st.session_state.saved_anthropic_key = data["a"]
        st.session_state.keys_loaded = True
        st.rerun()

with st.sidebar:
    st.header("API 설정")
    openai_key_ui = st.text_input(
        "OpenAI API Key",
        value=st.session_state.get("saved_openai_key", ""),
        type="password",
        help="비워두면 OPENAI_API_KEY 환경변수를 사용합니다.",
    )
    anthropic_key_ui = st.text_input(
        "Anthropic API Key",
        value=st.session_state.get("saved_anthropic_key", ""),
        type="password",
        help="비워두면 ANTHROPIC_API_KEY 환경변수를 사용합니다.",
    )

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("키 저장", use_container_width=True):
            ok_js = json.dumps(openai_key_ui)
            ak_js = json.dumps(anthropic_key_ui)
            st.components.v1.html(
                f'<script>localStorage.setItem("afd_openai_key",{ok_js});localStorage.setItem("afd_anthropic_key",{ak_js});</script>',
                height=0,
            )
            st.session_state.saved_openai_key = openai_key_ui
            st.session_state.saved_anthropic_key = anthropic_key_ui
            st.toast("API Key가 브라우저에 저장되었습니다.")
    with btn_col2:
        if st.button("키 삭제", use_container_width=True):
            st.components.v1.html(
                '<script>localStorage.removeItem("afd_openai_key");localStorage.removeItem("afd_anthropic_key");</script>',
                height=0,
            )
            st.session_state.saved_openai_key = ""
            st.session_state.saved_anthropic_key = ""
            st.toast("저장된 API Key가 삭제되었습니다.")

    st.divider()
    st.header("모델 설정")
    astra_model = st.selectbox("OpenAI 모델", options=OPENAI_MODELS, index=0)
    astra_model_custom = st.text_input("또는 직접 입력 (OpenAI)", placeholder="예: gpt-4o")
    fable_model = st.selectbox("Anthropic 모델", options=ANTHROPIC_MODELS, index=0)
    fable_model_custom = st.text_input("또는 직접 입력 (Anthropic)", placeholder="예: claude-sonnet-4-20250514")
    astra_effort = st.select_slider("Astra reasoning effort", options=["low", "medium", "high", "xhigh", "max"], value="high")
    fable_effort = st.select_slider("Fable effort", options=["low", "medium", "high"], value="high")
    max_tokens = st.number_input("Fable max_tokens", min_value=512, max_value=16000, value=3000, step=256)
    num_rounds = st.radio("토론 라운드 수", options=[1, 2, 3], index=2, horizontal=True,
                          help="1라운드=독립 주장만, 2라운드=+비판, 3라운드=+최종 반론")

    st.divider()
    st.header("Judge 설정")
    judge_provider = st.selectbox("Judge 제공자", options=JUDGE_PROVIDERS, index=1)
    if judge_provider == "OpenAI":
        judge_model = st.selectbox("Judge 모델", options=OPENAI_MODELS, index=0, key="judge_model_openai")
    else:
        judge_model = st.selectbox("Judge 모델", options=ANTHROPIC_MODELS, index=0, key="judge_model_anthropic")
    judge_model_custom = st.text_input("또는 직접 입력 (Judge)", placeholder="예: gpt-4o, claude-sonnet-4-20250514")

    st.caption("API Key는 브라우저 localStorage에만 저장됩니다.")

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
                astra_model=(astra_model_custom.strip() or astra_model).strip(),
                fable_model=(fable_model_custom.strip() or fable_model).strip(),
                astra_effort=astra_effort,
                fable_effort=fable_effort,
                max_tokens=int(max_tokens),
                num_rounds=num_rounds,
            )
            st.write("Judge 판정 중…")
            judge_actual_model = (judge_model_custom.strip() or judge_model).strip()
            judge_prompt = build_judge_prompt(
                topic.strip(),
                pro_label.strip() or "찬성 측",
                con_label.strip() or "반대 측",
                results, num_rounds,
            )
            judge_result = call_judge(
                provider=judge_provider,
                model=judge_actual_model,
                prompt=judge_prompt,
                openai_key=openai_key,
                anthropic_key=anthropic_key,
                max_tokens=int(max_tokens),
            )
            status.update(label="토론 및 판정 완료", state="complete", expanded=False)

        total_calls = num_rounds * 2 + 1
        st.success(f"토론이 완료되었습니다. (총 {num_rounds}라운드 + Judge 판정, API 호출 {total_calls}회)")

        astra_display = (astra_model_custom.strip() or astra_model).strip()
        fable_display = (fable_model_custom.strip() or fable_model).strip()

        for r in range(1, num_rounds + 1):
            label = ROUND_LABELS[r]
            icon = {1: "\u2694\ufe0f", 2: "\U0001f50d", 3: "\U0001f3c6"}.get(r, "")
            st.markdown(
                f'<div class="round-divider"><span>{icon} ROUND {r} &mdash; {label}</span></div>',
                unsafe_allow_html=True,
            )

            # Astra speaks
            render_debate_card_html(
                f"{astra_display} ({pro_label.strip() or '찬성 측'})",
                "astra",
                results[f"astra_{r}"],
            )

            # Arrow indicating response
            st.markdown('<div class="arrow-down">\u2b07\ufe0f</div>', unsafe_allow_html=True)

            # Fable responds
            render_debate_card_html(
                f"{fable_display} ({con_label.strip() or '반대 측'})",
                "fable",
                results[f"fable_{r}"],
            )

        # --- Judge Verdict ---
        verdict = parse_verdict(judge_result)
        verdict_labels = {
            "astra": ("Astra 승", "verdict-astra"),
            "fable": ("Fable 승", "verdict-fable"),
            "draw": ("무승부", "verdict-draw"),
        }
        verdict_text, verdict_cls = verdict_labels[verdict]

        st.markdown(
            '<div class="round-divider"><span>\u2696\ufe0f JUDGE \u2014 \ud310\uc815</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="verdict-box {verdict_cls}">{verdict_text}</div>',
            unsafe_allow_html=True,
        )
        render_debate_card_html(
            f"Judge \u00b7 {judge_actual_model}",
            "judge",
            judge_result,
        )

        # --- Transcript with Judge ---
        transcript = build_transcript(topic.strip(), pro_label, con_label, results, num_rounds)
        transcript += f"\n## Judge 판정 ({judge_actual_model})\n\n{judge_result}\n"
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
