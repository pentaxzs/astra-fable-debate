import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from string import Template

import markdown
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from openai import OpenAI
from anthropic import Anthropic


def md_to_html(text: str) -> str:
    """Convert markdown text to HTML."""
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


REPO_URL = "https://github.com/pentaxzs/astra-fable-debate"

# The fork/star badge on Streamlit Community Cloud is hosting chrome, not
# something app code can render. What the app does control is the ⋮ menu, so
# point it at the repo.
st.set_page_config(
    page_title="Astra × Fable Debate",
    page_icon="⚔️",
    layout="wide",
    menu_items={
        "Get Help": REPO_URL,
        "Report a bug": f"{REPO_URL}/issues/new",
        "About": (
            "**AI Debate Arena** — OpenAI와 Anthropic 모델을 N라운드로 토론시키고 "
            f"AI Judge가 판정합니다.\n\n[GitHub에서 소스 보기]({REPO_URL})"
        ),
    },
)

# Colours that have to flip with the theme. st.context.theme.type reports the
# theme actually resolved in the browser, so this follows both the OS setting
# and an explicit pick in Streamlit's settings menu.
_DARK = st.context.theme.type == "dark"
_T = {
    # topic cards
    "card_bg": "#1a1d24" if _DARK else "#ffffff",
    "card_fg": "#e6e6e6" if _DARK else "#222222",
    "card_line": "#3d434f" if _DARK else "#222222",
    "card_sel_bg": "#e6e6e6" if _DARK else "#222222",
    "card_sel_fg": "#16181d" if _DARK else "#ffffff",
    # debate transcript
    "note_fg": "#9aa0aa" if _DARK else "#777777",
    "divider_bg": "#0e1117" if _DARK else "#ffffff",
    "divider_fg": "#aab0ba" if _DARK else "#555555",
    "divider_line": "#454b57" if _DARK else "#bbbbbb",
    "card_text": "#e4e6ea" if _DARK else "#1a1a1a",
    "card_heading": "#f2f4f7" if _DARK else "#111111",
    "card_strong": "#ffffff" if _DARK else "#000000",
    "astra_bg": ("linear-gradient(135deg, #16251a 0%, #1a2b1d 100%)" if _DARK
                 else "linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%)"),
    "fable_bg": ("linear-gradient(135deg, #1e1a2e 0%, #1c1e33 100%)" if _DARK
                 else "linear-gradient(135deg, #ede7f6 0%, #e8eaf6 100%)"),
    "judge_bg": ("linear-gradient(135deg, #2a2417 0%, #2b2519 100%)" if _DARK
                 else "linear-gradient(135deg, #fff8e1 0%, #fff3e0 100%)"),
    "shadow": "rgba(0,0,0,0.35)" if _DARK else "rgba(0,0,0,0.06)",
    "banner_fg": "#9aa0aa" if _DARK else "#666666",
    "cta_glow": "rgba(255,75,75,0.28)" if _DARK else "rgba(255,75,75,0.35)",
}

st.markdown(
    Template(
    """
    <style>
    .block-container {max-width: 900px; padding-top: 1rem; padding-bottom: 3rem;}

    /* Start button. The label lives in a markdown <p> inside the button, so it
       has to be targeted directly: font-size on the button itself never
       reaches the text. */
    [data-testid="stBaseButton-primary"] {
        padding: 1.1rem 1.2rem !important;
        box-shadow: 0 4px 14px $cta_glow !important;
    }
    [data-testid="stBaseButton-primary"] p {
        font-size: 1.45rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.01em !important;
        line-height: 1.25 !important;
    }

    /* Widget labels + inputs: slightly larger for mobile readability */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    label[data-testid="stWidgetLabel"] {
        font-size: 0.95rem !important;
    }
    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input {
        font-size: 1.02rem !important;
        line-height: 1.55 !important;
    }

    /* Pills → horizontal scroll card strip (st.pills is rendered as a
       button group; the inner group is the scroll container) */
    [data-testid="stButtonGroup"]:has(button[data-variant="pills"]) > div:last-child {
        flex-wrap: nowrap !important;
        overflow-x: auto !important;
        overflow-y: hidden !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
        -webkit-overflow-scrolling: touch !important;
        scroll-snap-type: x proximity !important;
        gap: 8px !important;
        padding-bottom: 2px !important;
    }
    [data-testid="stButtonGroup"]:has(button[data-variant="pills"]) > div:last-child::-webkit-scrollbar {
        display: none !important;
    }

    /* Topic cards. Each card gets its own width, computed in Python so the
       label wraps to exactly CARD_LINES lines; the per-card overrides are
       injected further down as :nth-of-type rules. The width here is only the
       fallback for a card the Python pass did not size. */
    [data-testid="stButtonGroup"] button[data-variant="pills"] {
        flex: 0 0 auto !important;
        width: 140px !important;
        min-width: 0 !important;
        max-width: none !important;
        height: 62px !important;
        min-height: 62px !important;
        white-space: normal !important;
        scroll-snap-align: start !important;
        border: 1.5px solid $card_line !important;
        border-radius: 16px !important;
        background: $card_bg !important;
        color: $card_fg !important;
        padding: 4px 10px !important;
        font-size: 0.91rem !important;
        align-items: center !important;
        justify-content: flex-start !important;
        text-align: left !important;
        overflow: hidden !important;
    }
    [data-testid="stButtonGroup"] button[data-variant="pills"][aria-checked="true"],
    [data-testid="stButtonGroup"] button[data-variant="pills"][aria-selected="true"] {
        background: $card_sel_bg !important;
        color: $card_sel_fg !important;
    }
    /* Make every wrapper between the button and the text fill the card so the
       label can wrap and top-align instead of being centred on one line. */
    [data-testid="stButtonGroup"] button[data-variant="pills"] > div,
    [data-testid="stButtonGroup"] button[data-variant="pills"] > div > span,
    [data-testid="stButtonGroup"] button[data-variant="pills"] [data-testid="stMarkdownContainer"] {
        width: 100% !important;
        align-items: flex-start !important;
        justify-content: flex-start !important;
        text-align: left !important;
    }
    [data-testid="stButtonGroup"] button[data-variant="pills"] p,
    [data-testid="stButtonGroup"] button[data-variant="pills"] [data-testid="stMarkdownContainer"] {
        font-size: 0.91rem !important;
    }
    [data-testid="stButtonGroup"] button[data-variant="pills"] p {
        margin: 0 !important;
        line-height: 1.15 !important;
        white-space: normal !important;
        word-break: keep-all !important;
        overflow-wrap: anywhere !important;
        display: -webkit-box !important;
        -webkit-box-orient: vertical !important;
        -webkit-line-clamp: 3 !important;
        overflow: hidden !important;
    }

    .small-note {color:$note_fg; font-size:0.95rem;}

    .debate-card {
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        line-height: 1.7;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px $shadow;
        color: $card_text !important;
    }
    .debate-card h1, .debate-card h2, .debate-card h3 {
        font-size: 1.05rem !important;
        margin-top: 0.8rem; margin-bottom: 0.4rem;
        color: $card_heading !important;
    }
    .debate-card h4, .debate-card h5, .debate-card h6 {
        font-size: 0.95rem !important;
        margin-top: 0.6rem; margin-bottom: 0.3rem;
        color: $card_heading !important;
    }
    .debate-card p { margin-bottom: 0.6rem; color: $card_text !important; }
    .debate-card li { color: $card_text !important; }
    .debate-card strong { color: $card_strong !important; }

    .card-astra {
        background: $astra_bg;
        border-left: 5px solid #43a047;
        margin-right: 3rem;
    }
    .card-fable {
        background: $fable_bg;
        border-right: 5px solid #5e35b1;
        border-left: none;
        margin-left: 3rem;
    }

    .badge {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.9rem;
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
        border-top: 2px dashed $divider_line;
    }
    .round-divider span {
        background: $divider_bg;
        padding: 0.3rem 1.2rem;
        font-weight: 700;
        font-size: 1rem;
        color: $divider_fg;
        position: relative;
        border-radius: 20px;
        border: 2px solid $divider_line;
    }

    .arrow-down {
        text-align: center;
        font-size: 1.4rem;
        color: #999;
        margin: 0.4rem 0;
    }

    .card-judge {
        background: $judge_bg;
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
    """
    ).substitute(_T),
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


# --- Topic card sizing -------------------------------------------------------
# Each topic card is given its own width so the label wraps to exactly
# CARD_LINES lines: no clipping, no dead space under the text. Widths are
# derived from a character-width model measured against the real card font
# (Source Sans + the system Korean face) at CARD_FONT_PX. Predicted vs measured
# width for the longest topic: 487.0px vs 486.5px.

CARD_LINES = 3
CARD_FONT_PX = 14.56      # 0.91rem
CARD_PADDING_PX = 20      # 10px each side
CARD_BORDER_PX = 3        # 1.5px each side
CARD_SLACK_PX = 3         # absorbs sub-pixel drift between the model and the browser
CARD_MIN_PX = 104
CARD_MAX_PX = 260

_EM_UPPER = {"A": .6675, "B": .6675, "C": .7223, "D": .7223, "E": .6675, "F": .6117,
             "G": .778, "H": .7223, "I": .278, "J": .5001, "K": .6675, "L": .5569,
             "M": .8339, "N": .7223, "O": .778, "P": .6675, "Q": .778, "R": .7223,
             "S": .6675, "T": .6117, "U": .7223, "V": .6675, "W": .9444, "X": .6675,
             "Y": .6675, "Z": .6117}
_EM_LOWER = {"a": .5569, "b": .5569, "c": .5001, "d": .5569, "e": .5569, "f": .278,
             "g": .5569, "h": .5569, "i": .2232, "j": .2232, "k": .5001, "l": .2232,
             "m": .8339, "n": .5569, "o": .5569, "p": .5569, "q": .5569, "r": .3337,
             "s": .5001, "t": .278, "u": .5569, "v": .5001, "w": .7223, "x": .5001,
             "y": .5001, "z": .5001}
_EM_PUNCT = {".": .278, ",": .278, "·": .278, "(": .3337, ")": .3337, "%": .8896,
             "-": .3337, "?": .5001, "!": .278, "'": .2, '"': .35, ":": .278, ";": .278}
_EM_HANGUL, _EM_DIGIT, _EM_SPACE, _EM_FALLBACK = .866, .5569, .278, .6


def _char_em(ch: str) -> float:
    if "\uac00" <= ch <= "\ud7a3" or "\u1100" <= ch <= "\u11ff" or "\u3130" <= ch <= "\u318f":
        return _EM_HANGUL
    if ch.isdigit():
        return _EM_DIGIT
    if ch == " ":
        return _EM_SPACE
    return _EM_UPPER.get(ch) or _EM_LOWER.get(ch) or _EM_PUNCT.get(ch) or _EM_FALLBACK


def _text_em(text: str) -> float:
    return sum(_char_em(c) for c in text)


def _line_count(text: str, width_em: float) -> int:
    """Greedy word wrap, matching CSS word-break: keep-all (breaks at spaces only)."""
    lines, cur = 1, 0.0
    for word in text.split(" "):
        word_em = _text_em(word)
        if cur == 0:
            cur = word_em
        elif cur + _EM_SPACE + word_em <= width_em:
            cur += _EM_SPACE + word_em
        else:
            lines, cur = lines + 1, word_em
    return lines


def card_width_px(text: str) -> int:
    """Narrowest card that still wraps `text` to CARD_LINES lines or fewer."""
    lo = max(_text_em(w) for w in text.split(" "))   # never narrower than the longest word
    hi = _text_em(text)
    for _ in range(40):
        mid = (lo + hi) / 2
        if _line_count(text, mid) <= CARD_LINES:
            hi = mid
        else:
            lo = mid
    px = hi * CARD_FONT_PX + CARD_SLACK_PX + CARD_PADDING_PX + CARD_BORDER_PX
    return min(CARD_MAX_PX, max(CARD_MIN_PX, math.ceil(px)))


DEBATE_TOPICS = [
    "AI 시대에는 프로덕트 디자이너보다 프로덕트 매니저의 역할이 더 크게 축소될 것이다.",
    "AGI는 2030년 이전에 실현될 것이다.",
    "AI는 결국 인류 존립을 위협하는 존재가 될 것이다.",
    "5년 내에 AI가 주니어 개발자를 완전히 대체할 것이다.",
    "5년 내에 AI가 주니어 디자이너를 완전히 대체할 것이다.",
    "5년 내에 AI가 주니어 기획자를 완전히 대체할 것이다.",
    "AI 창작물에도 저작권을 인정해야 한다.",
    "AI 규제는 혁신을 저해하므로 최소화해야 한다.",
    "AI 시대에 대학 교육은 더 이상 필요하지 않다.",
    "자율주행차의 사고 책임은 탑승자가 아닌 제조사에 있다.",
    "AI가 의사의 진단 역할을 대체하는 것은 바람직하다.",
    "원격근무는 AI 시대에 오히려 줄어들 것이다.",
]

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
    "claude-fable-5-1",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-haiku-4-5-20251001",
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
                       results: dict, num_rounds: int,
                       openai_name: str = "", anthropic_name: str = "") -> str:
    transcript_parts = []
    for r in range(1, num_rounds + 1):
        label = ROUND_LABELS.get(r, f"라운드 {r}")
        transcript_parts.append(f"## ROUND {r} — {label}")
        transcript_parts.append(f"### {openai_name} ({pro_label})\n{results[f'astra_{r}']}")
        transcript_parts.append(f"### {anthropic_name} ({con_label})\n{results[f'fable_{r}']}")
    transcript = "\n\n".join(transcript_parts)

    return f"""당신은 공정한 제3자 AI 심판(Judge)이다.
아래는 "{topic}"에 대해 {openai_name}({pro_label})과 {anthropic_name}({con_label})이 {num_rounds}라운드에 걸쳐 진행한 토론 기록이다.

{transcript}

---

위 토론을 분석하여 아래 세 항목을 반드시 포함하여 한국어로 답하라.

### 1. 판정
"{pro_label} 승", "{con_label} 승", "무승부" 중 하나를 명확히 선언하라.
판정 시 모델명({openai_name}, {anthropic_name})이 아닌 역할명({pro_label}, {con_label})으로 표기하라.

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


def parse_verdict(judge_text: str, pro_label: str, con_label: str) -> str:
    """Extract verdict keyword from judge response based on role labels."""
    lower = judge_text.lower()
    pro = pro_label.lower()
    con = con_label.lower()
    # Search in the first ~500 chars where the verdict declaration usually is
    verdict_section = lower[:500]
    for keyword in [f"{pro} 승", f"{pro}이 승", f"{pro}의 승", f"{pro}가 승", f"{pro} 측 승", f"{pro}측 승"]:
        if keyword in verdict_section:
            return "pro"
    for keyword in [f"{con} 승", f"{con}이 승", f"{con}의 승", f"{con}가 승", f"{con} 측 승", f"{con}측 승"]:
        if keyword in verdict_section:
            return "con"
    if "무승부" in verdict_section:
        return "draw"
    # Fallback: search entire text
    for keyword in [f"{pro} 승", f"{pro}의 승"]:
        if keyword in lower:
            return "pro"
    for keyword in [f"{con} 승", f"{con}의 승"]:
        if keyword in lower:
            return "con"
    return "draw"


ROUND_LABELS = {
    1: "독립 주장",
    2: "상대 주장 비판",
    3: "최종 반론",
}


def build_transcript(topic: str, pro_label: str, con_label: str, results: dict, num_rounds: int,
                     openai_name: str = "", anthropic_name: str = "") -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# AI Debate Arena\n",
        f"- 생성 시각: {ts}",
        f"- 토론 주제: {topic}",
        f"- OpenAI ({openai_name}): {pro_label}",
        f"- Anthropic ({anthropic_name}): {con_label}",
        f"- 라운드 수: {num_rounds}",
        "",
    ]
    for r in range(1, num_rounds + 1):
        label = ROUND_LABELS[r]
        lines.append(f"## ROUND {r} — {label}\n")
        lines.append(f"### {openai_name} ({pro_label})\n{results[f'astra_{r}']}\n")
        lines.append(f"### {anthropic_name} ({con_label})\n{results[f'fable_{r}']}\n")
    return "\n".join(lines)


def _round1_prompts(topic, pro_label, con_label):
    base = "상대방의 의견은 아직 보지 않았다.\n독립적으로 가장 강력한 논거 3개를 제시하라.\n약한 주장이나 수사는 제외하고, 구체적인 인과관계와 전제를 명확히 하라.\n한국어로 답하라."
    astra = f"토론 주제:\n{topic}\n\n당신은 {pro_label}이다.\n{base}"
    fable = f"토론 주제:\n{topic}\n\n당신은 {con_label}이다.\n{base}"
    return astra, fable


def _round2_prompts(topic, pro_label, con_label, results, openai_name="", anthropic_name=""):
    critique = "상대 주장을 다음 구조로 비판하라.\n1. 가장 강한 주장\n2. 가장 약한 주장\n3. 논리적 허점 또는 숨은 전제\n4. 가장 강력한 반론\n\n상대의 주장을 공정하게 재구성한 뒤 비판하고, 허수아비 논증을 피하라.\n한국어로 답하라."
    astra = f"토론 주제:\n{topic}\n\n당신은 {pro_label}이다.\n\n상대측({anthropic_name}, {con_label})의 주장:\n---\n{results['fable_1']}\n---\n\n{critique}"
    fable = f"토론 주제:\n{topic}\n\n당신은 {con_label}이다.\n\n상대측({openai_name}, {pro_label})의 주장:\n---\n{results['astra_1']}\n---\n\n{critique}"
    return astra, fable


def _round3_prompts(topic, pro_label, con_label, results, openai_name="", anthropic_name=""):
    closing = "이제 새로운 주장을 무작정 늘어놓지 말고 상대의 핵심 반론에 직접 답하면서 최종 입장을 제시하라.\n마지막에는 반드시 다음 세 항목을 포함하라.\n- 내가 여전히 맞다고 보는 이유\n- 상대방에게 인정하는 부분\n- 최종 결론\n\n한국어로 답하라."
    astra = f"토론 주제:\n{topic}\n\n당신은 {pro_label}이다.\n\n상대측({anthropic_name})의 최초 주장:\n---\n{results['fable_1']}\n---\n\n상대측의 반론:\n---\n{results['fable_2']}\n---\n\n{closing}"
    fable = f"토론 주제:\n{topic}\n\n당신은 {con_label}이다.\n\n상대측({openai_name})의 최초 주장:\n---\n{results['astra_1']}\n---\n\n상대측의 반론:\n---\n{results['astra_2']}\n---\n\n{closing}"
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
            astra_prompt, fable_prompt = builder(topic, pro_label, con_label, results,
                                                  openai_name=astra_model, anthropic_name=fable_model)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fa = pool.submit(call_astra, openai_client, astra_prompt, astra_model, astra_effort)
            ff = pool.submit(call_fable, anthropic_client, fable_prompt, fable_model, max_tokens, fable_effort)
            results[f"astra_{r}"] = fa.result()
            results[f"fable_{r}"] = ff.result()

    return results


st.markdown(
    '<h1 style="font-size:clamp(1.6rem, 9.6vw, 2.6rem);line-height:1.25;margin:0 0 0.45rem 0;'
    'white-space:nowrap;overflow:visible;">⚔️ AI Debate Arena</h1>'
    '<p style="color:#888;font-size:0.95rem;line-height:1.5;margin:0 0 1rem 0;">'
    'OpenAI vs Anthropic 모델을 N라운드로 토론시키고 AI Judge가 판정하는 웹앱</p>',
    unsafe_allow_html=True,
)

# --- Persistent settings via a single cookie (JSON blob) ---
# All reads AND writes go through streamlit_js_eval (runs in the main page
# context). st.components.v1.html creates a sandboxed iframe whose cookies
# are invisible to the main page — that's why the old approach failed.

_COOKIE_NAME = "afd_settings"

_COOKIE_READ_JS = f"""
(function(){{
    var m = document.cookie.match(new RegExp('(?:^|; ){_COOKIE_NAME}=([^;]*)'));
    return m ? decodeURIComponent(m[1]) : '{{}}'
}})()
"""

if "keys_loaded" not in st.session_state:
    loaded = streamlit_js_eval(js_expressions=_COOKIE_READ_JS, key="load_cookie")
    if loaded is not None and loaded != 0:
        try:
            data = json.loads(loaded)
        except (json.JSONDecodeError, TypeError):
            data = {}
        st.session_state.saved_openai_key = data.get("openai_key", "")
        st.session_state.saved_anthropic_key = data.get("anthropic_key", "")
        st.session_state.saved_astra_model = data.get("astra_model", "")
        st.session_state.saved_fable_model = data.get("fable_model", "")
        st.session_state.saved_judge_provider = data.get("judge_provider", "")
        st.session_state.saved_judge_model = data.get("judge_model", "")
        st.session_state.keys_loaded = True
        st.rerun()

# If a pending save was requested, write the cookie now via streamlit_js_eval
if st.session_state.get("_pending_save"):
    blob = json.dumps(st.session_state["_pending_save"])
    save_js = (
        f'document.cookie="{_COOKIE_NAME}="'
        f'+encodeURIComponent({json.dumps(blob)})'
        f'+";max-age=31536000;path=/;SameSite=Lax"'
    )
    streamlit_js_eval(js_expressions=save_js, key="write_cookie")
    del st.session_state["_pending_save"]

# Determine if API keys are already saved (for auto-collapse)
_has_saved_keys = bool(
    st.session_state.get("saved_openai_key") or st.session_state.get("saved_anthropic_key")
)


def _schedule_cookie_save():
    """Collect all saved_* values and schedule a cookie write on next rerun."""
    st.session_state["_pending_save"] = {
        "openai_key": st.session_state.get("saved_openai_key", ""),
        "anthropic_key": st.session_state.get("saved_anthropic_key", ""),
        "astra_model": st.session_state.get("saved_astra_model", ""),
        "fable_model": st.session_state.get("saved_fable_model", ""),
        "judge_provider": st.session_state.get("saved_judge_provider", ""),
        "judge_model": st.session_state.get("saved_judge_model", ""),
    }


def _schedule_cookie_delete():
    """Schedule a cookie deletion on next rerun."""
    st.session_state["_pending_save"] = {}


with st.sidebar:
    # --- API 설정 (collapsible) ---
    with st.expander("API 설정", expanded=not _has_saved_keys):
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
                st.session_state.saved_openai_key = openai_key_ui
                st.session_state.saved_anthropic_key = anthropic_key_ui
                _schedule_cookie_save()
                st.toast("API Key가 저장되었습니다.")
                st.rerun()
        with btn_col2:
            if st.button("키 삭제", use_container_width=True):
                st.session_state.saved_openai_key = ""
                st.session_state.saved_anthropic_key = ""
                _schedule_cookie_delete()
                st.toast("저장된 API Key가 삭제되었습니다.")
                st.rerun()
        st.caption("API Key는 브라우저 쿠키에 저장됩니다 (홈화면 앱에서도 유지).")

    st.divider()
    st.header("모델 설정")

    # --- OpenAI (Astra) ---
    st.subheader("OpenAI (Astra)")
    _saved_astra = st.session_state.get("saved_astra_model", "")
    _astra_idx = OPENAI_MODELS.index(_saved_astra) if _saved_astra in OPENAI_MODELS else 0
    astra_model = st.selectbox("모델", options=OPENAI_MODELS, index=_astra_idx, key="sel_astra")
    astra_model_custom = st.text_input("직접 입력", placeholder="예: gpt-4o", key="custom_astra")
    astra_effort = st.select_slider("reasoning effort", options=["low", "medium", "high", "xhigh", "max"], value="high", key="effort_astra")

    st.markdown("---")

    # --- Anthropic (Fable) ---
    st.subheader("Anthropic (Fable)")
    _saved_fable = st.session_state.get("saved_fable_model", "")
    _fable_idx = ANTHROPIC_MODELS.index(_saved_fable) if _saved_fable in ANTHROPIC_MODELS else 0
    fable_model = st.selectbox("모델", options=ANTHROPIC_MODELS, index=_fable_idx, key="sel_fable")
    fable_model_custom = st.text_input("직접 입력", placeholder="예: claude-sonnet-5", key="custom_fable")
    fable_effort = st.select_slider("effort", options=["low", "medium", "high"], value="high", key="effort_fable")
    max_tokens = st.number_input("max_tokens", min_value=512, max_value=16000, value=3000, step=256)

    st.divider()

    # --- Judge 설정 ---
    st.header("Judge 설정")
    _saved_jp = st.session_state.get("saved_judge_provider", "")
    _jp_idx = JUDGE_PROVIDERS.index(_saved_jp) if _saved_jp in JUDGE_PROVIDERS else 1
    judge_provider = st.selectbox("제공자", options=JUDGE_PROVIDERS, index=_jp_idx)
    if judge_provider == "OpenAI":
        _saved_jm = st.session_state.get("saved_judge_model", "")
        _jm_idx = OPENAI_MODELS.index(_saved_jm) if _saved_jm in OPENAI_MODELS else 0
        judge_model = st.selectbox("모델", options=OPENAI_MODELS, index=_jm_idx, key="judge_model_openai")
    else:
        _saved_jm = st.session_state.get("saved_judge_model", "")
        _jm_idx = ANTHROPIC_MODELS.index(_saved_jm) if _saved_jm in ANTHROPIC_MODELS else 0
        judge_model = st.selectbox("모델", options=ANTHROPIC_MODELS, index=_jm_idx, key="judge_model_anthropic")
    judge_model_custom = st.text_input("직접 입력", placeholder="예: claude-sonnet-5", key="custom_judge")

    # --- 설정 저장 버튼 ---
    st.divider()
    if st.button("모델 설정 저장", use_container_width=True):
        _astra_final = (astra_model_custom.strip() or astra_model).strip()
        _fable_final = (fable_model_custom.strip() or fable_model).strip()
        _judge_final = (judge_model_custom.strip() or judge_model).strip()
        st.session_state.saved_astra_model = _astra_final
        st.session_state.saved_fable_model = _fable_final
        st.session_state.saved_judge_provider = judge_provider
        st.session_state.saved_judge_model = _judge_final
        _schedule_cookie_save()
        st.toast("모델 설정이 저장되었습니다.")
        st.rerun()

# --- Shuffle topics once per session ---
if "shuffled_topics" not in st.session_state:
    shuffled = DEBATE_TOPICS[:]
    random.shuffle(shuffled)
    st.session_state.shuffled_topics = shuffled

_topics = st.session_state.shuffled_topics
_default_topic = _topics[0]
_suggestion_topics = _topics[1:]

topic = st.text_area("토론 주제", value=st.session_state.get("selected_topic", _default_topic), height=100)

# Give each card its own width so its label lands on exactly CARD_LINES lines.
# Pills render in the order passed in, and that order is fixed for the session,
# so :nth-of-type indexes stay stable across reruns.
_card_css = "\n".join(
    f'[data-testid="stButtonGroup"] button[data-variant="pills"]:nth-of-type({i}) '
    f"{{width: {card_width_px(t)}px !important;}}"
    for i, t in enumerate(_suggestion_topics, start=1)
)
st.markdown(f"<style>{_card_css}</style>", unsafe_allow_html=True)

# Full topic text goes on the card; CSS line-clamps it to the card height.
_selected_pill = st.pills(
    "추천 주제",
    options=_suggestion_topics,
    default=None,
    label_visibility="collapsed",
    wrap=False,
)
if _selected_pill is not None and st.session_state.get("selected_topic") != _selected_pill:
    st.session_state.selected_topic = _selected_pill
    st.rerun()

col_role1, col_role2 = st.columns(2)
with col_role1:
    pro_label = st.text_input("OpenAI 측 역할", value="찬성 측")
with col_role2:
    con_label = st.text_input("Anthropic 측 역할", value="반대 측")


ROUND_SUMMARIES = {
    1: "독립 주장 · API 호출 3회",
    2: "독립 주장 + 상대 주장 비판 · API 호출 5회",
    3: "독립 주장 + 상대 주장 비판 + 최종 반론 · API 호출 7회",
}

_rounds = st.segmented_control(
    "토론 라운드",
    options=[1, 2, 3],
    default=3,
    required=True,
    format_func=lambda x: f"{x}라운드",
)
num_rounds = _rounds or 3
st.caption(ROUND_SUMMARIES[num_rounds])

_start_clicked = st.button(
    f"💬 {num_rounds}라운드 토론 시작", type="primary", use_container_width=True
)

if _start_clicked:
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

    astra_display = (astra_model_custom.strip() or astra_model).strip()
    fable_display = (fable_model_custom.strip() or fable_model).strip()

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
                astra_model=astra_display,
                fable_model=fable_display,
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
                openai_name=astra_display,
                anthropic_name=fable_display,
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

        # Model info banner
        st.markdown(
            f'<div style="text-align:center;color:{_T["banner_fg"]};font-size:0.85rem;margin-bottom:1rem;">'
            f'OpenAI: <b>{astra_display}</b> &nbsp;vs&nbsp; Anthropic: <b>{fable_display}</b> &nbsp;|&nbsp; Judge: <b>{judge_actual_model}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

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
        _pro = pro_label.strip() or "찬성 측"
        _con = con_label.strip() or "반대 측"
        verdict = parse_verdict(judge_result, _pro, _con)
        verdict_labels = {
            "pro": (f"{_pro} 승 ({astra_display})", "verdict-astra"),
            "con": (f"{_con} 승 ({fable_display})", "verdict-fable"),
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
        transcript = build_transcript(topic.strip(), pro_label, con_label, results, num_rounds,
                                      openai_name=astra_display, anthropic_name=fable_display)
        transcript += f"\n## Judge 판정 ({judge_actual_model})\n\n{judge_result}\n"
        st.download_button(
            "토론 결과 Markdown으로 저장",
            data=transcript,
            file_name="astra_fable_debate.md",
            mime="text/markdown",
            use_container_width=True,
        )

    except Exception as exc:
        exc_msg = str(exc).lower()
        if "not_found" in exc_msg or "404" in exc_msg:
            # Extract model name from error message
            import re
            model_match = re.search(r"model:\s*(\S+)", str(exc))
            bad_model = model_match.group(1) if model_match else "알 수 없음"
            st.error(
                f"모델 `{bad_model}`을(를) 찾을 수 없습니다. "
                f"사이드바에서 다른 모델을 선택하거나, '직접 입력'에 올바른 모델 ID를 입력해주세요."
            )
            st.info(
                "사용 가능한 모델 확인 방법:\n"
                "```\n"
                'curl https://api.anthropic.com/v1/models \\\n'
                '  -H "x-api-key: YOUR_KEY" \\\n'
                '  -H "anthropic-version: 2023-06-01"\n'
                "```"
            )
        elif "authentication" in exc_msg or "401" in exc_msg:
            st.error("API Key가 유효하지 않습니다. 사이드바에서 올바른 키를 입력해주세요.")
        else:
            st.error("API 호출 중 오류가 발생했습니다.")
            st.exception(exc)
