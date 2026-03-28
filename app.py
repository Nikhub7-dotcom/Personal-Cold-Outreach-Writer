import os
import gradio as gr
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from typing import TypedDict, List
from firecrawl import FirecrawlApp
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    api_key=groq_api_key
)

firecrawl = FirecrawlApp(api_key=firecrawl_api_key)


# ─── STATE ───────────────────────────────────────────────────────────────────

class EmailState(TypedDict):
    url: str
    manual_text: str
    raw_text: str
    hooks: List[str]
    sender_context: str
    tone: str
    draft_email: str
    subject_lines: List[str]
    quality_score: int
    retry_count: int
    final_email: str


# ─── NODES ───────────────────────────────────────────────────────────────────

def scraper(state: EmailState) -> EmailState:
    # if user pasted linkedin text manually, use that directly
    if state["manual_text"].strip():
        return {**state, "raw_text": state["manual_text"].strip()[:4000]}

    # otherwise scrape the url (works well for company websites)
    try:
        result = firecrawl.scrape_url(state["url"], params={"formats": ["markdown"]})
        raw_text = result.get("markdown", "")[:4000]
    except Exception:
        raw_text = ""

    return {**state, "raw_text": raw_text}


def hook_extractor(state: EmailState) -> EmailState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at extracting personalization hooks from profiles.
Extract exactly 3 specific, concrete hooks (job changes, recent posts, awards, projects).
Return ONLY a valid JSON list of 3 strings. Example: ["hook1", "hook2", "hook3"]
If no hooks found, return an empty list: []"""),
        ("human", "Extract hooks from this profile:\n\n{raw_text}")
    ])
    chain = prompt | llm | JsonOutputParser()
    try:
        hooks = chain.invoke({"raw_text": state["raw_text"]})
        if not isinstance(hooks, list):
            hooks = []
    except Exception:
        hooks = []
    return {**state, "hooks": hooks}


def email_writer(state: EmailState) -> EmailState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You write cold emails that get replies.
Rules:
- Under 120 words
- Reference exactly 2 of the provided hooks naturally
- One clear CTA at the end
- No cliches like 'I hope this finds you well'
- Tone must match the specified tone
- Return only the email body, no subject line"""),
        ("human", """Sender context: {sender_context}
Tone: {tone}
Hooks to use: {hooks}

Write the cold email body:""")
    ])
    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({
        "sender_context": state["sender_context"],
        "tone": state["tone"],
        "hooks": "\n".join(state["hooks"])
    })
    return {**state, "draft_email": draft}


def fallback_writer(state: EmailState) -> EmailState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You write cold emails that get replies.
Rules:
- Under 120 words
- One clear CTA at the end
- No cliches
- Tone must match the specified tone
- Return only the email body, no subject line"""),
        ("human", """Sender context: {sender_context}
Tone: {tone}

Write a general cold email body (no personalization hooks available):""")
    ])
    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({
        "sender_context": state["sender_context"],
        "tone": state["tone"]
    })
    return {**state, "draft_email": draft}


def subject_generator(state: EmailState) -> EmailState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate exactly 3 subject line variants for this cold email.
Return ONLY a valid JSON list of 3 strings.
Example: ["Subject 1", "Subject 2", "Subject 3"]"""),
        ("human", "Email body:\n\n{draft_email}")
    ])
    chain = prompt | llm | JsonOutputParser()
    try:
        subject_lines = chain.invoke({"draft_email": state["draft_email"]})
        if not isinstance(subject_lines, list):
            subject_lines = ["Following up", "Quick question", "Thought you'd find this useful"]
    except Exception:
        subject_lines = ["Following up", "Quick question", "Thought you'd find this useful"]
    return {**state, "subject_lines": subject_lines}


def quality_checker(state: EmailState) -> EmailState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Score this cold email from 1 to 10.
Deduct points for: generic opener (-3), no clear CTA (-2), over 120 words (-2), cliches (-1 each).
Return ONLY valid JSON. Example: {{"score": 8, "reason": "Good email"}}"""),
        ("human", "Score this email:\n\n{draft_email}")
    ])
    chain = prompt | llm | JsonOutputParser()
    try:
        result = chain.invoke({"draft_email": state["draft_email"]})
        score = result.get("score", 5)
    except Exception:
        score = 5
    return {**state, "quality_score": score}


def rewriter(state: EmailState) -> EmailState:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Rewrite this cold email to score higher.
Fix: generic openers, missing CTA, cliches, length over 120 words.
Return only the improved email body."""),
        ("human", "Rewrite this email:\n\n{draft_email}")
    ])
    chain = prompt | llm | StrOutputParser()
    rewritten = chain.invoke({"draft_email": state["draft_email"]})
    return {**state, "draft_email": rewritten, "retry_count": state["retry_count"] + 1}


def save_final(state: EmailState) -> EmailState:
    return {**state, "final_email": state["draft_email"]}


# ─── ROUTING ─────────────────────────────────────────────────────────────────

def route_hooks(state: EmailState) -> str:
    if state["hooks"]:
        return "email_writer"
    return "fallback_writer"


def quality_gate(state: EmailState) -> str:
    if state["quality_score"] >= 7:
        return "save_final"
    if state["retry_count"] >= 2:
        return "save_final"
    return "rewriter"


# ─── GRAPH ───────────────────────────────────────────────────────────────────

graph = StateGraph(EmailState)

graph.add_node("scraper", scraper)
graph.add_node("hook_extractor", hook_extractor)
graph.add_node("email_writer", email_writer)
graph.add_node("fallback_writer", fallback_writer)
graph.add_node("subject_generator", subject_generator)
graph.add_node("quality_checker", quality_checker)
graph.add_node("rewriter", rewriter)
graph.add_node("save_final", save_final)

graph.set_entry_point("scraper")

graph.add_edge("scraper", "hook_extractor")
graph.add_conditional_edges("hook_extractor", route_hooks)
graph.add_edge("email_writer", "subject_generator")
graph.add_edge("fallback_writer", "subject_generator")
graph.add_edge("subject_generator", "quality_checker")
graph.add_conditional_edges("quality_checker", quality_gate)
graph.add_edge("rewriter", "quality_checker")
graph.add_edge("save_final", END)

app = graph.compile()


# ─── CSS ─────────────────────────────────────────────────────────────────────

css = """
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { box-sizing: border-box; }

body, .gradio-container {
    background: #0a0a0f !important;
    font-family: 'DM Sans', sans-serif !important;
    min-height: 100vh;
}

.gradio-container {
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(99, 60, 255, 0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(0, 200, 150, 0.12) 0%, transparent 55%),
        radial-gradient(ellipse 40% 60% at 60% 30%, rgba(255, 80, 120, 0.08) 0%, transparent 50%),
        url('https://images.unsplash.com/photo-1557804506-669a67965ba0?w=1800&q=80') !important;
    background-size: cover !important;
    background-attachment: fixed !important;
    max-width: 100% !important;
    padding: 0 !important;
}

#hero {
    text-align: center;
    padding: 52px 20px 36px;
}

#hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: clamp(2rem, 5vw, 3.4rem) !important;
    font-weight: 800 !important;
    letter-spacing: -0.02em;
    color: #ffffff !important;
    margin: 0 0 12px !important;
    line-height: 1.1 !important;
}

#hero h1 span {
    background: linear-gradient(135deg, #a78bfa, #34d399, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

#hero p {
    font-size: 1rem !important;
    color: rgba(255,255,255,0.45) !important;
    font-weight: 300 !important;
    margin: 0 !important;
}

#main-card {
    max-width: 860px;
    margin: 0 auto 60px;
    padding: 0 20px;
}

.panel-card {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 20px !important;
    padding: 28px !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    margin-bottom: 16px !important;
}

.section-label {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: rgba(167,139,250,0.7) !important;
    margin-bottom: 16px !important;
}

.hint-text {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.3);
    margin-top: 10px;
    line-height: 1.6;
    padding: 10px 14px;
    background: rgba(167,139,250,0.06);
    border-radius: 8px;
    border-left: 2px solid rgba(167,139,250,0.3);
}

.gradio-container input[type="text"],
.gradio-container textarea,
label textarea,
label input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #f1f0ff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    transition: border-color 0.2s, background 0.2s !important;
    padding: 12px 16px !important;
}

.gradio-container input[type="text"]:focus,
.gradio-container textarea:focus {
    border-color: rgba(167,139,250,0.6) !important;
    background: rgba(255,255,255,0.07) !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.1) !important;
}

.gradio-container label > span,
.gradio-container .label-wrap span {
    color: rgba(255,255,255,0.75) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    margin-bottom: 6px !important;
}

.gradio-container .wrap-inner,
.gradio-container select {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #f1f0ff !important;
}

#generate-btn {
    background: linear-gradient(135deg, #7c3aed, #2563eb) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.03em !important;
    padding: 14px 32px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    margin-top: 8px !important;
}

#generate-btn:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}

#generate-btn:active {
    transform: translateY(0px) !important;
}

#email-output textarea,
#subject-output textarea {
    background: rgba(52, 211, 153, 0.05) !important;
    border: 1px solid rgba(52,211,153,0.2) !important;
    border-radius: 14px !important;
    color: #e2fdf4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.75 !important;
}

#status-bar {
    padding: 10px 16px;
    background: rgba(167,139,250,0.08);
    border: 1px solid rgba(167,139,250,0.15);
    border-radius: 10px;
    margin-bottom: 16px;
    font-size: 0.82rem;
    color: rgba(167,139,250,0.8);
    font-family: 'DM Sans', sans-serif;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 20px 0;
}

#footer {
    text-align: center;
    padding: 20px;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.2);
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.05em;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.3); border-radius: 4px; }
"""


# ─── GENERATE FUNCTION ───────────────────────────────────────────────────────

def generate_email(manual_text, url, sender_context, tone):
    result = app.invoke({
        "url": url,
        "manual_text": manual_text,
        "raw_text": "",
        "hooks": [],
        "sender_context": sender_context,
        "tone": tone,
        "draft_email": "",
        "subject_lines": [],
        "quality_score": 0,
        "retry_count": 0,
        "final_email": ""
    })
    email = result["final_email"]
    subjects = "\n".join(result["subject_lines"])
    return email, subjects


# ─── GRADIO UI ───────────────────────────────────────────────────────────────

with gr.Blocks(css=css, title="AI Cold Email Writer") as demo:

    gr.HTML("""
    <div id="hero">
        <h1>Cold Emails That <span>Actually Get Replies</span></h1>
        <p>Powered by LangGraph · LangChain · Llama 3.1 · Firecrawl</p>
    </div>
    """)

    with gr.Column(elem_id="main-card"):

        # ── 01 target ──
        with gr.Group(elem_classes="panel-card"):
            gr.HTML('<div class="section-label">01 — Target Profile</div>')

            manual_input = gr.Textbox(
                label="Paste LinkedIn Profile Text  (recommended)",
                placeholder="Open their LinkedIn → press Ctrl+A to select all → Ctrl+C to copy → paste here.\nThe more context you paste, the better the personalization.",
                lines=7
            )
            gr.HTML("""
            <div class="hint-text">
                LinkedIn blocks all scrapers. Pasting manually is the most reliable method —
                just open their profile, Ctrl+A, Ctrl+C, paste above.
            </div>
            """)

            gr.HTML('<div style="margin: 16px 0; text-align: center; color: rgba(255,255,255,0.2); font-size:0.82rem;">— or use a company website instead —</div>')

            url_input = gr.Textbox(
                label="Company Website URL  (auto-scraped)",
                placeholder="https://company.com",
                lines=1
            )

        # ── 02 sender context ──
        with gr.Group(elem_classes="panel-card"):
            gr.HTML('<div class="section-label">02 — Your Context</div>')
            sender_input = gr.Textbox(
                label="Who are you & why are you reaching out?",
                placeholder="I'm Nikhil, a final-year CS student at MSIT building AI tools. Reaching out to explore internship or collaboration opportunities...",
                lines=3
            )

        # ── 03 tone ──
        with gr.Group(elem_classes="panel-card"):
            gr.HTML('<div class="section-label">03 — Tone</div>')
            tone_input = gr.Dropdown(
                choices=["casual", "formal", "founder-to-founder"],
                value="casual",
                label="Email Tone"
            )

        gr.HTML('<div class="divider"></div>')

        generate_btn = gr.Button("Generate Email →", elem_id="generate-btn")

        gr.HTML("""
        <div id="status-bar">
            ⚡ Pipeline: Parse Profile → Extract Hooks → Write Email → Score → Auto-refine if needed
        </div>
        """)

        # ── outputs ──
        with gr.Group(elem_classes="panel-card"):
            gr.HTML('<div class="section-label">Generated Email</div>')
            email_output = gr.Textbox(
                label="",
                lines=12,
                interactive=False,
                elem_id="email-output"
            )

        with gr.Group(elem_classes="panel-card"):
            gr.HTML('<div class="section-label">Subject Line Variants</div>')
            subject_output = gr.Textbox(
                label="",
                lines=3,
                interactive=False,
                elem_id="subject-output"
            )

    gr.HTML('<div id="footer">Built with LangGraph + LangChain · Nikhil Kumar · 2026</div>')

    generate_btn.click(
        fn=generate_email,
        inputs=[manual_input, url_input, sender_input, tone_input],
        outputs=[email_output, subject_output]
    )

demo.launch(share=True)