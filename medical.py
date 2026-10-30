"""PulseNova AI - single-file, safety-first medical education prototype.

This application is an educational prototype. It does not diagnose or replace
professional medical care.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import streamlit as st
from langchain.chains import LLMChain
from langchain_community.cache import SQLiteCache
from langchain_core.caches import InMemoryCache
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIConnectionError, APITimeoutError, AuthenticationError, OpenAI, RateLimitError
from streamlit.errors import StreamlitSecretNotFoundError
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


# Community Cloud uses Secrets and does not need a local .env file.
# Keep the optional local configuration loader out of the Cloud startup path.
_env_path = Path(__file__).resolve().with_name(".env")
if _env_path.is_file():
    from dotenv import load_dotenv

    load_dotenv(_env_path)

COMMON_SYMPTOMS = ["Abdominal pain", "Back pain", "Chest pain", "Chills", "Cough", "Dizziness", "Fatigue", "Fever", "Headache", "Nausea", "Runny nose", "Shortness of breath", "Sore throat", "Vomiting"]
DURATIONS = ["Less than 24 hours", "1-3 days", "4-7 days", "1-2 weeks", "More than 2 weeks"]
GENDERS = ["Prefer not to say", "Female", "Male", "Non-binary", "Other"]
LANGUAGES = ["English", "Urdu", "Spanish", "French", "Arabic", "Hindi"]
MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4.1"]
URGENCY_LEVELS = {"LOW", "MEDIUM", "HIGH", "EMERGENCY"}
REQUIRED_KEYS = {"summary", "possible_conditions", "urgency_level", "recommended_next_steps", "questions_for_doctor", "warning_signs"}
EMERGENCY_TERMS = {"severe chest pain", "chest pain", "shortness of breath", "can't breathe", "cannot breathe", "unconscious", "fainting", "face drooping", "slurred speech", "severe bleeding", "suicidal", "overdose"}

JSON_SCHEMA = """{{
  "summary": "",
  "possible_conditions": [{{"name": "", "reason": ""}}],
  "urgency_level": "LOW|MEDIUM|HIGH|EMERGENCY",
  "recommended_next_steps": [],
  "questions_for_doctor": [],
  "warning_signs": []
}}"""
SYSTEM_RULES = """You are PulseNova AI, an educational symptom-guidance assistant, not a doctor.
Never diagnose, confirm a condition, prescribe treatment, or imply certainty. Use cautious language.
For possible emergencies set urgency_level to EMERGENCY and advise immediate local emergency help.
Return only valid JSON matching the exact schema. Write user-facing text in {language}; keep urgency uppercase English."""
SINGLE_PROMPT = PromptTemplate.from_template("""Create a safety-first educational assessment for a {age}-year-old {gender} person.
Symptoms: {symptoms}. Duration: {duration}. Severity: {severity}/10. Conditions: {existing_conditions}.
Medications: {medications}. Notes: {notes}. Answer language: {language}. Never provide a confirmed diagnosis.""")
ASSESSMENT_PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_RULES + "\nExact schema:\n" + JSON_SCHEMA), ("human", SINGLE_PROMPT.template)])
NARRATIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Explain validated health education calmly in {language}. Never diagnose. Start with a disclaimer, highlight urgency, and recommend professional care."),
    ("human", "Patient input: {patient_input}\nValidated assessment: {assessment}\nWrite a concise narrative without adding new claims."),
])


def get_api_key() -> str | None:
    value = os.getenv("OPENAI_API_KEY", "").strip()
    if not value:
        try:
            value = str(st.secrets.get("OPENAI_API_KEY", "")).strip()
        except (StreamlitSecretNotFoundError, FileNotFoundError):
            pass
    return value or None


def configure_cache(mode: str, database_path: Path | None = None):
    """Return a session-scoped cache; never change LangChain's process-global cache."""
    if mode == "In memory":
        if "memory_cache" not in st.session_state:
            st.session_state.memory_cache = InMemoryCache(maxsize=128)
        return st.session_state.memory_cache
    elif mode == "SQLite":
        if "sqlite_cache" not in st.session_state:
            if database_path is None:
                directory = Path(tempfile.gettempdir()) / "pulsenova-cache"
                directory.mkdir(parents=True, exist_ok=True)
                database_path = directory / f'{st.session_state.session_id}.db'
            st.session_state.sqlite_cache = SQLiteCache(database_path=str(database_path.resolve()))
        return st.session_state.sqlite_cache
    return False


def build_assessment_chain(model: str, api_key: str, cache=False) -> LLMChain:
    return LLMChain(llm=ChatOpenAI(model=model, api_key=api_key, cache=cache, temperature=0, timeout=45, max_retries=2), prompt=ASSESSMENT_PROMPT)


def service_error_message(exc: Exception) -> str:
    if isinstance(exc, AuthenticationError):
        return "The AI connection could not authenticate. Check OPENAI_API_KEY in the app's Secrets settings."
    if isinstance(exc, RateLimitError):
        return "The AI service is busy or its usage limit has been reached. Please try again later."
    if isinstance(exc, (APITimeoutError, APIConnectionError)):
        return "The AI service could not be reached in time. Please try again."
    return "The AI request could not be completed. Please try again or check the app configuration."


def run_structured_assessment(chain: LLMChain, payload: dict) -> str:
    return str(chain.invoke(payload).get("text", ""))


def stream_narrative(model: str, api_key: str, payload: dict, assessment: dict):
    llm = ChatOpenAI(model=model, api_key=api_key, cache=False, temperature=0.2, streaming=True, timeout=45)
    messages = NARRATIVE_PROMPT.format_messages(language=payload["language"], patient_input=json.dumps(payload, ensure_ascii=False), assessment=json.dumps(assessment, ensure_ascii=False))
    for chunk in llm.stream(messages):
        if chunk.content:
            yield str(chunk.content)


def create_message_demo():
    return [SystemMessage(content="Educate only; never diagnose."), HumanMessage(content="I have had a sore throat for two days."), AIMessage(content="I can share general information and warning signs, but cannot diagnose the cause.")]


def safe_parse_json(raw_output: str):
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", raw_output.strip(), flags=re.I)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None, "No complete JSON object was found."
    try:
        data = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc.msg}."
    if not isinstance(data, dict):
        return None, "JSON object required."
    if missing := REQUIRED_KEYS - set(data):
        return None, f"Missing fields: {', '.join(sorted(missing))}."
    if not isinstance(data["summary"], str) or not data["summary"].strip():
        return None, "summary must be a non-empty string."
    data["urgency_level"] = str(data["urgency_level"]).upper()
    if data["urgency_level"] not in URGENCY_LEVELS:
        return None, "Invalid urgency level."
    if not isinstance(data["possible_conditions"], list) or not all(isinstance(x, dict) and all(isinstance(x.get(k), str) and x[k].strip() for k in ("name", "reason")) for x in data["possible_conditions"]):
        return None, "possible_conditions must contain name/reason objects."
    for key in ("recommended_next_steps", "questions_for_doctor", "warning_signs"):
        if not isinstance(data[key], list) or not all(isinstance(item, str) and item.strip() for item in data[key]):
            return None, f"{key} must be a list of non-empty strings."
    return data, None


def emergency_flags(symptoms: str, severity: int):
    found = sorted(term for term in EMERGENCY_TERMS if term in symptoms.lower())
    if severity >= 9:
        found.append("severity 9-10")
    return found


def assess_locally(payload: dict, forced_emergency: bool = False):
    text, severity = payload["symptoms"].lower(), int(payload["severity"])
    urgency = "EMERGENCY" if forced_emergency or emergency_flags(text, severity) else "HIGH" if severity >= 7 or ("fever" in text and "cough" in text) else "MEDIUM" if severity >= 4 else "LOW"
    possibilities = []
    if any(x in text for x in ("cough", "sore throat", "runny nose", "fever")):
        possibilities.append({"name": "Respiratory irritation or infection", "reason": "These symptoms may have viral, bacterial, allergic, or environmental causes that require clinical context."})
    if any(x in text for x in ("nausea", "vomiting", "abdominal pain")):
        possibilities.append({"name": "Digestive-system cause", "reason": "Digestive symptoms can have many causes; examination and history help a clinician narrow them."})
    if not possibilities:
        possibilities = [{"name": "Several possible causes", "reason": "The information is not specific enough for this educational prototype to narrow safely."}]
    narratives = {"Urdu": "یہ صرف تعلیمی رہنمائی ہے، تشخیص نہیں۔ اپنی علامات کی نگرانی کریں اور مستند ڈاکٹر سے رابطہ کریں۔ شدید علامات میں فوراً ہنگامی مدد حاصل کریں۔", "Spanish": "Esta orientación es educativa y no es un diagnóstico. Vigile sus síntomas y consulte a un profesional.", "French": "Ces informations sont éducatives et ne constituent pas un diagnostic. Consultez un professionnel de santé."}
    return {
        "summary": f'{payload["symptoms"]} for {payload["duration"].lower()}, severity {severity}/10.',
        "possible_conditions": possibilities,
        "urgency_level": urgency,
        "recommended_next_steps": ["Arrange professional review at the urgency shown.", "Track symptom changes and new warning signs.", "Bring your medication and condition list to the clinician."],
        "questions_for_doctor": ["What causes should be considered after examination?", "Do I need testing?", "What changes should make me seek urgent care?"],
        "warning_signs": ["Trouble breathing, blue lips, new confusion, fainting, or inability to stay awake.", "Sudden severe pain, uncontrolled bleeding, or rapid worsening."],
        "narrative": narratives.get(payload.get("language"), "This is educational guidance, not a diagnosis. Monitor symptoms, consult a professional, and seek emergency help for severe or rapidly worsening symptoms."),
    }


def build_pdf_report(payload: dict, result: dict, disclaimer: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18*mm, leftMargin=18*mm, topMargin=16*mm, bottomMargin=16*mm)
    styles = getSampleStyleSheet(); styles.add(ParagraphStyle(name="PulseTitle", parent=styles["Title"], alignment=TA_CENTER, textColor=colors.HexColor("#1557FF")))
    story = [Paragraph("PulseNova AI - Educational Guidance", styles["PulseTitle"]), Spacer(1, 5*mm)]
    warning = Table([[Paragraph(f"<b>Safety notice:</b> {escape(disclaimer)}", styles["BodyText"])]], colWidths=[170*mm])
    warning.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F3FFD0")), ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#9DFF00")), ("PADDING", (0,0), (-1,-1), 10)]))
    story += [warning, Spacer(1, 5*mm), Paragraph(f'<b>Symptoms:</b> {escape(str(payload["symptoms"]))}', styles["BodyText"]), Paragraph(f'<b>Urgency:</b> {escape(result["urgency_level"])}', styles["Heading2"]), Paragraph(escape(result["summary"]), styles["BodyText"])]
    for heading, items in (("Possible explanations - not diagnoses", [f'{x["name"]}: {x["reason"]}' for x in result["possible_conditions"]]), ("Recommended next steps", result["recommended_next_steps"]), ("Questions for a clinician", result["questions_for_doctor"]), ("Warning signs", result["warning_signs"])):
        story += [Spacer(1, 3*mm), Paragraph(heading, styles["Heading2"])] + [Paragraph(f"- {escape(str(item))}", styles["BodyText"]) for item in items]
    doc.build(story)
    return buffer.getvalue()


def transcribe_voice(audio_file) -> str:
    if not get_api_key():
        raise RuntimeError("Configure OPENAI_API_KEY in the app's Secrets settings or local .env to transcribe voice input.")
    result = OpenAI(api_key=get_api_key()).audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=(audio_file.name or "symptoms.wav", audio_file.getvalue(), "audio/wav"))
    return result if isinstance(result, str) else result.text


APP_DIR = Path(__file__).resolve().parent
DISCLAIMER = (
    "Educational AI prototype only - not a doctor, diagnosis, emergency service, "
    "or treatment plan. Always consult a qualified healthcare professional."
)

st.set_page_config(
    page_title="PulseNova AI",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_design(daylight: bool) -> None:
    """Apply the requested midnight, cobalt, and acid-lime visual system."""
    bg = "#F4F7FF" if daylight else "#05060A"
    panel = "#FFFFFF" if daylight else "#0D111C"
    text = "#071025" if daylight else "#F4F7FF"
    muted = "#526079" if daylight else "#AAB5CC"
    cobalt = "#124AD6" if daylight else "#4B78FF"
    lime_ink = "#3F6500" if daylight else "#B7FF2A"
    field = "#F7F9FF" if daylight else "#111729"
    border = "#C9D5F5" if daylight else "#26345D"
    st.html(f"""
    <style>
    :root{{--midnight:{bg};--panel:{panel};--ink:{text};--muted:{muted};--cobalt:{cobalt};--lime:#B7FF2A;--lime-ink:{lime_ink};--field:{field};--soft-border:{border};--card-gap:1rem;--section-gap:1.25rem;}}
    .stApp{{background:radial-gradient(circle at 85% 5%,color-mix(in srgb,var(--cobalt) 20%,transparent),transparent 28%),radial-gradient(circle at 10% 60%,rgba(183,255,42,.08),transparent 22%),var(--midnight);color:var(--ink)}}
    header[data-testid="stHeader"]{{display:block!important;height:3.5rem!important;background:transparent!important;pointer-events:auto!important}}
    [data-testid="stToolbar"]{{display:flex!important;visibility:visible!important;background:transparent!important}}
    [data-testid="stAppDeployButton"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],#MainMenu{{display:none!important}}
    [data-testid="stExpandSidebarButton"]{{display:flex!important;visibility:visible!important;position:fixed!important;top:.65rem!important;left:.7rem!important;z-index:1000001!important;pointer-events:auto!important}}
    [data-testid="stExpandSidebarButton"],[data-testid="stExpandSidebarButton"] button,[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapseButton"] button{{align-items:center!important;justify-content:center!important;min-width:2.45rem!important;width:2.45rem!important;height:2.45rem!important;border:1px solid var(--cobalt)!important;border-radius:12px!important;background:color-mix(in srgb,var(--panel) 92%,var(--cobalt))!important;color:var(--ink)!important;box-shadow:0 0 18px color-mix(in srgb,var(--cobalt) 34%,transparent)!important;transition:transform .18s ease,box-shadow .18s ease,border-color .18s ease!important;pointer-events:auto!important}}
    [data-testid="stExpandSidebarButton"]:hover,[data-testid="stExpandSidebarButton"] button:hover,[data-testid="stSidebarCollapseButton"]:hover,[data-testid="stSidebarCollapseButton"] button:hover{{transform:scale(1.06);border-color:var(--lime)!important;color:var(--lime-ink)!important;box-shadow:0 0 24px rgba(183,255,42,.25)!important}}
    [data-testid="stExpandSidebarButton"],[data-testid="stSidebarCollapseButton"]{{border:0!important;background:transparent!important;box-shadow:none!important}}
    [data-testid="stExpandSidebarButton"] button,[data-testid="stSidebarCollapseButton"] button{{background:#101B38!important;color:#FFFFFF!important}}
    [data-testid="stSidebarHeader"],[data-testid="stSidebarCollapseButton"]{{visibility:visible!important;opacity:1!important}}
    [data-testid="stSidebarCollapseButton"]{{display:flex!important}}
    [data-testid="stSidebarCollapseButton"] button{{position:relative}}
    [data-testid="stSidebarCollapseButton"] button:after{{content:"‹";position:absolute;inset:0;display:grid;place-items:center;color:#FFFFFF;font:500 32px/1 system-ui}}
    [data-testid="stSidebarCollapseButton"] button>*{{visibility:hidden!important}}
    [data-testid="stExpandSidebarButton"] button *,[data-testid="stSidebarCollapseButton"] button *{{color:#FFFFFF!important;fill:#FFFFFF!important;opacity:1!important}}
    [data-testid="stExpandSidebarButton"] svg,[data-testid="stSidebarCollapseButton"] svg{{display:block!important;color:#FFFFFF!important;fill:#FFFFFF!important;opacity:1!important}}
    [data-testid="stSidebar"]{{position:relative!important;overflow:hidden!important;background:radial-gradient(circle at 14% 8%,rgba(21,87,255,.20),transparent 28%),linear-gradient(180deg,#060811,#0B1024)!important;border-right:1px solid rgba(21,87,255,.45)}}
    [data-testid="stSidebarUserContent"]{{padding-top:.65rem!important;padding-bottom:1rem!important}}
    [data-testid="stSidebar"]:after{{content:"";position:absolute;right:0;top:-28%;width:2px;height:24%;background:linear-gradient(180deg,transparent,var(--cobalt),var(--lime),transparent);box-shadow:0 0 16px var(--cobalt);animation:sidebarBeam 6s linear infinite;pointer-events:none}}
    [data-testid="stSidebar"] h1{{color:#F7F9FF!important;font-size:1.75rem!important;letter-spacing:-.04em!important;text-shadow:0 0 22px rgba(75,120,255,.45)!important;animation:brandReveal .75s cubic-bezier(.2,.8,.2,1) both,brandGlow 3.8s 1s ease-in-out infinite}}
    [data-testid="stSidebar"] h1:after{{content:"";display:block;width:54px;height:3px;margin-top:.55rem;border-radius:99px;background:linear-gradient(90deg,var(--lime),var(--cobalt));box-shadow:0 0 14px rgba(183,255,42,.35)}}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p{{color:#FFFFFF!important;opacity:1!important;font-weight:650!important;letter-spacing:.02em!important;text-shadow:0 0 12px rgba(75,120,255,.38)!important;animation:captionReveal .7s .18s ease-out both}}
    [data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{{margin-top:1.1rem!important;padding:.62rem .72rem!important;border-left:4px solid var(--lime)!important;border-radius:0 10px 10px 0!important;background:linear-gradient(90deg,rgba(21,87,255,.25),rgba(21,87,255,.03))!important;color:#F7F9FF!important;font-size:.92rem!important;font-weight:800!important;letter-spacing:.075em!important;text-transform:uppercase!important;box-shadow:inset 0 0 18px rgba(21,87,255,.08)}}
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p{{color:#DDE5FF!important;font-weight:700!important;letter-spacing:.015em}}
    [data-testid="stSidebar"] details summary{{color:#F7F9FF!important;font-weight:750!important}}
    [data-testid="stSidebar"] details summary:hover{{color:var(--lime)!important}}
    .st-key-appearance_toolbar{{position:sticky!important;top:.42rem!important;z-index:999990!important;display:flex!important;justify-content:flex-end!important;pointer-events:none!important;margin:-.15rem 0 .55rem!important}}
    .st-key-theme_toggle{{pointer-events:auto!important;width:auto!important}}
    .st-key-theme_toggle label{{position:relative;width:108px!important;min-width:108px!important;max-width:108px!important;height:58px!important;min-height:58px!important;padding:0!important;overflow:hidden;border:1px solid var(--soft-border);border-radius:999px;background:color-mix(in srgb,var(--panel) 92%,transparent);box-shadow:0 8px 24px rgba(0,0,0,.16),inset 0 1px rgba(255,255,255,.12);cursor:pointer;backdrop-filter:blur(18px);transition:border-color .25s,box-shadow .25s}}
    .st-key-theme_toggle label>*{{opacity:0!important}}
    .st-key-theme_toggle label:before{{content:"☾     ☼";white-space:pre;position:absolute;left:12px;top:10px;width:84px;height:38px;padding:0 10px;box-sizing:border-box;border-radius:999px;background:linear-gradient(120deg,#152650,#124AD6);color:#DDE5FF;font:22px/38px system-ui;box-shadow:inset 0 2px 7px #0006;transition:background .35s}}
    .st-key-theme_toggle label:after{{content:"☾";position:absolute;left:16px;top:14px;display:grid;place-items:center;width:30px;height:30px;border-radius:50%;background:linear-gradient(145deg,#FFFFFF,#C9D5F5);color:#124AD6;font:700 22px/1 system-ui;box-shadow:0 3px 8px #0005,inset 0 1px 2px #fff;transition:transform .4s cubic-bezier(.2,.8,.2,1),background .3s,color .3s}}
    .st-key-theme_toggle label:has(input:checked):before{{background:linear-gradient(120deg,#8DCE18,#B7FF2A);color:#304C00}}
    .st-key-theme_toggle label:has(input:checked):after{{content:"☼";transform:translateX(46px) rotate(90deg);background:linear-gradient(145deg,#FFFFFF,#ECFFC8);color:#3F6500}}
    .st-key-theme_toggle label:hover{{border-color:var(--cobalt);box-shadow:0 8px 26px color-mix(in srgb,var(--cobalt) 24%,transparent)}}
    .st-key-theme_toggle label:has(input:focus-visible){{outline:3px solid var(--cobalt);outline-offset:4px}}
    .st-key-theme_toggle [data-testid="stMarkdownContainer"] p{{color:var(--ink)!important;font-size:.9rem!important;font-weight:750!important}}
    .signal-deck{{position:relative;isolation:isolate;overflow:hidden;display:grid;grid-template-columns:1.15fr 1.85fr;align-items:center;gap:1rem;min-height:88px;margin:0 0 .78rem;padding:.82rem 1.1rem;border:1px solid color-mix(in srgb,var(--cobalt) 55%,var(--soft-border));border-radius:18px;background:linear-gradient(110deg,color-mix(in srgb,var(--panel) 93%,var(--cobalt)),color-mix(in srgb,var(--panel) 96%,var(--lime)));box-shadow:0 12px 34px rgba(0,0,0,.13),inset 0 1px rgba(255,255,255,.10);animation:deckReveal .72s cubic-bezier(.2,.8,.2,1) both}}
    .signal-deck:before{{content:"";position:absolute;z-index:-1;inset:-120%;background:conic-gradient(from 40deg,transparent 0 42%,color-mix(in srgb,var(--cobalt) 17%,transparent) 48%,transparent 54% 74%,color-mix(in srgb,var(--lime) 10%,transparent) 80%,transparent 86%);animation:deckOrbit 16s linear infinite}}
    .signal-copy small{{display:block;color:var(--lime-ink);font-weight:850;letter-spacing:.17em;text-transform:uppercase}}
    .signal-copy strong{{display:block;margin-top:.28rem;color:var(--ink);font-size:1.34rem;letter-spacing:-.025em}}
    .signal-copy p{{margin:.25rem 0 0;color:var(--muted);font-size:.88rem}}
    .signal-right{{display:flex;flex-direction:column;gap:.72rem}}
    .signal-chips{{display:flex;justify-content:flex-end;gap:.55rem;flex-wrap:wrap}}
    .signal-chip{{display:inline-flex;align-items:center;gap:.42rem;padding:.43rem .68rem;border:1px solid var(--soft-border);border-radius:999px;background:color-mix(in srgb,var(--panel) 90%,var(--cobalt));color:var(--ink);font-size:.78rem;font-weight:750;box-shadow:0 7px 18px rgba(0,0,0,.10);animation:chipFloat 4s ease-in-out infinite}}
    .signal-chip:nth-child(2){{animation-delay:.45s}}.signal-chip:nth-child(3){{animation-delay:.9s}}
    .live-dot{{width:8px;height:8px;border-radius:50%;background:var(--lime);box-shadow:0 0 0 0 rgba(183,255,42,.55);animation:livePulse 2s ease-out infinite}}
    .signal-track{{position:relative;align-self:stretch;height:24px;overflow:hidden;border-radius:999px;background:linear-gradient(90deg,transparent,color-mix(in srgb,var(--cobalt) 9%,transparent),transparent)}}
    .signal-track:before{{content:"";position:absolute;left:-42%;top:11px;width:42%;height:2px;background:linear-gradient(90deg,transparent,var(--cobalt),var(--lime),transparent);filter:drop-shadow(0 0 6px var(--cobalt));animation:signalSweep 3.5s ease-in-out infinite}}
    .signal-track:after{{content:"⌁  ╱╲╱╲  ─────  +  ─────  ╱╲";position:absolute;inset:0;color:color-mix(in srgb,var(--lime-ink) 72%,transparent);font:800 18px/24px monospace;letter-spacing:.2em;text-align:right;white-space:nowrap;opacity:.72}}
    [data-testid="stMainBlockContainer"],.block-container{{max-width:1280px!important;padding-top:.2rem!important}}
    .nova-hero{{position:relative;overflow:hidden;padding:1.25rem 1.5rem;border-radius:18px;border:1px solid rgba(21,87,255,.65);background:linear-gradient(120deg,#070910 15%,#0B1540 56%,#102D8F);box-shadow:0 16px 40px rgba(21,87,255,.16);isolation:isolate}}
    .nova-hero:before{{content:"";position:absolute;inset:-80%;background:conic-gradient(from 90deg,transparent,#1557FF44,transparent,#B7FF2A33,transparent);animation:scan 12s linear infinite;z-index:-1}}
    .nova-hero h1{{font-size:clamp(2.25rem,5vw,3.4rem);line-height:1;margin:.35rem 0;color:white;letter-spacing:-.065em;text-shadow:0 0 25px rgba(21,87,255,.55)}}
    .nova-hero h1 span,.nova-kicker{{color:var(--lime)}}
    .nova-hero p{{position:relative;max-width:70%;color:#D9E2FF;font-size:.9rem;line-height:1.5;margin:.55rem 0 0}}
    .nova-kicker{{font-size:.7rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}}
    .side-console{{margin:.9rem 0 1rem;padding:.9rem;border:1px solid rgba(75,120,255,.35);border-radius:17px;background:linear-gradient(145deg,rgba(21,87,255,.15),rgba(183,255,42,.045));box-shadow:inset 0 1px rgba(255,255,255,.07),0 12px 28px rgba(0,0,0,.18)}}
    .side-console-head{{display:flex;align-items:center;justify-content:space-between;color:#F7F9FF;font-size:.72rem;font-weight:850;letter-spacing:.11em;text-transform:uppercase}}
    .side-live{{display:inline-flex;align-items:center;gap:.35rem;color:#B7FF2A;font-size:.65rem;letter-spacing:.06em}}
    .side-journey{{display:grid;grid-template-columns:1fr auto 1fr auto 1fr;align-items:center;gap:.35rem;margin-top:.85rem}}
    .side-step{{display:flex;flex-direction:column;align-items:center;gap:.35rem;color:#DDE5FF;font-size:.67rem;font-weight:700;text-align:center}}
    .side-step b{{display:grid;place-items:center;width:25px;height:25px;border:1px solid rgba(183,255,42,.65);border-radius:50%;background:rgba(183,255,42,.10);color:#B7FF2A;box-shadow:0 0 12px rgba(183,255,42,.12)}}
    .side-link{{width:14px;height:1px;background:linear-gradient(90deg,#4B78FF,#B7FF2A);opacity:.65}}
    .pulse-line{{position:absolute;right:4%;top:50%;width:28%;height:80px;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 300 80'%3E%3Cpath d='M0 40h60l14-24 25 52 26-50 20 42 18-20h137' fill='none' stroke='%23B7FF2A' stroke-width='4'/%3E%3C/svg%3E") center/contain no-repeat;filter:drop-shadow(0 0 10px #B7FF2A);animation:pulse 2.2s ease-in-out infinite}}
    .feature-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:var(--card-gap);margin:var(--section-gap) 0}}
    .feature-card{{position:relative;padding:1rem;border-radius:12px;font-size:.82rem;line-height:1.45;background:linear-gradient(145deg,color-mix(in srgb,var(--cobalt) 14%,var(--panel)),color-mix(in srgb,var(--lime) 4%,var(--panel)));border:1px solid var(--soft-border);color:var(--ink);transition:.25s transform,.25s border,.25s box-shadow}}
    .feature-card:hover{{transform:translateY(-6px);border-color:var(--lime);box-shadow:0 15px 45px rgba(21,87,255,.22)}}
    .feature-card b{{display:block;color:var(--lime-ink);margin-bottom:.3rem}}
    .safety-strip{{position:relative;overflow:hidden;padding:1rem 1.1rem;border:1px solid var(--lime);border-left:7px solid var(--lime);border-radius:14px;background:linear-gradient(90deg,color-mix(in srgb,var(--lime) 15%,var(--panel)),color-mix(in srgb,var(--cobalt) 10%,var(--panel)));color:var(--ink);box-shadow:0 0 24px rgba(183,255,42,.10);animation:safetyBreath 3.4s ease-in-out infinite}}
    .safety-strip b{{color:var(--lime-ink);letter-spacing:.055em}}
    .safety-strip:after{{content:"";position:absolute;inset:-60% auto -60% -30%;width:20%;background:linear-gradient(90deg,transparent,rgba(255,255,255,.36),transparent);transform:skewX(-18deg);animation:safetyScan 5.5s ease-in-out infinite}}
    [data-testid="stForm"]{{position:relative;overflow:hidden;animation:formReveal .72s cubic-bezier(.2,.8,.2,1) both}}
    [data-testid="stForm"]:before{{content:"";position:absolute;left:0;top:0;width:100%;height:3px;background:linear-gradient(90deg,var(--cobalt),var(--lime),var(--cobalt));background-size:220% 100%;animation:formBeam 4s linear infinite}}
    [data-testid="stForm"],[data-testid="stMetric"],details{{background:color-mix(in srgb,var(--panel) 94%,transparent)!important;border-color:var(--soft-border)!important;box-shadow:0 10px 35px rgba(0,0,0,.14)}}
    [data-testid="stForm"] input,[data-testid="stForm"] textarea{{background:var(--field)!important;color:var(--ink)!important;caret-color:var(--cobalt)!important}}
    [data-testid="stForm"] input::placeholder,[data-testid="stForm"] textarea::placeholder{{color:var(--muted)!important;opacity:.82}}
    [data-testid="stForm"] [data-baseweb="input"],[data-testid="stForm"] [data-baseweb="textarea"],[data-testid="stForm"] [data-baseweb="select"]>div{{background:var(--field)!important;color:var(--ink)!important;border-color:var(--soft-border)!important}}
    [data-testid="stForm"] [data-baseweb="select"] svg,[data-testid="stForm"] button svg,[data-testid="stForm"] [data-testid="stWidgetLabel"] svg{{color:var(--cobalt)!important;fill:currentColor!important;opacity:1!important}}
    [data-testid="stForm"] [data-baseweb="tag"]{{background:var(--cobalt)!important;color:white!important;border:1px solid color-mix(in srgb,var(--cobalt) 70%,white)!important}}
    [data-testid="stForm"] [data-baseweb="tag"] span,[data-testid="stForm"] [data-baseweb="tag"] svg{{color:white!important;fill:white!important}}
    [data-testid="stForm"] [data-testid="stWidgetLabel"] p,[data-testid="stForm"] label,[data-testid="stForm"] p{{color:var(--ink)}}
    [data-testid="stForm"] [data-testid="stCheckbox"] svg{{color:#05060A!important;fill:#05060A!important}}
    [data-baseweb="popover"] [role="listbox"],div[role="listbox"]{{background:var(--panel)!important;border:1px solid var(--soft-border)!important;box-shadow:0 16px 40px rgba(0,0,0,.32)!important}}
    [role="option"]{{background:var(--panel)!important;color:var(--ink)!important}}
    [role="option"]:hover,[role="option"][aria-selected="true"]{{background:color-mix(in srgb,var(--cobalt) 18%,var(--panel))!important;color:var(--ink)!important}}
    .form-intro{{display:flex;align-items:center;gap:.9rem;margin:.1rem 0 1.25rem;animation:introSlide .8s .12s both}}
    .form-intro h2{{margin:0;color:var(--ink);font-size:1.65rem}}
    .form-intro span{{display:block;color:var(--cobalt);font-size:.75rem;font-weight:800;letter-spacing:.16em;text-transform:uppercase}}
    .form-intro p{{margin:.2rem 0 0;color:var(--muted)}}
    .form-pulse{{position:relative;flex:0 0 48px;width:48px;height:48px;border-radius:50%;background:radial-gradient(circle,var(--lime) 0 18%,var(--cobalt) 20% 36%,transparent 38%);box-shadow:0 0 24px color-mix(in srgb,var(--cobalt) 65%,transparent);animation:formPulse 2.2s ease-in-out infinite}}
    .form-pulse:after{{content:"";position:absolute;inset:-7px;border:1px solid var(--cobalt);border-radius:50%;animation:pulseRing 2.2s ease-out infinite}}
    div.stButton>button,div.stDownloadButton>button{{border:1px solid var(--cobalt)!important;background:linear-gradient(100deg,var(--cobalt),#0B2B98)!important;color:white!important;font-weight:750!important;box-shadow:0 0 22px color-mix(in srgb,var(--cobalt) 32%,transparent);transition:.2s}}
    div.stButton>button:hover,div.stDownloadButton>button:hover{{border-color:#B7FF2A!important;color:#B7FF2A!important;transform:translateY(-2px);box-shadow:0 0 32px rgba(183,255,42,.22)}}
    input:focus,textarea:focus{{box-shadow:0 0 0 2px var(--cobalt),0 0 22px color-mix(in srgb,var(--cobalt) 25%,transparent)!important}}
    ::selection{{background:#B7FF2A;color:#05060A}}
    [data-testid="stMainBlockContainer"]{{padding-left:clamp(1rem,3vw,2.5rem)!important;padding-right:clamp(1rem,3vw,2.5rem)!important}}
    [data-testid="stMainBlockContainer"]>[data-testid="stVerticalBlock"]{{gap:var(--section-gap)!important}}
    [data-testid="stSidebarUserContent"]>[data-testid="stVerticalBlock"]{{gap:var(--card-gap)!important}}
    [data-testid="stHorizontalBlock"]{{column-gap:var(--card-gap)!important;row-gap:var(--card-gap)!important}}
    [data-testid="stMain"] [data-testid="stMarkdownContainer"] p{{font-size:.9rem;line-height:1.5}}
    [data-testid="stMain"] h2{{font-size:1.35rem!important;padding-top:.25rem!important;padding-bottom:.4rem!important}}
    [data-testid="stMain"] h3{{font-size:1.05rem!important;padding-top:.2rem!important;padding-bottom:.3rem!important}}
    [data-testid="stWidgetLabel"] p{{font-size:.82rem!important}}
    [data-testid="stButtonGroup"] button,[data-testid="stButtonGroup"] [role="radio"]{{background:var(--panel)!important;color:var(--muted)!important;border-color:var(--soft-border)!important;font-size:.82rem!important}}
    [data-testid="stButtonGroup"] [aria-checked="true"],[data-testid="stButtonGroup"] [aria-pressed="true"]{{background:var(--cobalt)!important;color:#FFFFFF!important;border-color:var(--cobalt)!important}}
    [data-testid="stSidebar"] [data-testid="stButtonGroup"] button{{background:#141D34!important;color:#DDE5FF!important;border-color:#3B4968!important}}
    [data-testid="stSidebar"] [data-testid="stButtonGroup"] [aria-checked="true"],[data-testid="stSidebar"] [data-testid="stButtonGroup"] [aria-pressed="true"]{{background:#124AD6!important;color:#FFFFFF!important}}
    [data-testid="stForm"] [data-testid="stSelectbox"] button,[data-testid="stForm"] [data-testid="stMultiSelect"] button{{background:var(--field)!important;color:var(--cobalt)!important}}
    [data-testid="stForm"] [data-testid="stSelectbox"] [role="combobox"],[data-testid="stForm"] [data-testid="stMultiSelect"] [role="combobox"]{{color:var(--ink)!important}}
    [data-testid="stForm"]{{padding:1.25rem!important;border-radius:16px!important}}
    [data-testid="stForm"] [data-testid="stVerticalBlock"]{{gap:var(--card-gap)!important}}
    [data-testid="stMetric"]{{padding:1rem!important;border-radius:12px!important}}
    [data-testid="stMetricValue"]{{font-size:1.6rem!important}}
    [data-testid="stExpander"] details{{border-radius:12px!important}}
    [data-testid="stExpander"] summary{{color:var(--ink)!important;font-size:.88rem!important}}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary{{color:#F7F9FF!important}}
    [data-testid="stSidebar"] [data-testid="stAlert"]{{background:#181E30!important;border:1px solid #3B4968!important;border-radius:12px!important;padding:.75rem!important}}
    [data-testid="stSidebar"] [data-testid="stAlert"] p{{color:#E6EDFF!important;font-size:.82rem!important;line-height:1.5!important}}
    [data-testid="stSidebar"] h1{{font-size:1.45rem!important;padding-top:.25rem!important;padding-bottom:.35rem!important}}
    [data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{{margin-top:.4rem!important;padding:.5rem .65rem!important;font-size:.78rem!important}}
    .side-console{{padding:.85rem;margin:0;border-radius:12px}}
    .signal-deck{{min-height:0;margin:0 0 var(--section-gap);padding:.9rem 1rem;border-radius:14px;grid-template-columns:1.15fr 1.4fr;gap:var(--card-gap)}}
    .signal-copy small{{font-size:.64rem;letter-spacing:.12em}}
    .signal-copy strong{{font-size:1.05rem}}
    .signal-copy p{{font-size:.78rem}}
    .signal-chip{{font-size:.68rem;padding:.3rem .5rem;animation:none}}
    .signal-chips{{gap:.35rem}}
    .signal-right{{gap:.3rem}}
    .signal-track{{height:18px}}
    .safety-strip{{padding:.7rem .85rem;border-left-width:4px;font-size:.8rem;line-height:1.5;border-radius:12px}}
    .form-intro{{gap:var(--card-gap);margin:0}}
    .form-intro h2{{font-size:1.3rem}}
    .form-intro p{{font-size:.82rem}}
    .form-pulse{{flex-basis:34px;width:34px;height:34px}}
    .pulse-line{{top:36%;right:3%;width:23%;opacity:.65}}
    @media(max-width:1000px){{.signal-deck{{grid-template-columns:1fr}}.signal-chips{{justify-content:flex-start}}.signal-track{{display:none}}}}
    @media(max-width:600px){{.nova-hero{{padding:1rem}}.nova-hero p{{max-width:100%}}.pulse-line{{display:none}}.side-console-head{{letter-spacing:.04em}}}}
    @keyframes scan{{to{{transform:rotate(360deg)}}}} @keyframes pulse{{50%{{opacity:.45;transform:scaleX(.97)}}}}
    @keyframes safetyBreath{{0%,100%{{box-shadow:0 0 16px rgba(183,255,42,.08)}}50%{{box-shadow:0 0 30px rgba(183,255,42,.22)}}}}
    @keyframes safetyScan{{0%,58%{{left:-30%;opacity:0}}68%{{opacity:1}}88%,100%{{left:120%;opacity:0}}}}
    @keyframes formReveal{{from{{opacity:0;transform:translateY(24px);clip-path:inset(0 0 100% 0)}}to{{opacity:1;transform:none;clip-path:inset(0)}}}}
    @keyframes formBeam{{to{{background-position:220% 0}}}} @keyframes introSlide{{from{{opacity:0;transform:translateX(-18px)}}to{{opacity:1;transform:none}}}}
    @keyframes formPulse{{50%{{transform:scale(1.08);filter:brightness(1.18)}}}} @keyframes pulseRing{{to{{inset:-16px;opacity:0}}}}
    @keyframes sidebarBeam{{from{{top:-28%}}to{{top:110%}}}} @keyframes brandReveal{{from{{opacity:0;transform:translateX(-20px)}}to{{opacity:1;transform:none}}}}
    @keyframes brandGlow{{50%{{text-shadow:0 0 30px rgba(75,120,255,.72)}}}} @keyframes captionReveal{{from{{opacity:0;transform:translateY(10px);letter-spacing:.08em}}to{{opacity:1;transform:none;letter-spacing:.02em}}}}
    @keyframes deckReveal{{from{{opacity:0;transform:translateY(-16px) scale(.98)}}to{{opacity:1;transform:none}}}} @keyframes deckOrbit{{to{{transform:rotate(360deg)}}}}
    @keyframes chipFloat{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-4px)}}}} @keyframes livePulse{{70%{{box-shadow:0 0 0 9px rgba(183,255,42,0)}}100%{{box-shadow:0 0 0 0 rgba(183,255,42,0)}}}}
    @keyframes signalSweep{{0%{{left:-42%;opacity:0}}20%{{opacity:1}}80%{{opacity:1}}100%{{left:110%;opacity:0}}}}
    @media(max-width:760px){{.feature-grid{{grid-template-columns:1fr}}.pulse-line{{opacity:.25;width:55%}}.signal-deck{{grid-template-columns:1fr;min-height:auto}}.signal-chips{{justify-content:flex-start}}}}
    @media(prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
    </style>""")


def initialize_state() -> None:
    defaults = {
        "assessment": None,
        "narrative": "",
        "history": [],
        "raw_output": "",
        "parse_error": "",
        "cache_status": "Not used",
        "appearance": "Midnight",
        "voice_text": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("session_id", uuid.uuid4().hex)


def patient_payload(
    age: str,
    gender: str,
    symptoms: list[str],
    other_symptoms: str,
    duration: str,
    severity: int,
    conditions: str,
    medications: str,
    notes: str,
    language: str,
) -> dict:
    combined = symptoms + ([other_symptoms.strip()] if other_symptoms.strip() else [])
    return {
        "age": age.strip(),
        "gender": gender,
        "symptoms": ", ".join(combined),
        "duration": duration,
        "severity": severity,
        "existing_conditions": conditions.strip() or "None reported",
        "medications": medications.strip() or "None reported",
        "notes": notes.strip() or "None",
        "language": language,
    }


def validate_payload(payload: dict, consent: bool) -> list[str]:
    errors: list[str] = []
    try:
        age = int(payload["age"])
        if not 1 <= age <= 120:
            errors.append("Age must be between 1 and 120.")
    except ValueError:
        errors.append("Enter age as a whole number.")
    if not payload["symptoms"]:
        errors.append("Choose or describe at least one symptom.")
    if not consent:
        errors.append("Please acknowledge the educational-use notice.")
    return errors


def run_assessment(payload: dict, model: str, cache_mode: str) -> None:
    """Run emergency pre-check, then real LLM or safe local demonstration."""
    # A failed request must not leave a previous patient's result visible.
    st.session_state.assessment = None
    st.session_state.narrative = ""
    st.session_state.parse_error = ""
    st.session_state.raw_output = ""
    flags = emergency_flags(payload["symptoms"], int(payload["severity"]))
    api_key = get_api_key()
    started = time.perf_counter()

    if flags:
        result = assess_locally(payload, forced_emergency=True)
        raw_output = "Safety pre-check generated an emergency response before any API call."
        cache_status = "Bypassed for safety"
    elif api_key:
        try:
            cache = configure_cache(cache_mode)
            chain = build_assessment_chain(model=model, api_key=api_key, cache=cache)
            raw_output = run_structured_assessment(chain, payload)
        except Exception as exc:
            st.error(service_error_message(exc), title="Assessment unavailable")
            return
        result, error = safe_parse_json(raw_output)
        if error:
            st.session_state.parse_error = error
            st.session_state.raw_output = raw_output
            st.error(
                "The AI response could not be validated. No medical result was shown. "
                "You can review the raw output in the diagnostics panel below.",
                title="Safe parsing stopped an invalid response",
            )
            return
        cache_status = cache_mode
    else:
        result = assess_locally(payload)
        raw_output = "Local demonstration engine used because OPENAI_API_KEY is not configured."
        cache_status = "Demo mode"

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    result["response_time_ms"] = elapsed_ms
    st.session_state.assessment = result
    st.session_state.narrative = ""
    st.session_state.raw_output = raw_output
    st.session_state.parse_error = ""
    st.session_state.cache_status = cache_status
    st.session_state.current_payload = payload
    st.session_state.current_model = model
    st.session_state.history.insert(
        0,
        {
            "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
            "symptoms": payload["symptoms"],
            "severity": payload["severity"],
            "urgency": result["urgency_level"],
        },
    )
    del st.session_state.history[100:]


def urgency_notice(level: str) -> None:
    messages = {
        "LOW": (st.success, "Low urgency signal", "Monitor symptoms and follow the guidance below."),
        "MEDIUM": (st.info, "Medical review advised", "Contact a qualified healthcare professional soon."),
        "HIGH": (st.warning, "Prompt medical care advised", "Seek same-day professional assessment."),
        "EMERGENCY": (st.error, "Seek emergency help now", "Call your local emergency number or go to the nearest emergency department immediately."),
    }
    renderer, title, body = messages.get(level, messages["MEDIUM"])
    renderer(body, title=title)


def render_results(model: str) -> None:
    result = st.session_state.assessment
    if not result:
        return

    st.divider()
    st.header("Your educational guidance", anchor=False)
    st.html(f'<div class="safety-strip"><b>Safety reminder:</b> {DISCLAIMER}</div>')
    urgency_notice(result["urgency_level"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Urgency", result["urgency_level"])
    m2.metric("Severity", f'{st.session_state.current_payload["severity"]}/10')
    m3.metric("Response", f'{result.get("response_time_ms", 0)} ms')
    m4.metric("Cache", st.session_state.cache_status)

    overview, actions, questions, warnings, narrative_tab = st.tabs(
        ["Overview", "Next steps", "Doctor questions", "Warning signs", "Live narrative"]
    )
    with overview:
        st.subheader("Symptom summary", anchor=False)
        st.write(result["summary"])
        st.subheader("Possible conditions - education only", anchor=False)
        for index, condition in enumerate(result["possible_conditions"], 1):
            with st.expander(f'{index}. {condition["name"]}'):
                st.write(condition["reason"])
                st.caption("This is general information, not a diagnosis.")
    with actions:
        for item in result["recommended_next_steps"]:
            st.success(item)
    with questions:
        for item in result["questions_for_doctor"]:
            st.info(item)
    with warnings:
        if result["warning_signs"]:
            for item in result["warning_signs"]:
                st.error(item)
        else:
            st.success("No immediate warning signs were identified from the information entered. Continue monitoring.")
    with narrative_tab:
        if not st.session_state.narrative:
            api_key = get_api_key()
            if api_key and result["urgency_level"] != "EMERGENCY":
                with st.spinner("Preparing plain-language guidance..."):
                    chunks = stream_narrative(
                        model=model,
                        api_key=api_key,
                        payload=st.session_state.current_payload,
                        assessment=result,
                    )
                    try:
                        st.session_state.narrative = st.write_stream(chunks)
                    except Exception:
                        st.session_state.narrative = result["summary"]
                        st.info("The live explanation is unavailable. Your structured guidance is still available in the other tabs.")
                        st.write(st.session_state.narrative)
            else:
                demo_text = result.get("narrative") or "Review the structured guidance and contact a healthcare professional if symptoms persist or worsen."
                st.session_state.narrative = st.write_stream(_word_stream(demo_text))
        else:
            st.write(st.session_state.narrative)

    report = build_pdf_report(st.session_state.current_payload, result, DISCLAIMER)
    st.download_button(
        "Download guidance report",
        data=report,
        file_name="pulsenova_educational_report.pdf",
        mime="application/pdf",
        icon=":material/download:",
        width="stretch",
    )
    st.caption("The downloaded report repeats the educational-use warning and should be shared with a clinician for context only.")


def _word_stream(text: str):
    for word in text.split():
        yield word + " "
        time.sleep(0.018)


initialize_state()

CURSOR_FX = st.components.v2.component(
    "pulsenova_cursor_fx",
    html="""
      <button id="sidebar-toggle" type="button" aria-label="Open settings sidebar"><span>☰</span><em>Menu</em></button>
      <div id="progress"></div><div id="halo"></div><div id="sparks"></div>
    """,
    css="""
      #sidebar-toggle{position:fixed;z-index:2147483000;left:12px;top:12px;display:flex;align-items:center;gap:8px;height:42px;padding:0 14px;border:1px solid #4B78FF;border-radius:13px;background:rgba(7,10,20,.94);color:#F5F7FF;font:700 13px/1 system-ui;letter-spacing:.02em;cursor:pointer;box-shadow:0 0 22px #1557FF55;backdrop-filter:blur(12px);transition:transform .18s,border-color .18s,box-shadow .18s;animation:menuBreathe 3.6s ease-in-out infinite}
      #sidebar-toggle span{font-size:20px;color:#B7FF2A;line-height:1;animation:menuBars 2.8s ease-in-out infinite} #sidebar-toggle em{font-style:normal}
      #sidebar-toggle:hover,#sidebar-toggle:focus-visible{transform:translateY(-1px) scale(1.03);border-color:#B7FF2A;box-shadow:0 0 28px #B7FF2A44;outline:none}
      #progress{position:fixed;z-index:999999;left:0;top:0;height:3px;width:0;background:linear-gradient(90deg,#1557FF,#B7FF2A);box-shadow:0 0 14px #1557FF;pointer-events:none}
      #halo{display:none}
      .spark{position:fixed;z-index:999999;width:6px;height:6px;border-radius:50%;background:#B7FF2A;box-shadow:0 0 9px #B7FF2A;pointer-events:none;animation:burst .65s ease-out forwards}
      @keyframes burst{to{transform:translate(var(--x),var(--y)) scale(0);opacity:0}}
      @keyframes menuBreathe{50%{box-shadow:0 0 30px #1557FF88,0 0 12px #B7FF2A22}}
      @keyframes menuBars{50%{transform:scaleX(.78);filter:brightness(1.35)}}
      @media(pointer:coarse),(prefers-reduced-motion:reduce){#halo{display:none}.spark{display:none}#sidebar-toggle,#sidebar-toggle span{animation:none}}
    """,
    js="""
export default function(component) {
  const { parentElement } = component
  const halo = parentElement.querySelector('#halo')
  const progress = parentElement.querySelector('#progress')
  const sparks = parentElement.querySelector('#sparks')
  const menu = parentElement.querySelector('#sidebar-toggle')
  const move = (e) => { if (halo) { halo.style.left=e.clientX+'px'; halo.style.top=e.clientY+'px' } }
  const main = document.querySelector('[data-testid="stMain"]')
  const scroll = () => { const d=main || document.documentElement; const p=d.scrollTop/Math.max(1,d.scrollHeight-d.clientHeight); if(progress) progress.style.width=(p*100)+'%' }
  const click = (e) => { if(!sparks) return; for(let i=0;i<9;i++){ const s=document.createElement('i'); s.className='spark'; s.style.left=e.clientX+'px'; s.style.top=e.clientY+'px'; const a=(Math.PI*2*i)/9; const r=28+Math.random()*34; s.style.setProperty('--x',Math.cos(a)*r+'px'); s.style.setProperty('--y',Math.sin(a)*r+'px'); sparks.appendChild(s); setTimeout(()=>s.remove(),700) } }
  const toggleSidebar = () => {
    const sidebar = document.querySelector('[data-testid="stSidebar"]')
    if (!sidebar) return
    const box = sidebar.getBoundingClientRect()
    const closed = box.right < 40 || box.width < 80
    if (closed) {
      const nativeOpen = document.querySelector('[data-testid="stExpandSidebarButton"]')
      if (nativeOpen) { (nativeOpen.querySelector('button') || nativeOpen).click(); return }
      sidebar.style.setProperty('transform','translateX(0)','important')
      sidebar.style.setProperty('left','0','important')
      sidebar.style.setProperty('visibility','visible','important')
      sidebar.style.setProperty('z-index','2147482999','important')
      sidebar.dataset.pulsenovaForcedOpen = 'true'
      menu?.setAttribute('aria-expanded','true')
    }
  }
  window.addEventListener('pointermove',move); main?.addEventListener('scroll',scroll,{passive:true}); window.addEventListener('click',click); scroll()
  if (menu) menu.onclick = toggleSidebar
  return () => { window.removeEventListener('pointermove',move); main?.removeEventListener('scroll',scroll); window.removeEventListener('click',click); if(menu) menu.onclick=null }
}
    """,
)

inject_design(st.session_state.appearance == "Daylight")
CURSOR_FX(key="global_cursor_fx", height=12)

def sync_theme_toggle() -> None:
    st.session_state.appearance = "Daylight" if st.session_state.theme_toggle else "Midnight"


st.session_state.setdefault("theme_toggle", st.session_state.appearance == "Daylight")
with st.container(horizontal=True, horizontal_alignment="right", key="appearance_toolbar"):
    st.toggle(
        "Day mode",
        key="theme_toggle",
        on_change=sync_theme_toggle,
    )

with st.sidebar:
    st.title("PulseNova AI")
    st.caption("Clarity for the next conversation")
    st.html(
        """
        <div class="side-console">
          <div class="side-console-head">Guidance console <span class="side-live"><i class="live-dot"></i> Ready</span></div>
          <div class="side-journey">
            <span class="side-step"><b>1</b>Describe</span><i class="side-link"></i>
            <span class="side-step"><b>2</b>Review</span><i class="side-link"></i>
            <span class="side-step"><b>3</b>Discuss</span>
          </div>
        </div>
        """
    )
    st.error(DISCLAIMER, title="Medical safety notice")
    st.subheader("Model configuration", anchor=False)
    model = st.selectbox("OpenAI model", MODELS, help="Used only when OPENAI_API_KEY is configured.")
    cache_mode = st.segmented_control(
        "LLM cache",
        ["Off", "In memory", "SQLite"],
        default="In memory",
        key="llm_cache_mode",
        help="Both caches are isolated to your session. SQLite writes to temporary server storage, which can reset on reboot or redeploy.",
    ) or "Off"
    if cache_mode == "SQLite":
        st.warning("SQLite caching stores prompts and responses in a session-specific temporary server database. Cloud restarts may erase it. Avoid identifying information.")
    language = st.selectbox("Answer language", LANGUAGES)
    if get_api_key():
        st.success("OpenAI connection is configured.")
    else:
        st.info("No API key found. Safe local demo mode is active.")
    with st.expander("How the AI pipeline works"):
        st.markdown(
            "1. Validate inputs and scan red flags.\n"
            "2. Build reusable PromptTemplate and ChatPromptTemplate messages.\n"
            "3. Run ChatOpenAI through LLMChain.\n"
            "4. Safely validate structured JSON.\n"
            "5. Stream a plain-language explanation."
        )
        with st.popover("Message demo"):
            for message in create_message_demo():
                st.code(f"{message.__class__.__name__}: {message.content}")

st.html(
    """
    <div class="signal-deck">
      <div class="signal-copy">
        <small>Live guidance workspace</small>
        <strong>A clearer next step starts here.</strong>
        <p>Safety checks activate before any AI assessment.</p>
      </div>
      <div class="signal-right">
        <div class="signal-chips">
          <span class="signal-chip"><i class="live-dot"></i> Safety engine online</span>
          <span class="signal-chip">◇ Private session</span>
          <span class="signal-chip">◎ Multilingual ready</span>
        </div>
        <div class="signal-track"></div>
      </div>
    </div>
    <section class="nova-hero"><div class="pulse-line"></div>
      <div class="nova-kicker">Safety-first health intelligence</div>
      <h1>Pulse<span>Nova</span></h1>
      <p>Describe what you are experiencing and receive structured, calm, educational guidance designed to help you prepare for a conversation with a healthcare professional.</p>
    </section>
    <div class="feature-grid">
      <div class="feature-card"><b>Signal, not diagnosis</b>Structured summaries, possible explanations, and next steps.</div>
      <div class="feature-card"><b>Safety gate first</b>Red-flag screening runs before any AI request.</div>
      <div class="feature-card"><b>Your data, your mode</b>Session storage by default; persistent cache only when chosen.</div>
    </div>
    <div class="safety-strip"><b>EDUCATIONAL SYSTEM // NOT A DIAGNOSIS</b><br>PulseNova cannot replace a licensed clinician, emergency service, examination, diagnosis, or treatment. If you may be in danger, call your local emergency number immediately.</div>
    """
)

view = st.segmented_control(
    "Workspace",
    ["New assessment", "Session history", "Analytics", "Implementation guide"],
    default="New assessment",
    key="workspace_view",
    required=True,
    label_visibility="collapsed",
)

if view == "New assessment":
    with st.expander("Voice symptom input", icon=":material/mic:"):
        voice_audio = st.audio_input("Record your symptoms", help="Your recording is transcribed only when you click the button below.")
        if voice_audio and st.button("Transcribe recording", icon=":material/graphic_eq:", width="stretch"):
            try:
                with st.spinner("Transcribing your recording..."):
                    st.session_state["other_symptoms"] = transcribe_voice(voice_audio)
                    st.session_state.voice_text = st.session_state["other_symptoms"]
                st.success("Transcription added to the symptom form.")
            except Exception as exc:
                st.error(service_error_message(exc) if get_api_key() else "Configure OPENAI_API_KEY to enable transcription.", title="Voice transcription unavailable")
    with st.form("assessment_form", border=True):
        st.html("""
        <div class="form-intro">
          <div class="form-pulse"></div>
          <div><span>Guided symptom assessment</span><h2>Tell us what is happening</h2><p>Share only what you are comfortable providing. We will organize it into a safer clinician-ready summary.</p></div>
        </div>
        """)
        row1, row2 = st.columns(2)
        with row1:
            age = st.text_input("Age", placeholder="e.g. 35", max_chars=3)
            gender = st.selectbox("Gender", GENDERS)
            symptoms = st.multiselect("Symptoms", COMMON_SYMPTOMS)
            other_symptoms = st.text_input("Other symptoms", placeholder="Describe anything not listed", key="other_symptoms")
        with row2:
            duration = st.selectbox("How long have symptoms been present?", DURATIONS)
            severity = st.slider("Severity", 1, 10, 4, help="1 is mild; 10 is the worst imaginable.")
            conditions = st.text_area("Existing medical conditions", placeholder="Include relevant diagnoses or write none", height=80)
            medications = st.text_area("Current medications", placeholder="Include prescriptions, over-the-counter medicines, or supplements", height=80)
        notes = st.text_area("Additional notes", placeholder="What makes symptoms better or worse? Any recent changes?", height=80)
        consent = st.checkbox("I understand this is educational guidance and not a medical diagnosis.")
        submitted = st.form_submit_button(
            "Assess symptoms safely",
            icon=":material/health_and_safety:",
            type="primary",
            width="stretch",
        )

    if submitted:
        payload = patient_payload(
            age, gender, symptoms, other_symptoms, duration, severity,
            conditions, medications, notes, language,
        )
        errors = validate_payload(payload, consent)
        if errors:
            st.error("\n".join(f"- {item}" for item in errors), title="Please check the form")
        else:
            with st.spinner("Running safety checks and preparing guidance..."):
                run_assessment(payload, model, cache_mode)

    render_results(model)
    if st.session_state.parse_error:
        with st.expander("Response diagnostics"):
            st.error(st.session_state.parse_error)
            st.code(st.session_state.raw_output or "No raw output available", language="json")

elif view == "Session history":
    st.header("Session history", anchor=False)
    st.info("History exists only in this browser session and disappears when the session ends.")
    if st.session_state.history:
        st.dataframe(pd.DataFrame(st.session_state.history), width="stretch", hide_index=True)
    else:
        st.info("Complete an assessment to start your session history.")

elif view == "Analytics":
    st.header("Urgency analytics", anchor=False)
    st.caption("Anonymous counts from this browser session only.")
    if st.session_state.history:
        chart_data = (
            pd.DataFrame(st.session_state.history)["urgency"]
            .value_counts()
            .rename_axis("Urgency")
            .reset_index(name="Assessments")
        )
        st.bar_chart(chart_data, x="Urgency", y="Assessments", alt="Number of session assessments by urgency level")
    else:
        st.info("No session data is available yet.")

else:
    st.header("Step-by-step implementation", anchor=False)
    st.caption("The entire application is contained in medical.py; requirements.txt installs its dependencies. Use Streamlit Secrets on Community Cloud and .env for local configuration.")
    steps = [
        ("1", "Load configuration", "Read OPENAI_API_KEY from Streamlit Secrets, environment variables, or a local .env file. Keep credentials out of GitHub."),
        ("2", "Define safety rules", "The system prompt forbids diagnosis, treatment claims, and false certainty."),
        ("3", "Build the schema", "The model is constrained to the exact summary, conditions, urgency, next steps, questions, and warning-sign fields."),
        ("4", "Create reusable prompts", "PromptTemplate carries patient variables; ChatPromptTemplate combines system and human messages."),
        ("5", "Configure ChatOpenAI", "The selected model is wrapped in a reusable LLMChain for structured assessment."),
        ("6", "Validate inputs first", "Age, symptoms, and consent are checked before any API request can happen."),
        ("7", "Run the emergency gate", "High-risk phrases and extreme severity trigger immediate emergency guidance before the LLM."),
        ("8", "Choose a cache", "Memory and SQLite caches are isolated per session. SQLite uses temporary server storage and is not durable on Community Cloud."),
        ("9", "Parse defensively", "Markdown fences are removed, JSON is decoded in try/except, and every required field is validated."),
        ("10", "Stream the narrative", "llm.stream() yields chunks into st.write_stream() for a responsive experience."),
        ("11", "Render and export", "The dashboard applies urgency colors, session history, analytics, and an educational PDF report."),
        ("12", "Enhance access", "Multilingual output, microphone transcription, responsive design, appearance modes, and reduced-motion support complete the experience."),
    ]
    for number, title, body in steps:
        with st.container(border=True):
            st.subheader(f"{number}. {title}", anchor=False)
            st.write(body)

st.divider()
st.caption("PulseNova AI is an educational programming prototype, not a medical device. Consult a qualified professional for diagnosis and treatment.")
