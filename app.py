# app.py — AI ART HELPER (Groq-powered + CrewAI)
from dotenv import load_dotenv
load_dotenv()

import os
import io
import traceback
import streamlit as st
from groq import Groq

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from xml.sax.saxutils import escape

# -------------------------
# IMPORT CREWAI
# -------------------------
from crewai import Agent, Task, Crew, LLM

# -------------------------
# GROQ CLIENT
# -------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or ""
if not GROQ_API_KEY:
    st.warning("⚠️ GROQ_API_KEY not set in .env — Groq calls will fail.")
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# -------------------------
# GENERIC GROQ CHAT
# -------------------------
def groq_chat(prompt: str, model="llama-3.1-8b-instant", temperature=0.6):
    if not client:
        return "Groq is not configured. Add GROQ_API_KEY in .env."
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Groq Error: {e}"


# -------------------------
# AI ART PROMPT GENERATOR
# -------------------------
def generate_art_prompt(description: str, style: str):
    system = (
        "You are an expert AI art prompt engineer. "
        "Generate a high-quality prompt with cinematic details, camera setup, "
        "lighting, mood, color palette, realism level, and composition."
    )
    prompt = f"{system}\n\nDescription: {description}\nStyle: {style}"
    return groq_chat(prompt, temperature=0.8)


# -------------------------
# ART STYLE ANALYZER
# -------------------------
def analyze_art_style(description: str):
    system = (
        "You are a professional art critic. "
        "Analyze the description and explain style, genre, lighting, color theory, "
        "emotion, composition, and improvements."
    )
    prompt = f"{system}\n\nArtwork: {description}"
    return groq_chat(prompt, temperature=0.55)


# -------------------------
# CREWAI MULTI-AGENT SYSTEM
# -------------------------
def run_crewai(query: str, role_hint: str = None):

    groq_llm = LLM(
        model="groq/llama-3.1-8b-instant",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.6
    )

    # ---- AGENTS ----
    art_director = Agent(
        name="Art Director",
        role="Concept Creator",
        goal="Turn the user's idea into a high-quality art concept.",
        backstory="A seasoned art director specializing in cinematic visuals.",
        llm=groq_llm,
    )

    style_expert = Agent(
        name="Style Expert",
        role="Style Enhancer",
        goal="Refine the concept with lighting, colors, mood, and references.",
        backstory="Expert in modern and classical art styles.",
        llm=groq_llm,
    )

    photographer = Agent(
        name="Photographer",
        role="Camera Consultant",
        goal="Add perfect camera, lens, framing, and realism settings.",
        backstory="A world-class photographer with cinematic experience.",
        llm=groq_llm,
    )

    # ---- TASKS ----
    task1 = Task(
        description=f"Create the core concept based on the user request: {query}",
        expected_output="A detailed concept with scene, subject, mood, and artistic idea.",
        agent=art_director,
    )

    task2 = Task(
        description="Enhance the concept with lighting, color palette, style, and mood.",
        expected_output="A refined artistic style with detailed visual improvements.",
        agent=style_expert,
    )

    task3 = Task(
        description="Add camera, lens, frame, angle, and composition enhancements.",
        expected_output="Perfect camera settings and cinematography instructions.",
        agent=photographer,
    )

    # ---- CREW PIPELINE ----
    crew = Crew(
        agents=[art_director, style_expert, photographer],
        tasks=[task1, task2, task3],
        verbose=False
    )

    try:
        result = crew.kickoff()
        return str(result)
    except Exception as e:
        return f"CrewAI Error: {e}"


# -------------------------
# PDF GENERATOR
# -------------------------
def generate_pdf(title, body):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20*mm,
        rightMargin=20*mm,
        topMargin=20*mm,
        bottomMargin=20*mm
    )

    title_style = ParagraphStyle(
        "title",
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#e26a4a"),
        spaceAfter=12,
    )

    body_style = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=12,
        leading=16,
    )

    story = [
        Paragraph(escape(title), title_style),
        Spacer(1, 10),
        Paragraph(escape(body).replace("\n", "<br/>"), body_style),
    ]
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="AI Art Helper", page_icon="🎨", layout="centered")
st.title("🎨 AI Art Helper (Groq + CrewAI)")
st.markdown("Generate AI art prompts • Analyze styles • Use multi-agent creative workflow")


# Tabs
tab1, tab2, tab3 = st.tabs(["🎨 Prompt Generator", "🖌️ Style Analyzer", "🤖 CrewAI Helper"])


# TAB 1 — Prompt Generator
with tab1:
    st.header("🎨 Generate AI Art Prompt")
    desc = st.text_area("Describe your idea", height=200)
    style = st.text_input("Preferred Style (Optional)")

    if st.button("✨ Generate Prompt"):
        if not desc.strip():
            st.error("Please enter a description.")
        else:
            result = generate_art_prompt(desc, style)
            st.session_state["prompt"] = result
            st.success("Prompt Generated Successfully!")

    if st.session_state.get("prompt"):
        st.subheader("Generated Prompt")
        edited = st.text_area("Edit Prompt", st.session_state["prompt"], height=250)
        pdf_bytes = generate_pdf("AI Art Prompt", edited)
        st.download_button("📄 Download as PDF", data=pdf_bytes, file_name="ai_art_prompt.pdf")


# TAB 2 — Style Analyzer
with tab2:
    st.header("🖌️ Analyze Art Style")
    desc2 = st.text_area("Describe the artwork", height=220)

    if st.button("🔍 Analyze Style"):
        if not desc2.strip():
            st.error("Please enter a description.")
        else:
            result = analyze_art_style(desc2)
            st.session_state["analysis"] = result
            st.success("Style Analysis Completed!")

    if st.session_state.get("analysis"):
        st.subheader("Style Analysis")
        edited = st.text_area("Edit Analysis", st.session_state["analysis"], height=250)
        pdf_bytes = generate_pdf("Art Style Report", edited)
        st.download_button("📄 Download PDF", data=pdf_bytes, file_name="art_style_report.pdf")


# TAB 3 — CrewAI Multi-Agent Helper
with tab3:
    st.header("🤖 CrewAI Creative Helper")
    query = st.text_area("Ask anything (concept ideas, lighting, style, cinematography)")
    role_hint = st.text_input("Optional role hint (e.g. Photographer, Director, Color Expert)")

    if st.button("🚀 Run CrewAI Workflow"):
        if not query.strip():
            st.error("Please enter a question.")
        else:
            reply = run_crewai(query, role_hint)
            st.subheader("CrewAI Response")
            st.write(reply)


# Footer
st.markdown(
    "<p style='text-align:center; color:gray; margin-top:35px;'>"
    "Developed by <b>Sanika Umape</b> 🎨"
    "</p>",
    unsafe_allow_html=True
)