import streamlit as st
import streamlit.components.v1 as components
from google import genai
from google.genai import types
from pypdf import PdfReader
import json
import re
import html
import hashlib

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Smart Student Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# GEMINI CONNECTION
# ============================================================

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except Exception as e:
    st.error("❌ Gemini API connection failed.")
    st.info(
        "Please check your .streamlit/secrets.toml file "
        "and make sure GEMINI_API_KEY is correct."
    )
    st.stop()

MODEL = "gemini-3.5-flash"

# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "quiz_data": None,
    "quiz_submitted": False,
    "quiz_score": 0,
    "quiz_source": "",
    "analysis_result": "",
    "summary_result": "",
    "questions_result": "",
    "quiz_history": [],
    "voice_question": "",
    "voice_answer": "",
    "last_audio_hash": "",
    "last_spoken_hash": "",
    "voice_processing": False,
    "translated_text": "",
    "translator_source": "",
    "translator_target": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        radial-gradient(circle at top left, #061a2b 0%, #050816 35%, #12001f 100%);
        color: white;
    }

    .main-title {
        text-align: center;
        font-size: 48px;
        font-weight: 900;
        color: #00f7ff;
        text-shadow:
            0 0 5px #00f7ff,
            0 0 15px #00f7ff,
            0 0 30px #00f7ff;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 22px;
        color: #d7d9ff;
        margin-bottom: 8px;
    }

    .tagline {
        text-align: center;
        font-size: 16px;
        color: #b8c0ff;
        margin-bottom: 30px;
    }

    .feature-card {
        border: 1px solid #00eaff;
        border-radius: 18px;
        padding: 25px;
        background: rgba(10, 12, 35, 0.85);
        box-shadow: 0 0 15px rgba(0, 234, 255, 0.15);
        min-height: 160px;
    }

    .metric-card {
        border: 1px solid #00eaff;
        border-radius: 18px;
        padding: 22px 15px;
        background: rgba(10, 12, 35, 0.88);
        box-shadow: 0 0 15px rgba(0, 234, 255, 0.12);
        min-height: 105px;
        text-align: center;
        margin-bottom: 10px;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #00f7ff;
        line-height: 1.2;
    }

    .metric-label {
        margin-top: 8px;
        font-size: 14px;
        color: #b8c0ff;
        letter-spacing: 0.5px;
    }

    .voice-card {
        border: 1px solid #00eaff;
        border-radius: 20px;
        padding: 28px;
        background: rgba(8, 10, 30, 0.92);
        box-shadow:
            0 0 10px rgba(0, 234, 255, 0.15),
            inset 0 0 20px rgba(0, 234, 255, 0.03);
    }

    .answer-card {
        border: 1px solid #8b5cf6;
        border-radius: 20px;
        padding: 25px;
        background: rgba(15, 10, 35, 0.95);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.18);
    }


    /* ========================================================
       NEON BLUE SELECTABLE BUTTONS
       Dark interior + cyan border + bright cyan hover glow.
       ======================================================== */

    div.stButton > button,
    div.stButton > button[kind="primary"],
    div.stButton > button[data-testid="stBaseButton-primary"],
    button[kind="primary"] {
        border-radius: 12px !important;
        border: 1px solid #00ffff !important;
        background: rgba(0,255,255,0.08) !important;
        color: #00ffff !important;
        font-weight: 700 !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 0 0 rgba(0,255,255,0) !important;
    }

    div.stButton > button:hover,
    div.stButton > button[kind="primary"]:hover,
    div.stButton > button[data-testid="stBaseButton-primary"]:hover,
    button[kind="primary"]:hover {
        background: #00ffff !important;
        color: #050510 !important;
        border-color: #00ffff !important;
        transform: translateY(-1px);
        box-shadow:
            0 0 8px #00ffff,
            0 0 20px rgba(0,255,255,0.75),
            0 0 35px rgba(0,255,255,0.35) !important;
    }

    div.stButton > button:active,
    div.stButton > button[kind="primary"]:active,
    div.stButton > button[data-testid="stBaseButton-primary"]:active,
    button[kind="primary"]:active {
        transform: scale(0.98);
        box-shadow: 0 0 10px #00ffff !important;
    }

    .footer {
        text-align: center;
        color: #8f96c8;
        padding: 25px;
        font-size: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER: CLEAN AI TEXT FOR SPEECH
# ============================================================

def clean_for_speech(text):
    """
    Converts Gemini Markdown output into natural spoken text.

    Removes:
    # headings
    ## headings
    ### headings
    **bold**
    *italic*
    bullet points
    numbered markdown
    code blocks
    links
    emojis that can sound strange
    """

    if not text:
        return ""

    text = str(text)

    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", " ", text)

    # Remove inline code
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # Remove markdown headings
    text = re.sub(r"^\s*#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove bold / italic markers
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("*", "")
    text = text.replace("_", " ")

    # Remove markdown bullets
    text = re.sub(r"^\s*[-•]\s*", "", text, flags=re.MULTILINE)

    # Remove numbered list markers
    text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)

    # Convert markdown links to their text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # Remove horizontal rules
    text = re.sub(r"^\s*[-_=]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Remove common markdown symbols
    text = text.replace("#", "")
    text = text.replace(">", "")

    # Remove markdown-style words that sometimes appear in AI output
    text = re.sub(r"\b(?:hash\s*){2,}\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bhash\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\basterisk\b", "", text, flags=re.IGNORECASE)

    # Replace some symbols with natural pauses
    text = text.replace(":", ": ")
    text = text.replace(";", "; ")

    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# ============================================================
# HELPER: SPEAK ANSWER IN BROWSER
# ============================================================

def speak_text(text):
    """
    Uses browser SpeechSynthesis to speak the answer.
    Markdown is removed before speaking.
    """

    clean_text = clean_for_speech(text)

    if not clean_text:
        return

    safe_text = html.escape(clean_text)

    components.html(
        f"""
        <script>

        const textToSpeak = {json.dumps(clean_text)};

        function speakAnswer() {{

            if (!("speechSynthesis" in window)) {{
                console.log("Speech synthesis is not supported.");
                return;
            }}

            window.speechSynthesis.cancel();

            const utterance =
                new SpeechSynthesisUtterance(textToSpeak);

            utterance.lang = "en-IN";
            utterance.rate = 0.95;
            utterance.pitch = 1.02;
            utterance.volume = 1.0;

            const voices = window.speechSynthesis.getVoices();

            let selectedVoice = null;

            const preferredLanguages = [
                "en-IN",
                "en-US",
                "en-GB"
            ];

            for (const language of preferredLanguages) {{
                selectedVoice = voices.find(
                    voice => voice.lang === language
                );

                if (selectedVoice) {{
                    break;
                }}
            }}

            if (selectedVoice) {{
                utterance.voice = selectedVoice;
            }}

            window.speechSynthesis.speak(utterance);
        }}

        window.speechSynthesis.onvoiceschanged = function() {{
            speakAnswer();
        }};

        setTimeout(function() {{
            speakAnswer();
        }}, 300);

        </script>

        <div style="
            padding: 10px;
            font-family: Arial;
            color: #00f7ff;
            text-align: center;
        ">
            🔊 AI is speaking your answer...
        </div>
        """,
        height=60
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🧠 AI STUDY HUB")

    st.markdown("---")

    st.markdown("### 🚀 Available Tools")

    st.write("📂 PDF Upload")
    st.write("📄 Notes Analyzer")
    st.write("📝 AI Summarizer")
    st.write("❓ Questions & MCQs")
    st.write("🎯 Interactive Quiz")
    st.write("📊 Performance Dashboard")
    st.write("🎤 Voice Assistant")
    st.write("🌍 AI Translator")
    st.write("📥 Download Results")

    st.markdown("---")

    st.info(
        "💡 Upload your study material or paste your notes "
        "and let Gemini AI help you learn faster."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🧠 AI SMART STUDENT ASSISTANT</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Your Intelligent Study Companion</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="tagline">
    Learn smarter • Understand faster • Prepare better
    <br>
    ⚡ Powered by Artificial Intelligence
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DASHBOARD CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)

def dashboard_card(value, label, icon=""):
    display_value = f"{icon} {value}".strip()
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{display_value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c1:
    dashboard_card("06", "AI FEATURES")

with c2:
    dashboard_card("24/7", "AVAILABLE")

with c3:
    dashboard_card("⚡", "FAST AI")

with c4:
    dashboard_card("∞", "LEARNING")

st.divider()


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("📂 Upload Your Study Material")

st.write(
    "Upload a PDF and AI will automatically extract its text."
)

uploaded_file = st.file_uploader(
    "📄 Choose a PDF file",
    type=["pdf"]
)

pdf_text = ""

if uploaded_file is not None:

    try:

        reader = PdfReader(uploaded_file)

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                pdf_text += extracted + "\n"

        if pdf_text.strip():

            st.success(
                f"✅ PDF loaded successfully! "
                f"{len(reader.pages)} pages detected."
            )

            with st.expander("👀 Preview PDF Content"):

                st.text_area(
                    "Extracted Text",
                    pdf_text[:5000],
                    height=250,
                    label_visibility="collapsed"
                )

        else:

            st.warning(
                "⚠️ No selectable text was found. "
                "This may be a scanned/image-only PDF."
            )

    except Exception as e:

        st.error("❌ Unable to read this PDF.")
        st.write(e)


# ============================================================
# MANUAL NOTES
# ============================================================

st.header("📚 Or Paste Your Study Material")

manual_text = st.text_area(
    "Study Material",
    height=220,
    placeholder=(
        "Paste your notes, textbook content, "
        "lecture notes or study material here..."
    ),
    label_visibility="collapsed"
)


# ============================================================
# SELECT STUDY MATERIAL
# ============================================================

if pdf_text.strip():

    study_material = pdf_text

    st.info(
        "📂 PDF content selected for AI processing."
    )

elif manual_text.strip():

    study_material = manual_text

    st.info(
        "📝 Pasted study material selected for AI processing."
    )

else:

    study_material = ""


st.divider()


# ============================================================
# AI LEARNING TOOLS
# ============================================================

st.header("🚀 AI Learning Tools")

st.write(
    "Choose a tool and let AI do the heavy lifting."
)

tool1, tool2, tool3 = st.columns(3)


# ============================================================
# NOTES ANALYZER
# ============================================================

with tool1:

    st.subheader("📄 Notes Analyzer")

    st.write(
        "Understand your material with simple explanations "
        "and key points."
    )

    analyze = st.button(
        "🧠 Analyze Notes",
        key="analyze",
        type="primary"
    )


# ============================================================
# SUMMARIZER
# ============================================================

with tool2:

    st.subheader("📝 AI Summarizer")

    st.write(
        "Convert lengthy material into quick revision notes."
    )

    summarize = st.button(
        "⚡ Summarize",
        key="summarize",
        type="primary"
    )


# ============================================================
# QUESTIONS
# ============================================================

with tool3:

    st.subheader("❓ Questions & MCQs")

    st.write(
        "Generate important exam questions and MCQs."
    )

    generate_questions = st.button(
        "🎯 Generate Questions",
        key="questions",
        type="primary"
    )


# ============================================================
# VALIDATION
# ============================================================

if (
    analyze
    or summarize
    or generate_questions
) and not study_material.strip():

    st.warning(
        "⚠️ Please upload a PDF or paste your study material first."
    )


# ============================================================
# NOTES ANALYZER
# ============================================================

if analyze and study_material.strip():

    with st.spinner(
        "🤖 AI is analyzing your study material..."
    ):

        prompt = f"""
You are an expert AI study assistant.

Analyze the following study material.

Use these sections:

Simple Explanation

Explain the topic in very simple student-friendly language.

Key Points

Give 5 important points.

Important Questions

Give 5 important examination questions.

Quick Revision

Give a short revision section containing the most important information.

STUDY MATERIAL:

{study_material}
"""

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            st.session_state.analysis_result = response.text

            st.success("✅ Analysis completed!")

        except Exception as e:

            st.error("❌ AI Error")
            st.write(e)


# ============================================================
# DISPLAY ANALYSIS
# ============================================================

if st.session_state.analysis_result:

    st.markdown(
        '<div class="answer-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state.analysis_result
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "📥 Download Notes Analysis",
        data=st.session_state.analysis_result,
        file_name="AI_Notes_Analysis.txt",
        mime="text/plain",
        key="download_analysis"
    )


# ============================================================
# SUMMARIZER
# ============================================================

if summarize and study_material.strip():

    with st.spinner(
        "⚡ Creating your summary..."
    ):

        prompt = f"""
You are an expert AI study assistant.

Summarize the following study material for a college student.

Give:

Short Summary

Explain the complete topic briefly.

Important Concepts

List the most important concepts.

Exam Revision

Give concise revision notes.

Use simple language.

STUDY MATERIAL:

{study_material}
"""

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            st.session_state.summary_result = response.text

            st.success("✅ Summary generated!")

        except Exception as e:

            st.error("❌ AI Error")
            st.write(e)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

if st.session_state.summary_result:

    st.markdown(
        '<div class="answer-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state.summary_result
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "📥 Download AI Summary",
        data=st.session_state.summary_result,
        file_name="AI_Study_Summary.txt",
        mime="text/plain",
        key="download_summary"
    )


# ============================================================
# QUESTIONS & MCQs
# ============================================================

if generate_questions and study_material.strip():

    with st.spinner(
        "🎯 Generating exam questions..."
    ):

        prompt = f"""
You are an expert college examination question generator.

Based ONLY on the study material below.

Generate:

Important Questions

Generate 5 descriptive examination questions.

Multiple Choice Questions

Generate 5 MCQs.

For every MCQ provide:

Question

A) Option A
B) Option B
C) Option C
D) Option D

Correct Answer

Explanation

Exam Tip

Give one useful exam preparation tip.

STUDY MATERIAL:

{study_material}
"""

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            st.session_state.questions_result = response.text

            st.success("✅ Questions generated!")

        except Exception as e:

            st.error("❌ AI Error")
            st.write(e)


# ============================================================
# DISPLAY QUESTIONS
# ============================================================

if st.session_state.questions_result:

    st.markdown(
        '<div class="answer-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state.questions_result
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    st.download_button(
        "📥 Download Questions & MCQs",
        data=st.session_state.questions_result,
        file_name="AI_Questions_and_MCQs.txt",
        mime="text/plain",
        key="download_questions"
    )


# ============================================================
# INTERACTIVE QUIZ
# ============================================================

st.divider()

st.header("🎯 Interactive AI Quiz")

st.write(
    "Test your knowledge using an AI-generated quiz "
    "based on your study material."
)

quiz_col1, quiz_col2 = st.columns([3, 1])

with quiz_col1:

    generate_quiz = st.button(
        "🚀 Generate Interactive Quiz",
        key="generate_quiz",
        type="primary"
    )

with quiz_col2:

    if st.button(
        "🔄 Reset Quiz",
        key="reset_quiz"
    ):

        st.session_state.quiz_data = None
        st.session_state.quiz_submitted = False
        st.session_state.quiz_score = 0
        st.session_state.quiz_source = ""

        st.rerun()


# ============================================================
# GENERATE QUIZ
# ============================================================

if generate_quiz:

    if not study_material.strip():

        st.warning(
            "⚠️ Please upload a PDF or paste your study material first."
        )

    else:

        with st.spinner(
            "🤖 AI is creating your interactive quiz..."
        ):

            quiz_prompt = f"""
Create a college-level multiple-choice quiz based ONLY
on the following study material.

Create exactly 5 questions.

Return ONLY valid JSON.

Do NOT use markdown.
Do NOT use code fences.
Do NOT add explanations outside the JSON.

Required format:

{{
    "questions": [
        {{
            "question": "Question text",
            "options": [
                "Option A",
                "Option B",
                "Option C",
                "Option D"
            ],
            "answer": "Option A",
            "explanation": "Short explanation"
        }}
    ]
}}

Rules:

1. Exactly 5 questions.
2. Each question must have exactly 4 options.
3. The answer must exactly match one of the four options.
4. Questions must be based ONLY on the supplied material.
5. Make questions useful for college examination preparation.

STUDY MATERIAL:

{study_material}
"""

            try:

                response = client.models.generate_content(
                    model=MODEL,
                    contents=quiz_prompt
                )

                raw_text = response.text.strip()

                raw_text = re.sub(
                    r"^```json\s*",
                    "",
                    raw_text,
                    flags=re.IGNORECASE
                )

                raw_text = re.sub(
                    r"^```\s*",
                    "",
                    raw_text
                )

                raw_text = re.sub(
                    r"\s*```$",
                    "",
                    raw_text
                )

                quiz_data = json.loads(raw_text)

                if (
                    "questions" not in quiz_data
                    or len(quiz_data["questions"]) != 5
                ):

                    raise ValueError(
                        "AI did not generate exactly 5 questions."
                    )

                for q in quiz_data["questions"]:

                    if "question" not in q:
                        raise ValueError(
                            "Question missing."
                        )

                    if "options" not in q:
                        raise ValueError(
                            "Options missing."
                        )

                    if len(q["options"]) != 4:
                        raise ValueError(
                            "Each question must have 4 options."
                        )

                    if "answer" not in q:
                        raise ValueError(
                            "Correct answer missing."
                        )

                    if q["answer"] not in q["options"]:
                        raise ValueError(
                            "Correct answer does not match an option."
                        )

                st.session_state.quiz_data = quiz_data
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = 0
                st.session_state.quiz_source = study_material

                st.success(
                    "🎉 Quiz generated successfully!"
                )

                st.rerun()

            except json.JSONDecodeError:

                st.error(
                    "❌ AI returned an invalid quiz format."
                )

            except Exception as e:

                st.error(
                    "❌ Unable to generate the quiz."
                )

                st.write(e)


# ============================================================
# DISPLAY QUIZ
# ============================================================

if st.session_state.quiz_data is not None:

    quiz_data = st.session_state.quiz_data

    st.markdown("---")

    st.subheader("🧠 Your AI-Generated Quiz")

    if not st.session_state.quiz_submitted:

        st.info(
            "💡 Select one answer for each question "
            "and then click Submit Quiz."
        )

    with st.form("quiz_form"):

        selected_answers = []

        for i, q in enumerate(
            quiz_data["questions"]
        ):

            st.markdown(
                f"### Question {i + 1} / 5"
            )

            st.markdown(
                f"**{q['question']}**"
            )

            answer = st.radio(
                "Choose your answer:",
                q["options"],
                key=f"quiz_answer_{i}",
                index=None
            )

            selected_answers.append(answer)

            st.markdown("---")

        submitted = st.form_submit_button(
            "✅ Submit Quiz",
            type="primary"
        )

    if submitted:

        score = 0

        for i, q in enumerate(
            quiz_data["questions"]
        ):

            selected = selected_answers[i]

            if selected == q["answer"]:
                score += 1

        st.session_state.quiz_score = score
        st.session_state.quiz_submitted = True

        percentage = (score / 5) * 100

        st.session_state.quiz_history.append({
            "score": score,
            "percentage": percentage
        })

        st.rerun()


# ============================================================
# QUIZ RESULTS
# ============================================================

if (
    st.session_state.quiz_data is not None
    and st.session_state.quiz_submitted
):

    quiz_data = st.session_state.quiz_data
    score = st.session_state.quiz_score

    st.divider()

    st.header("🏆 Quiz Results")

    percentage = (score / 5) * 100

    if percentage == 100:

        performance = "🔥 PERFECT SCORE!"

        st.success(
            "🔥 PERFECT SCORE! You absolutely smashed it!"
        )

    elif percentage >= 80:

        performance = "🎉 EXCELLENT"

        st.success(
            "🎉 Excellent work! You're well prepared."
        )

    elif percentage >= 60:

        performance = "👍 GOOD"

        st.info(
            "👍 Good job! A little more revision will help."
        )

    else:

        performance = "📚 NEEDS REVISION"

        st.warning(
            "📚 Keep studying! Review the explanations below."
        )

    r1, r2, r3 = st.columns(3)

    with r1:

        st.metric(
            "Score",
            f"{score} / 5"
        )

    with r2:

        st.metric(
            "Percentage",
            f"{percentage:.0f}%"
        )

    with r3:

        if percentage >= 80:
            grade = "A"
        elif percentage >= 60:
            grade = "B"
        elif percentage >= 40:
            grade = "C"
        else:
            grade = "D"

        st.metric(
            "Grade",
            grade
        )

    st.markdown("---")

    st.subheader("📖 Answer Review")

    quiz_download = []

    quiz_download.append(
        "AI SMART STUDENT ASSISTANT - QUIZ RESULTS"
    )

    quiz_download.append("=" * 50)
    quiz_download.append("")

    quiz_download.append(
        f"Score: {score} / 5"
    )

    quiz_download.append(
        f"Percentage: {percentage:.0f}%"
    )

    quiz_download.append(
        f"Grade: {grade}"
    )

    quiz_download.append(
        f"Performance: {performance}"
    )

    quiz_download.append("")
    quiz_download.append("=" * 50)
    quiz_download.append("ANSWER REVIEW")
    quiz_download.append("")

    for i, q in enumerate(
        quiz_data["questions"]
    ):

        selected = st.session_state.get(
            f"quiz_answer_{i}"
        )

        st.markdown(
            f"### Question {i + 1}"
        )

        st.write(
            q["question"]
        )

        quiz_download.append(
            f"Question {i + 1}: {q['question']}"
        )

        if selected == q["answer"]:

            st.success(
                f"✅ Your answer: {selected}"
            )

            quiz_download.append(
                f"Your answer: {selected}"
            )

            quiz_download.append(
                "Result: CORRECT"
            )

        else:

            st.error(
                f"❌ Your answer: "
                f"{selected if selected else 'Not answered'}"
            )

            st.success(
                f"✅ Correct answer: {q['answer']}"
            )

            quiz_download.append(
                "Your answer: "
                f"{selected if selected else 'Not answered'}"
            )

            quiz_download.append(
                f"Correct answer: {q['answer']}"
            )

            quiz_download.append(
                "Result: WRONG"
            )

        st.info(
            f"💡 Explanation: {q['explanation']}"
        )

        quiz_download.append(
            f"Explanation: {q['explanation']}"
        )

        quiz_download.append("")
        quiz_download.append("-" * 50)
        quiz_download.append("")

    quiz_download_text = "\n".join(
        quiz_download
    )

    st.download_button(
        "📥 Download Quiz Results",
        data=quiz_download_text,
        file_name="AI_Quiz_Results.txt",
        mime="text/plain",
        key="download_quiz_results"
    )

    st.markdown("---")

    if st.button(
        "🔄 Generate New Quiz",
        key="new_quiz"
    ):

        st.session_state.quiz_data = None
        st.session_state.quiz_submitted = False
        st.session_state.quiz_score = 0
        st.session_state.quiz_source = ""

        st.rerun()


# ============================================================
# PERFORMANCE DASHBOARD
# ============================================================

st.divider()

st.header("📊 Student Performance Dashboard")

history = st.session_state.quiz_history

if len(history) == 0:

    st.info(
        "🎯 Complete at least one quiz to unlock your "
        "performance dashboard."
    )

else:

    total_quizzes = len(history)

    percentages = [
        item["percentage"]
        for item in history
    ]

    average_score = (
        sum(percentages) / total_quizzes
    )

    best_score = max(percentages)

    latest_score = percentages[-1]

    if average_score >= 80:

        performance_level = "🟢 Excellent"

        recommendation = (
            "🔥 Excellent performance! Keep practicing "
            "to maintain your strong results."
        )

    elif average_score >= 60:

        performance_level = "🟡 Good"

        recommendation = (
            "👍 Good progress! Revise your weaker topics "
            "and attempt more quizzes."
        )

    else:

        performance_level = "🔴 Needs Improvement"

        recommendation = (
            "📚 Spend more time revising your study material "
            "and take the quiz again."
        )

    d1, d2, d3, d4 = st.columns(4)

    with d1:

        st.metric(
            "📝 Quizzes Attempted",
            total_quizzes
        )

    with d2:

        st.metric(
            "🎯 Average Score",
            f"{average_score:.0f}%"
        )

    with d3:

        st.metric(
            "🏆 Best Score",
            f"{best_score:.0f}%"
        )

    with d4:

        st.metric(
            "📌 Latest Score",
            f"{latest_score:.0f}%"
        )

    st.markdown("---")

    st.subheader("🎯 Current Performance")

    if average_score >= 80:

        st.success(
            f"🟢 {performance_level}"
        )

    elif average_score >= 60:

        st.info(
            f"🟡 {performance_level}"
        )

    else:

        st.warning(
            f"🔴 {performance_level}"
        )

    st.subheader("📈 Performance Trend")

    chart_data = {
        "Quiz": [
            f"Quiz {i + 1}"
            for i in range(total_quizzes)
        ],
        "Score": percentages
    }

    st.line_chart(
        chart_data,
        x="Quiz",
        y="Score"
    )

    st.subheader("💡 Study Recommendation")

    st.info(
        recommendation
    )

    st.subheader("📚 Quiz History")

    history_rows = []

    for i, item in enumerate(history):

        history_rows.append({
            "Quiz": f"Quiz {i + 1}",
            "Score": f"{item['score']} / 5",
            "Percentage": f"{item['percentage']:.0f}%"
        })

    st.table(history_rows)

    st.markdown("---")

    if st.button(
        "🗑️ Reset Performance History",
        key="reset_history"
    ):

        st.session_state.quiz_history = []

        st.success(
            "✅ Performance history cleared!"
        )

        st.rerun()


# =========================================================
# 🌍 AI STUDY TRANSLATOR
# =========================================================

st.divider()
st.header("🌍 AI Study Translator")
st.write("Translate your study material, notes and questions instantly using Gemini AI.")

st.markdown("""
<style>
.translator-card {
    background: rgba(10,10,30,.88);
    border: 1px solid rgba(0,255,255,.35);
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
    box-shadow: 0 0 25px rgba(0,255,255,.08);
}

.translator-result {
    background: rgba(5,8,25,.95);
    border: 1px solid #00ffff;
    border-radius: 15px;
    padding: 20px;
    margin-top: 18px;
    box-shadow: 0 0 15px rgba(0,255,255,.12);
}
</style>
""", unsafe_allow_html=True)

translator_languages = [
    "English", "Tamil", "Hindi", "Telugu", "Malayalam", "Kannada",
    "Bengali", "Marathi", "Gujarati", "Punjabi", "Urdu", "French",
    "German", "Spanish", "Italian", "Portuguese", "Japanese", "Korean",
    "Chinese", "Arabic", "Russian"
]

tr_col1, tr_col2 = st.columns(2)

with tr_col1:
    translator_source = st.selectbox(
        "🌐 From Language",
        translator_languages,
        index=0,
        key="translator_source_select"
    )

with tr_col2:
    translator_target = st.selectbox(
        "🌍 To Language",
        translator_languages,
        index=1,
        key="translator_target_select"
    )

st.markdown('<div class="translator-card">', unsafe_allow_html=True)

translator_input = st.text_area(
    "📝 Enter your text",
    height=220,
    placeholder="Type or paste the text you want to translate here...",
    key="translator_input"
)

st.markdown('</div>', unsafe_allow_html=True)

tr_button_col1, tr_button_col2 = st.columns([1, 1])

with tr_button_col1:
    translate_clicked = st.button(
        "🌐 Translate",
        key="translate_button",
        type="primary"
    )

with tr_button_col2:
    clear_translation = st.button(
        "🗑️ Clear Translation",
        key="clear_translation"
    )

if clear_translation:
    st.session_state.translated_text = ""
    st.session_state.translator_source = ""
    st.session_state.translator_target = ""
    st.rerun()

if translate_clicked:
    if not translator_input.strip():
        st.warning("⚠️ Please enter some text to translate.")

    elif translator_source == translator_target:
        st.info("ℹ️ Source and target languages are the same. Choose a different target language.")

    else:
        with st.spinner("✨ AI is translating your text..."):
            translator_prompt = f"""
You are a professional multilingual translator for a college student.

Translate the following text from {translator_source} to {translator_target}.

IMPORTANT RULES:
1. Preserve the original meaning exactly.
2. Do not summarize.
3. Do not add explanations before or after the translation.
4. Do not remove important information.
5. Preserve paragraphs and formatting where possible.
6. Translate technical and academic terminology accurately.
7. Return ONLY the translated text.

TEXT TO TRANSLATE:
{translator_input}
"""

            try:
                translator_response = client.models.generate_content(
                    model=MODEL,
                    contents=translator_prompt
                )

                translated = (translator_response.text or "").strip()

                if not translated:
                    raise ValueError("Gemini returned an empty translation.")

                st.session_state.translated_text = translated
                st.session_state.translator_source = translator_source
                st.session_state.translator_target = translator_target
                st.success("✅ Translation completed successfully!")

            except Exception as e:
                st.error("❌ Translation failed.")
                st.write(e)

if st.session_state.translated_text:
    st.markdown('<div class="translator-result">', unsafe_allow_html=True)
    st.markdown(
        f"### ✨ {st.session_state.translator_target} Translation"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.text_area(
        "Translated Text",
        value=st.session_state.translated_text,
        height=250,
        key="translation_result_display"
    )

    download_name = (
        f"AI_Translation_"
        f"{st.session_state.translator_source}_to_"
        f"{st.session_state.translator_target}.txt"
    )

    st.download_button(
        "📥 Download Translation",
        data=st.session_state.translated_text,
        file_name=download_name,
        mime="text/plain",
        key="download_translation"
    )



# ============================================================
# 🎤 AI VOICE ASSISTANT
# ============================================================

st.divider()

st.header("🎤 AI Voice Assistant")

st.write(
    "Speak naturally to your AI Smart Student Assistant and receive a spoken answer."
)

# Everything related to voice input stays inside ONE box.
with st.container(border=True):

    st.markdown(
        """
        <div style="padding:8px 8px 2px 8px;">
            <h2 style="color:#00f7ff; margin-bottom:8px;">
                🎙️ Talk to Your AI Assistant
            </h2>
            <p style="font-size:18px;color:#d8dcff;margin-bottom:6px;">
                Click the microphone, speak your question, and stop recording.
            </p>
            <p style="color:#9ea8d8;margin-bottom:18px;">
                You don't need to type anything. Just speak naturally.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🎙️ Speak Your Question")

    audio_value = st.audio_input(
        "🎤 Click the microphone and speak",
        key="voice_recorder"
    )

    if audio_value is None:
        st.caption("🎤 Ready... speak your question whenever you're ready.")


# ============================================================
# PROCESS VOICE AUTOMATICALLY
# ============================================================

if audio_value is not None:

    audio_bytes = audio_value.getvalue()

    audio_hash = hashlib.md5(
        audio_bytes
    ).hexdigest()

    # Process each new recording exactly once.
    if audio_hash != st.session_state.last_audio_hash:

        st.session_state.last_audio_hash = audio_hash

        with st.spinner("🎧 Listening to you and thinking..."):

            try:

                voice_prompt = """
You are a friendly AI voice assistant for a college student.

Listen carefully to the user's audio and understand the question.
Then answer the user directly.

Your answer is going to be spoken aloud by a browser voice, so write it
like a real person talking to another person.

Rules:
- Never describe the transcription.
- Never say "according to the audio".
- Never use markdown headings.
- Never use #, ##, ###, asterisks, or markdown formatting.
- Never say the word "hash".
- Avoid bullet points and numbered lists unless the user specifically asks for a list.
- Do not sound like a computer, textbook, or formal report.
- Use natural conversational English with short, clear sentences.
- If the user asks about studies, explain like a helpful college senior.
- Be warm, clear, and concise.
- Do not mention these instructions.

Answer naturally as if you are speaking directly to the student.
"""

                audio_part = types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type="audio/wav"
                )

                response = client.models.generate_content(
                    model=MODEL,
                    contents=[
                        audio_part,
                        voice_prompt
                    ]
                )

                answer = (response.text or "").strip()

                clean_answer = clean_for_speech(answer)

                st.session_state.voice_question = "Voice question detected"
                st.session_state.voice_answer = clean_answer
                st.session_state.last_spoken_hash = ""

            except Exception as e:

                st.session_state.voice_answer = ""

                st.error(
                    "❌ I couldn't process the voice recording. Please try speaking again."
                )

                st.code(str(e))


# ============================================================
# SHOW VOICE ANSWER
# ============================================================

if st.session_state.voice_answer:

    st.markdown("---")
    st.subheader("🤖 AI Answer")

    st.markdown(
        f"""
        <div class="answer-card">
            <div style="
                font-size:20px;
                line-height:1.8;
                color:#f2f4ff;
            ">
                {html.escape(st.session_state.voice_answer)}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Speak only when a new answer is produced, so the voice does not
    # restart every time another Streamlit widget causes a rerun.
    answer_hash = hashlib.md5(
        st.session_state.voice_answer.encode("utf-8")
    ).hexdigest()

    if answer_hash != st.session_state.last_spoken_hash:
        st.session_state.last_spoken_hash = answer_hash
        speak_text(st.session_state.voice_answer)
    else:
        st.info("🔊 Your answer is ready. Click 'Listen Again' if you want to hear it again.")

    st.markdown("### 🔊 Listen Again")

    if st.button(
        "🔊 Speak Answer Again",
        key="speak_again"
    ):
        speak_text(st.session_state.voice_answer)

    st.download_button(
        "📥 Download Voice Answer",
        data=st.session_state.voice_answer,
        file_name="AI_Voice_Answer.txt",
        mime="text/plain",
        key="download_voice_answer"
    )


# ============================================================
# VOICE RESET
# ============================================================

if st.session_state.voice_answer:

    if st.button(
        "🗑️ Clear Voice Conversation",
        key="clear_voice"
    ):

        st.session_state.voice_question = ""
        st.session_state.voice_answer = ""
        st.session_state.last_audio_hash = ""
        st.session_state.last_spoken_hash = ""

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    🧠 AI Smart Student Assistant

    <br>

    Powered by Gemini AI • Built for Smarter Learning 🚀

    <br><br>

    🎤 Voice • 📚 Study • 🧠 Learn • 🎯 Practice

    </div>
    """,
    unsafe_allow_html=True
)
