import streamlit as st
import google.generativeai as genai
import os
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
import sys
from pathlib import Path

# Ensure local `modules/` imports work regardless of launch cwd.
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))
from modules.job_role_classifier import JobRoleClassifier
from modules.question_generator import generate_questions, generate_ideal_answers
from modules.mismatch_detector import detect_mismatch
from modules.answer_evaluator import evaluate_batch, evaluate_batch_tfidf
from modules.llm_client import OllamaClient

load_dotenv() ## load all our environment variables
_GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if _GOOGLE_API_KEY:
    genai.configure(api_key=_GOOGLE_API_KEY)

st.set_page_config(
    page_title="Resume Tracking",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

_APP_CSS = """
<style>
  .block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1200px; }
  [data-testid="stHeader"] { background: transparent; }
  [data-testid="stToolbar"] { visibility: hidden; height: 0; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

  .hero {
    border-radius: 18px;
    padding: 26px 26px;
    background: radial-gradient(1200px 500px at 0% 0%, rgba(99, 102, 241, 0.25), transparent 60%),
                radial-gradient(1200px 500px at 100% 0%, rgba(16, 185, 129, 0.18), transparent 55%),
                linear-gradient(180deg, rgba(17, 24, 39, 0.55), rgba(17, 24, 39, 0.25));
    border: 1px solid rgba(255, 255, 255, 0.08);
  }
  .hero h1 { margin: 0; font-size: 38px; line-height: 1.15; }
  .hero p { margin: 10px 0 0 0; opacity: 0.9; font-size: 15px; }

  .card {
    border-radius: 16px;
    padding: 18px 18px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(17, 24, 39, 0.35);
  }
  .muted { opacity: 0.75; }
  .pill {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-size: 12px;
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(99, 102, 241, 0.28);
    margin-right: 8px;
  }
</style>
"""
st.markdown(_APP_CSS, unsafe_allow_html=True)

def _render_role_prediction(pred: dict) -> None:
    role = pred.get("predicted_role", "Unknown/Other")
    conf = float(pred.get("confidence", 0.0) or 0.0)
    rejected = bool(pred.get("rejected", False))
    top_k = pred.get("top_k") or []

    st.subheader("Predicted Job Role (ML Baseline)")

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        st.metric("Role", role)
    with c2:
        st.metric("Confidence", f"{conf*100:.0f}%")
    with c3:
        st.metric("Status", "Low confidence" if rejected else "OK")

    with st.expander("Top predictions", expanded=True):
        if not top_k:
            st.info("No ranking available.")
            return

        for i, item in enumerate(top_k, start=1):
            r = item.get("role", "Unknown/Other")
            c = float(item.get("confidence", 0.0) or 0.0)
            st.write(f"**{i}. {r}**")
            st.progress(min(max(c, 0.0), 1.0), text=f"{c*100:.1f}%")


@st.cache_resource
def _get_embedder():
    """
    Lazy-load the embedding model.
    This avoids hard-crashing the app on machines where Torch/transformers
    are not properly installed. Errors will be shown in the UI instead.
    """
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        raise RuntimeError(
            "Failed to load sentence-transformers embedding model. "
            "Run `pip install -r requirements.txt` and ensure PyTorch installs correctly on Windows."
        ) from e


# Model IDs change over time; set GEMINI_MODEL in .env to pin one from ListModels.
# See: https://ai.google.dev/gemini-api/docs/models
_GEMINI_MODEL_ENV = (os.getenv("GEMINI_MODEL") or "").strip()
_DEFAULT_GEMINI_MODEL_CANDIDATES = [
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-2.5-flash-preview-05-20",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-pro",
]


def get_gemini_repsonse(prompt: str) -> str:
    """Call Gemini with the first model that works (404 = try next)."""
    if _GEMINI_MODEL_ENV:
        order = [_GEMINI_MODEL_ENV] + [
            m for m in _DEFAULT_GEMINI_MODEL_CANDIDATES if m != _GEMINI_MODEL_ENV
        ]
    else:
        order = list(_DEFAULT_GEMINI_MODEL_CANDIDATES)

    last_err: Exception | None = None
    for model_name in order:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if not response.text:
                last_err = RuntimeError(f"Empty response from {model_name}")
                continue
            return response.text
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if "404" in err_str or "not found" in err_str:
                continue
            raise
    raise RuntimeError(
        "No working Gemini model found. Set GEMINI_MODEL in .env to a model from "
        "https://ai.google.dev/gemini-api/docs/models or run: genai.list_models(). "
        f"Last error: {last_err}"
    )

def input_pdf_text(uploaded_file):
    reader=pdf.PdfReader(uploaded_file)
    text=""
    for page in range(len(reader.pages)):
        page=reader.pages[page]
        text+=str(page.extract_text())
    return text

input_prompt="""
Hey Act Like a skilled or very experience ATS(Application Tracking System)
with a deep understanding of tech field,software engineering,data science ,data analyst
and big data engineer. Your task is to evaluate the resume based on the given job description.
You must consider the job market is very competitive and you should provide 
best assistance for improving thr resumes. Assign the percentage Matching based 
on Jd and
the missing keywords with high accuracy
resume:{text}
description:{jd}

I want the response in one single string having the structure
 JD Percentage Match(highlighted bold text): next line  Matching Keywords which are in resume information then  in next line missing keywords(Highlighted bold text) with pointwise but short and concise and next line with spaces for profile summary listed in resume information and at last give some recommendations.   
"""
## streamlit app
st.sidebar.title("Resume Tracking")
st.sidebar.caption("Settings for interview + analysis")

llm_provider = st.sidebar.selectbox(
    "Question generation provider",
    options=["Local (Ollama)", "None (templates)"],
    index=0,
    help="Use a local LLM via Ollama, or use deterministic templates (no LLM).",
)
if llm_provider.startswith("Local"):
    _ollama_host = (os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip()
    _ollama_model = (os.getenv("OLLAMA_MODEL") or "llama3.1:8b").strip()
    try:
        _ollama_ok = OllamaClient(host=_ollama_host, model=_ollama_model).healthcheck()
    except Exception:
        _ollama_ok = False
    if _ollama_ok:
        st.sidebar.success(f"Ollama status: Connected ({_ollama_host})")
    else:
        st.sidebar.error(f"Ollama status: Not reachable ({_ollama_host})")
        st.sidebar.caption("Question generation will fall back to templates.")

skills_text = st.sidebar.text_input(
    "Skills (comma-separated, optional)",
    placeholder="e.g. Python, SQL, Airflow, Docker",
    help="Used to tailor interview questions. If empty, questions are role-based only.",
)

enable_llm_analysis = st.sidebar.checkbox(
    "Enable Gemini ATS analysis (requires valid GOOGLE_API_KEY)",
    value=True,
    help="Uses Google Gemini to generate ATS-style matching, missing keywords, and recommendations.",
)

enable_role_classification = st.sidebar.checkbox(
    "Enable job role classification (beta)",
    value=True,
    help="Predict a likely job role from the resume text using a lightweight ML model (TF-IDF + Logistic Regression).",
)

role_prediction_mode = st.sidebar.selectbox(
    "Role prediction input",
    options=["Resume only", "Resume + Job Description"],
    index=1,
    help="Use resume text only, or combine resume + pasted JD for better target-role alignment.",
)

with st.sidebar.expander("Tips", expanded=False):
    st.write("- Upload a PDF resume for best extraction quality.")
    st.write("- Paste a JD to get matching + missing keywords.")
    st.write("- If you don’t have Ollama, choose templates.")

@st.cache_resource
def _get_job_role_classifier(_model_mtime: float) -> JobRoleClassifier:
    # Prefer a trained model saved on disk; fall back to demo if missing.
    # Train one with: python scripts/train_job_role_classifier.py --data data/job_roles_sample.csv
    return JobRoleClassifier.load_or_demo()


def _model_mtime() -> float:
    try:
        return JobRoleClassifier.default_model_path().stat().st_mtime
    except Exception:
        return 0.0


st.markdown(
    """
<div class="hero">
  <span class="pill">ATS match</span>
  <span class="pill">Mock interview</span>
  <span class="pill">Role prediction</span>
  <h1>Resume Tracking</h1>
  <p>Upload your resume, paste a job description, and get fast matching insights + interview practice.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

c_left, c_right = st.columns([1.25, 1], gap="large")
with c_left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    with st.form("resume_form", clear_on_submit=False):
        st.subheader("Start here")
        jd = st.text_area(
            "Job description (optional)",
            placeholder="Paste the job description to get matching + missing keywords...",
            height=180,
        )
        uploaded_file = st.file_uploader(
            "Resume (PDF only)",
            type="pdf",
            help="Upload a PDF resume for best results.",
        )
        submit = st.form_submit_button("Analyze resume")
    st.markdown("</div>", unsafe_allow_html=True)

with c_right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What you’ll get")
    st.markdown(
        """
- **JD match & missing keywords** (Gemini, if enabled)
- **Predicted job role** (lightweight ML baseline)
- **Mock interview questions** (Ollama or templates)
- **Answer scoring + feedback** (TF‑IDF or embeddings)
""".strip()
    )
    st.caption(
        "Privacy note: when Gemini is enabled, your text is sent to Google’s API. Use local templates/Ollama for offline flows."
    )
    st.markdown("</div>", unsafe_allow_html=True)


if submit and uploaded_file is not None:
    # IMPORTANT: Streamlit buttons are true only on the click run.
    # We persist the pipeline outputs in session_state so the UI doesn't "go black"
    # on the next rerun when the user starts typing answers.
    text = input_pdf_text(uploaded_file)
    predicted_role_for_questions = "Software Engineer"
    resume_role_pred = None
    jd_role_pred = None
    pred: dict | None = None

    if enable_role_classification:
        try:
            clf = _get_job_role_classifier(_model_mtime())
            if role_prediction_mode.startswith("Resume +") and jd and jd.strip():
                role_input_text = text + "\n\nJOB DESCRIPTION:\n" + jd
            else:
                role_input_text = text
            pred = clf.predict(role_input_text, top_k=3, reject_below=0.20)
            predicted_role_for_questions = pred["predicted_role"]
            resume_role_pred = clf.predict(text, top_k=1, reject_below=0.20)["predicted_role"]
            jd_role_pred = clf.predict(jd, top_k=1, reject_below=0.20)["predicted_role"] if jd and jd.strip() else None
        except Exception:
            pred = None

    # questions
    provider = "ollama" if llm_provider.startswith("Local") else "none"
    skills = [s.strip() for s in skills_text.split(",")] if skills_text else []
    question_warning = None
    try:
        questions = generate_questions(
            predicted_role_for_questions,
            skills,
            jd,
            provider=provider,
            ollama_model=(os.getenv("OLLAMA_MODEL") or "llama3.1:8b").strip(),
            ollama_host=(os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip(),
            n_questions=6,
        )
        if not questions:
            raise RuntimeError("Question generator returned no questions.")
    except Exception:
        # Never fail silently: fall back to deterministic templates.
        questions = generate_questions(
            predicted_role_for_questions,
            skills,
            jd,
            provider="none",
            n_questions=6,
        )
        question_warning = (
            "Local Ollama question generation was unavailable. "
            "Using template questions instead."
        )

    st.session_state.pipeline = {
        "resume_text": text,
        "jd": jd,
        "skills": skills,
        "provider": provider,
        "pred": pred,
        "predicted_role_for_questions": predicted_role_for_questions,
        "resume_role_pred": resume_role_pred,
        "jd_role_pred": jd_role_pred,
        "questions": questions,
        "question_warning": question_warning,
    }


if "pipeline" in st.session_state and st.session_state.pipeline:
    data = st.session_state.pipeline
    text = data["resume_text"]
    jd = data["jd"]
    skills = data["skills"]
    provider = data["provider"]
    pred = data["pred"]
    predicted_role_for_questions = data["predicted_role_for_questions"]
    resume_role_pred = data["resume_role_pred"]
    jd_role_pred = data["jd_role_pred"]
    generated_questions: list[str] = data["questions"] or []
    question_warning = data.get("question_warning")

    if pred:
        _render_role_prediction(pred)
        st.markdown("---")

    # Domain mismatch warning (Resume vs JD)
    try:
        mm = detect_mismatch(text, jd, resume_role=resume_role_pred, jd_role=jd_role_pred)
        if mm.is_mismatch:
            st.warning(f"{mm.reason} (similarity={mm.similarity:.2f})")
        else:
            st.caption(f"Resume↔JD alignment score (TF-IDF cosine): {mm.similarity:.2f}")
    except Exception:
        pass

    if question_warning:
        st.warning(question_warning)

    if generated_questions:
        st.subheader("Mock Interview Questions")
        for i, q in enumerate(generated_questions, start=1):
            st.write(f"{i}. {q}")
        st.markdown("---")

        st.subheader("Answer Evaluation (Embeddings)")
        st.caption("Type your answers below, then click Evaluate. Scores are based on similarity to ideal answers.")

        eval_method = st.selectbox(
            "Evaluation method",
            options=["TF-IDF (safe)", "Embeddings (sentence-transformers)"],
            index=0,
            help="If embeddings cause crashes on your machine, use TF-IDF (safe).",
        )

        q_key = "|".join(generated_questions)
        if st.session_state.get("candidate_answers_key") != q_key:
            st.session_state.candidate_answers_key = q_key
            st.session_state.candidate_answers = [""] * len(generated_questions)

        for i, q in enumerate(generated_questions):
            st.write(f"**Q{i+1}. {q}**")
            st.session_state.candidate_answers[i] = st.text_area(
                f"Your answer (Q{i+1})",
                value=st.session_state.candidate_answers[i],
                height=120,
                key=f"ans_{i}",
            )

        # Cache ideal answers + evaluation results so clicking Evaluate doesn't re-run expensive calls repeatedly.
        ideal_key = f"{predicted_role_for_questions}|{jd}|{','.join(skills)}|{'|'.join(generated_questions)}|{provider}"
        if st.session_state.get("ideal_answers_key") != ideal_key:
            st.session_state.ideal_answers_key = ideal_key
            st.session_state.ideal_answers = None
            st.session_state.evaluation_results = None

        if st.button("Evaluate answers"):
            try:
                provider = "ollama" if llm_provider.startswith("Local") else "none"
                skills = [s.strip() for s in skills_text.split(",")] if skills_text else []

                # Step 1: Ideal answers (cached)
                if st.session_state.get("ideal_answers") is None:
                    st.info("Generating ideal answers (first run can take ~30–90s).")
                    prog = st.progress(0.0, text="Starting...")
                    ideal_answers = []
                    # Generate one-by-one but keep UI alive with progress updates.
                    for idx, q in enumerate(generated_questions, start=1):
                        prog.progress((idx - 1) / max(1, len(generated_questions)), text=f"Ideal answer {idx}/{len(generated_questions)}")
                        try:
                            ideal_answers.append(
                                generate_ideal_answers(
                                    [q],
                                    role=predicted_role_for_questions,
                                    skills=skills,
                                    job_description=jd,
                                    provider=provider,
                                    ollama_model=(os.getenv("OLLAMA_MODEL") or "llama3.1:8b").strip(),
                                    ollama_host=(os.getenv("OLLAMA_HOST") or "http://localhost:11434").strip(),
                                )[0]
                            )
                        except Exception:
                            # Fallback ideal answer if Ollama fails for a specific question
                            ideal_answers.append("Provide a structured, concise answer with a concrete example.")
                    prog.progress(1.0, text="Ideal answers ready.")
                    st.session_state.ideal_answers = ideal_answers
                else:
                    ideal_answers = st.session_state.ideal_answers

                # Step 2: Evaluate (cached)
                with st.spinner("Scoring your answers..."):
                    if eval_method.startswith("Embeddings"):
                        embedder = _get_embedder()
                        results = evaluate_batch(
                            generated_questions,
                            st.session_state.candidate_answers,
                            ideal_answers,
                            embedder=embedder,
                        )
                    else:
                        results = evaluate_batch_tfidf(
                            generated_questions,
                            st.session_state.candidate_answers,
                            ideal_answers,
                        )
                st.session_state.evaluation_results = results

                # Aggregate score
                scores = [r["score"] for r in st.session_state.evaluation_results]
                avg = sum(scores) / max(1, len(scores))
                st.metric("Overall score", f"{avg:.0f}/100")

                with st.expander("Per-question results", expanded=True):
                    for i, (q, r) in enumerate(zip(generated_questions, st.session_state.evaluation_results), start=1):
                        st.write(f"**Q{i}. Score: {r['score']}/100 (similarity={r['similarity']:.2f})**")
                        st.write(r["feedback"])
                        with st.expander(f"Show ideal answer (Q{i})"):
                            st.write(r["ideal_answer"])
                st.markdown("---")
            except Exception as e:
                st.error("Evaluation failed. See details below.")
                st.exception(e)

        # If we already evaluated on a previous run, keep displaying results while typing.
        if st.session_state.get("evaluation_results"):
            results = st.session_state.evaluation_results
            scores = [r["score"] for r in results]
            avg = sum(scores) / max(1, len(scores))
            st.metric("Overall score", f"{avg:.0f}/100")
            with st.expander("Per-question results", expanded=False):
                for i, r in enumerate(results, start=1):
                    st.write(f"**Q{i}. Score: {r['score']}/100 (similarity={r['similarity']:.2f})**")
                    st.write(r["feedback"])
    else:
        st.error(
            "Could not generate interview questions. "
            "Check Ollama or switch question provider to 'None (templates)'."
        )

    if enable_llm_analysis:
        if not _GOOGLE_API_KEY:
            st.error(
                "Gemini analysis is enabled but `GOOGLE_API_KEY` is missing. "
                "Add it to your `.env` file (e.g. `GOOGLE_API_KEY=...`) and restart Streamlit."
            )
        else:
            try:
                space="                                                      "
                response=get_gemini_repsonse(text+" as resume information "+space+jd+" as job description "+space+input_prompt)
                st.subheader(response)
            except Exception as e:
                st.error(
                    "Gemini request failed. This is usually caused by an invalid/expired API key or API access issues."
                )
                st.exception(e)
