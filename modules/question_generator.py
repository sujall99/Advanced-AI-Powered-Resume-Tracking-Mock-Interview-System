"""
Interview Question Generation
=============================

Provides a single function requested in your spec:
    generate_questions(role, skills)

Supports:
- Local LLM via Ollama (fully offline after model download)
- A deterministic fallback (no LLM) so the app always works

Output:
- list[str] questions (mix of technical + behavioral)
"""

from __future__ import annotations

from typing import List, Sequence, Optional

from modules.llm_client import OllamaClient


def generate_questions(
    role: str,
    skills: Sequence[str],
    job_description: str | None = None,
    *,
    provider: str = "ollama",
    ollama_model: str = "llama3.1:8b",
    ollama_host: str = "http://localhost:11434",
    n_questions: int = 6,
) -> List[str]:
    role = (role or "").strip() or "Software Engineer"
    skills_list = [s.strip() for s in skills if s and s.strip()]
    jd = (job_description or "").strip()
    n_questions = max(3, min(int(n_questions), 12))

    if provider == "none":
        return _template_questions(role, skills_list, jd, n_questions=n_questions)

    if provider == "ollama":
        prompt = _build_prompt(role, skills_list, jd, n_questions=n_questions)
        client = OllamaClient(host=ollama_host, model=ollama_model)
        text = client.generate(prompt, num_predict=600)
        qs = _parse_questions(text)
        if qs:
            return qs[:n_questions]
        return [text.strip()]

    raise ValueError("Unsupported provider. Use provider='ollama' or provider='none'.")


def generate_ideal_answers(
    questions: Sequence[str],
    *,
    role: str,
    skills: Sequence[str],
    job_description: str | None = None,
    provider: str = "ollama",
    ollama_model: str = "llama3.1:8b",
    ollama_host: str = "http://localhost:11434",
) -> List[str]:
    """
    Generate concise "ideal answers" for evaluation.
    This is used by the embedding-based evaluator.
    """
    qs = [q.strip() for q in questions if q and q.strip()]
    role = (role or "").strip() or "Software Engineer"
    skills_list = [s.strip() for s in skills if s and s.strip()]
    jd = (job_description or "").strip()

    if provider == "none":
        return ["Provide a structured, concise answer with a concrete example." for _ in qs]

    if provider == "ollama":
        client = OllamaClient(host=ollama_host, model=ollama_model)
        answers: List[str] = []
        for q in qs:
            prompt = f"""
You are an expert interviewer.
Write the ideal answer to the following interview question.

Role: {role}
Candidate skills: {", ".join(skills_list) if skills_list else "Not provided"}
Job description: {jd if jd else "Not provided"}

Question: {q}

Rules:
- Keep it concise (6-10 bullet points OR 6-10 short sentences).
- Include key technical terms and decision trade-offs where relevant.
- Avoid fluff.
""".strip()
            answers.append(client.generate(prompt, num_predict=450))
        return answers

    raise ValueError("Unsupported provider. Use provider='ollama' or provider='none'.")


def _build_prompt(role: str, skills: List[str], jd: str, *, n_questions: int) -> str:
    skills_str = ", ".join(skills) if skills else "Not provided"
    jd_str = jd if jd else "Not provided"
    return f"""
You are an expert interview panelist.
Create a concise set of interview questions for the following candidate.

Job role: {role}
Candidate skills: {skills_str}
Job description (JD): {jd_str}

Requirements:
- Generate exactly {n_questions} questions.
- Mix technical and behavioral questions (at least 2 behavioral).
- Prefer questions that directly test JD requirements using the candidate's skills.
- Keep each question to one line.
- Number them 1..{n_questions}.
""".strip()


def _parse_questions(text: str) -> List[str]:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    out: List[str] = []
    for ln in lines:
        # Accept formats like "1) ..." or "1. ..."
        if len(ln) >= 3 and ln[0].isdigit() and (ln[1:3] in {". ", ") "}):
            out.append(ln.split(maxsplit=1)[1] if " " in ln else ln)
    if out:
        return out
    # If model didn't number lines, fall back to all non-empty lines.
    if len(lines) <= 15:
        return lines
    return []


def _template_questions(role: str, skills: List[str], jd: str, *, n_questions: int) -> List[str]:
    # Deterministic fallback for offline/no-LLM mode
    base = [
        f"Walk me through a recent project most relevant to a {role} role.",
        "Tell me about a time you had to debug a production issue. How did you approach it?",
        "How do you prioritize tasks when requirements are unclear or changing?",
        "Explain a design decision you made that improved performance or scalability.",
        "How do you ensure your work is testable and maintainable?",
        "Describe a conflict in a team setting and how you resolved it.",
    ]
    if jd.strip():
        base.insert(1, "From the job description, what do you think the top 3 priorities are for this role and why?")
    if skills:
        # Add a few skill-driven technical prompts
        for s in skills[:6]:
            base.append(f"Deep dive: explain a challenging problem you solved using {s}.")
            base.append(f"What are common pitfalls or best practices when working with {s}?")
    return base[:n_questions]

