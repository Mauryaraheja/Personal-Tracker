"""
Mock interview: write questions the way a real interviewer would, then
grade each answer against its question's key points.

Both halves are split the same way as the market check. Groq only does
what needs judgement -- writing a question, deciding whether an answer
covers key point 2 -- and Python does everything it can work out or
check itself: that a question is about something really in the CV or
the job posting, that the reply has the right shape, the key point text
and the score.
"""

from .clients import tavily_client
from .llm import ask_groq_for_json
from .models import (
    CvQuestionsReply,
    MarksReply,
    PostingQuestionsReply,
    SkillQuestionsReply,
)
from .roadmap import format_sources_for_prompt
from .text import normalize_text

# How much each mark is worth. The score is always added up here, in
# Python -- never taken from the model.
MARK_VALUES = {"covered": 1, "partly": 0.5, "missed": 0}

# A CV question is about the candidate's own work, so there's no textbook
# answer to check. It's graded on how well they explain themselves -- the
# same four things every time, so Python fills them in, not Groq.
CV_KEY_POINTS = [
    "Said what they did",
    "Said why they did it that way",
    "Named an alternative they considered",
    "Gave a result, or what they learned",
]


def _quote_is_in(quote: str, text: str) -> bool:
    """True if the quote really comes from the text.

    Groq often shortens a long quote with "..." to skip words -- seen in
    a real run. So each piece between the dots is checked on its own:
    every piece must be in the text, in the same order. A made-up piece
    still fails.
    """
    text = _normalize_text(text)
    pieces = [_normalize_text(piece) for piece in quote.replace("…", "...").split("...")]
    pieces = [piece for piece in pieces if piece]
    if not pieces:
        return False

    position = 0
    for piece in pieces:
        found = text.find(piece, position)
        if found == -1:
            return False
        position = found + len(piece)
    return True


def _questions_about(questions: list, source_text: str, source_name: str, count: int) -> list:
    """Keep the first `count` questions, and check each one is really
    about something the source says.

    The model can only quote the CV or the posting we gave it, so this is
    the same check as a quote in grading: a question about words that
    aren't there is a question about something the candidate never wrote.
    """
    questions = questions[:count]  # more than asked for is easy to fix
    for item in questions:
        if not _quote_is_in(item.based_on, source_text):
            raise ValueError(
                f"A {source_name} question is based on words that aren't in the {source_name}: {item.based_on!r}"
            )
    return questions


def format_key_points_for_prompt(key_points: list[str]) -> str:
    """Number the key points, so Groq can refer to each one by its number."""
    return "\n".join(f"{i}. {point}" for i, point in enumerate(key_points, start=1))


def ask_groq_for_marks(question: dict, answer: str) -> tuple[list[dict], str]:
    """One Groq call: a mark for each numbered key point, plus one tip."""

    key_points_text = format_key_points_for_prompt(question["key_points"])

    prompt = f"""
    An interviewer asked this question:

    {question["question"]}

    A good answer covers these numbered key points:

    {key_points_text}

    Here is the candidate's answer:

    {answer}

    For EACH numbered key point, decide how well the answer covers it,
    based ONLY on what the answer actually says -- do not give credit for
    anything it doesn't say.

    mark must be one of:
    - "covered": the answer clearly makes this point
    - "partly": the answer touches on it, but vaguely or incompletely
    - "missed": the answer doesn't make this point

    evidence: if mark is "covered" or "partly", copy the words from the
    answer that show it -- an exact quote, word for word, not a
    paraphrase. If mark is "missed", this must be null.

    feedback: one concrete tip that would most improve this answer.

    Return a JSON object with exactly two keys:
    - "marks": a list with one object per key point above, each with
      exactly these keys: "point" (the key point's number from the list
      above), "mark", "evidence"
    - "feedback": a string

    Respond with JSON only, no extra text.
    """

    reply = MarksReply.model_validate(ask_groq_for_json(prompt, "grading"))
    marks = [mark.model_dump() for mark in reply.marks]
    feedback = reply.feedback

    # The teacher checks the helper's slip: every key point number must
    # be there exactly once -- none skipped, none repeated, none made up.
    expected = set(range(1, len(question["key_points"]) + 1))
    returned = [m.get("point") for m in marks]
    if len(returned) != len(expected) or set(returned) != expected:
        raise ValueError(f"Groq didn't mark each key point exactly once: {returned}")


    # Evidence must be real. "missed" has no quote. "covered" and
    # "partly" need a quote that is really in the answer -- otherwise
    # Groq could give credit for something the candidate never said.
    for m in marks:
        evidence = m.get("evidence")
        if m["mark"] == "missed":
            if evidence is not None:
                raise ValueError(f"Groq gave a quote for a missed key point: {evidence!r}")
            continue
        # An empty quote would pass the check below: "" is "in" every text.
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"Groq gave no quote for a {m['mark']!r} key point")
        if not _quote_is_in(evidence, answer):
            raise ValueError(f"Groq's quote isn't in the answer: {evidence!r}")

    return marks, feedback


def grade_answer(question: dict, answer: str) -> dict:
    """Grade one answer against its question's key points (an AnswerGrade)."""
    key_points = question["key_points"]

    # An empty answer needs no judging: every key point is missed. Python
    # can work that out itself, so it doesn't cost a Groq call.
    if not answer.strip():
        marks = [
            {"point": i, "mark": "missed", "evidence": None}
            for i in range(1, len(key_points) + 1)
        ]
        feedback = "You didn't write an answer. Try covering the key points below."
    else:
        marks, feedback = ask_groq_for_marks(question, answer)

    # Groq can list the marks in any order -- show them as 1, 2, 3.
    marks.sort(key=lambda m: m["point"])

    # Groq only ever gives point numbers. The text comes from our own
    # question, so it can't come back reworded.
    for mark in marks:
        mark["key_point"] = key_points[mark["point"] - 1]

    return {
        "question_id": question["id"],
        "answer": answer,
        "marks": marks,
        "score": sum(MARK_VALUES[m["mark"]] for m in marks),
        "max_score": len(key_points),
        "feedback": feedback,
    }


def generate_cv_questions(cv_text: str, job_title: str, count: int = 2) -> list[dict]:
    """Ask Groq for questions about specific things the CV says, the way
    an interviewer would. Returns InterviewQuestion dicts without an id --
    ids are added once the whole interview is put together."""

    prompt = f"""
    You are interviewing a candidate for this job: {job_title}

    Here is their CV:

    {cv_text}

    Write {count} interview questions about specific things this CV says
    the candidate did -- a project, a job, a tool they chose. Prefer the
    parts that matter most for this job. Ask the way a real interviewer
    would: why they made a choice, how it worked, what they would do
    differently. Each question must be about ONE specific part of the CV.

    Return a JSON object with a single key "questions", a list of
    {count} objects, each with exactly these keys:
    - "question": the question
    - "based_on": the words from the CV this question is about, copied
      exactly, word for word -- one short phrase or sentence

    Respond with JSON only, no extra text.
    """

    reply = CvQuestionsReply.model_validate(ask_groq_for_json(prompt, "CV questions"))

    return [
        {
            "type": "cv",
            "skill_id": None,
            "question": item.question,
            "key_points": list(CV_KEY_POINTS),
            "source_url": None,
            "based_on": item.based_on,
        }
        for item in _questions_about(reply.questions, cv_text, "CV", count)
    ]


def generate_posting_questions(posting_text: str, count: int = 2) -> list[dict]:
    """Ask Groq for technical questions about what the job posting asks
    for. Unlike a CV question, each one has a textbook answer, so Groq
    also writes the 3-5 key points a strong answer covers."""

    prompt = f"""
    Here is the job posting the candidate is applying for:

    {posting_text}

    You are the interviewer for this job. Write {count} technical
    interview questions about specific things this posting asks for --
    a tool, a skill, a responsibility. Each question must be about ONE
    specific part of the posting.

    For each question, also list 3 to 5 key points that a strong answer
    would cover. Each key point is one short phrase.

    Return a JSON object with a single key "questions", a list of
    {count} objects, each with exactly these keys:
    - "question": the question
    - "based_on": the words from the posting this question is about,
      copied exactly, word for word -- one short phrase or sentence
    - "key_points": a list of 3 to 5 short strings

    Respond with JSON only, no extra text.
    """

    reply = PostingQuestionsReply.model_validate(
        ask_groq_for_json(prompt, "job posting questions"))

    return [
        {
            "type": "job_posting",
            "skill_id": None,
            "question": item.question,
            "key_points": item.key_points,
            "source_url": None,
            "based_on": item.based_on,
        }
        for item in _questions_about(reply.questions, posting_text, "job posting", count)
    ]



def generate_skill_questions(skill: dict, count: int = 1) -> list[dict]:
    """Questions about one roadmap skill. Tavily looks for real interview
    questions first; then ONE Groq call picks good ones from those pages,
    and writes its own only when the pages don't have one that fits."""

    results = tavily_client.search(
        query=f"{skill['name']} interview questions",
        max_results=3,
        search_depth="advanced",
    ).get("results", [])
    real_urls = {r.get("url") for r in results}

    prompt = f"""
    Skill: {skill["name"]}
    What it involves: {skill["description"]}
    Level needed: {skill["level_required"]}

    Here are web pages with real interview questions about this skill:

    {format_sources_for_prompt(results) or "(no pages found)"}

    Write {count} interview questions about this skill, at the level
    above. Prefer a real question from the pages when one fits the level
    -- then set "source_url" to that page's URL, copied exactly. If no
    question on the pages fits, write your own and set "source_url" to
    null.

    For each question, also list 3 to 5 key points that a strong answer
    would cover. Each key point is one short phrase.

    Return a JSON object with a single key "questions", a list of
    {count} objects, each with exactly these keys:
    - "question": the question
    - "source_url": the page's URL, or null if you wrote it yourself
    - "key_points": a list of 3 to 5 short strings

    Respond with JSON only, no extra text.
    """

    reply = SkillQuestionsReply.model_validate(
        ask_groq_for_json(prompt, f"{skill['name']} questions"))

    questions = []
    for item in reply.questions[:count]:
        # A URL only counts if Tavily really returned it -- otherwise the
        # "real question from the web" label would be made up.
        if item.source_url is not None and item.source_url not in real_urls:
            raise ValueError(f"Groq gave a URL that Tavily never returned: {item.source_url!r}")

        questions.append({
            "type": "skill",
            "skill_id": skill["id"],  # from our own roadmap, never from Groq
            "question": item.question,
            "key_points": item.key_points,
            "source_url": item.source_url,
            "based_on": None,
        })

    return questions


def build_interview(cv_text: str | None, job_title: str, posting_text: str | None,
                    skills: list[dict]) -> list[dict]:
    """Put a whole interview together, in the order a real interviewer
    asks: the CV first, then the job posting, then the role's skills.

    A missing CV or posting is simply skipped -- no Groq call for it.
    Every question gets its id here, so they're numbered in the order
    they're asked.
    """
    has_cv = bool(cv_text and cv_text.strip())
    has_posting = bool(posting_text and posting_text.strip())
    if not has_cv and not has_posting and not skills:
        raise ValueError("Nothing to ask about -- add a CV, a job posting or a skill.")

    questions = []
    if has_cv:
        questions += generate_cv_questions(cv_text, job_title)
    if has_posting:
        questions += generate_posting_questions(posting_text)
    for skill in skills:
        questions += generate_skill_questions(skill)

    for i, question in enumerate(questions, start=1):
        question["id"] = f"qst_{i:03d}"

    return questions
