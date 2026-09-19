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

import json
from .clients import groq_client, tavily_client
from .roadmap import format_sources_for_prompt

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


def _normalize_text(text: str) -> str:
    """Lowercase and squash extra spaces, so a quote still matches the
    answer when only its capitals or spacing differ."""
    return " ".join(text.split()).lower()


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


def _ask_groq_for_json(prompt: str, what: str) -> dict:
    """Send one prompt to Groq in JSON mode and return the reply as a dict.

    Every Groq call in this file starts the same way, so it lives here
    once. `what` names the step in the error message, e.g. "grading".
    """
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )

    raw_text = response.choices[0].message.content

    # Same rule as build_roadmap: a reply with the wrong shape raises,
    # instead of quietly turning into an empty result.
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as err:
        raise ValueError(f"Groq's reply for {what} wasn't JSON: {raw_text[:200]!r}") from err


def _checked_questions(data: dict, source_text: str, source_name: str, count: int) -> list[dict]:
    """The checks every list of generated questions gets.

    There must be a list; extra questions are cut to `count`; and each
    question needs its text plus a `based_on` that really appears in the
    source (the CV or the job posting) -- otherwise the question is about
    something that isn't there.
    """
    items = data.get("questions")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Groq's reply for {source_name} questions had no list: {str(data)[:200]}")

    # More questions than asked for is easy to fix: keep the first ones.
    items = items[:count]
    for item in items:
        question = item.get("question")
        based_on = item.get("based_on")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Groq gave a {source_name} question with no text: {item!r}")
        if not isinstance(based_on, str) or not based_on.strip():
            raise ValueError(f"Groq gave a {source_name} question with no based_on: {item!r}")
        # Same check as a quote in grading: the words must really be there.
        if not _quote_is_in(based_on, source_text):
            raise ValueError(
                f"A {source_name} question is based on words that aren't in the {source_name}: {based_on!r}"
            )
    return items


def _check_key_points(key_points) -> None:
    """A question needs 3 to 5 key points, each one real text."""
    if (
        not isinstance(key_points, list)
        or not 3 <= len(key_points) <= 5
        or not all(isinstance(point, str) and point.strip() for point in key_points)
    ):
        raise ValueError(f"A question needs 3 to 5 key points, got: {key_points!r}")


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

    data = _ask_groq_for_json(prompt, "grading")

    marks = data.get("marks")
    if not isinstance(marks, list) or not marks:
        raise ValueError(f"Groq's grading reply had no list of marks: {str(data)[:200]}")

    feedback = data.get("feedback")
    if not isinstance(feedback, str) or not feedback.strip():
        raise ValueError(f"Groq's grading reply had no feedback: {str(data)[:200]}")

    # The teacher checks the helper's slip: every key point number must
    # be there exactly once -- none skipped, none repeated, none made up.
    expected = set(range(1, len(question["key_points"]) + 1))
    returned = [m.get("point") for m in marks]
    if len(returned) != len(expected) or set(returned) != expected:
        raise ValueError(f"Groq didn't mark each key point exactly once: {returned}")

    # Every mark must be one of the three words we know how to score.
    for m in marks:
        if m.get("mark") not in MARK_VALUES:
            raise ValueError(f"Groq gave a mark that isn't allowed: {m.get('mark')!r}")

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

    data = _ask_groq_for_json(prompt, "CV questions")

    return [
        {
            "type": "cv",
            "skill_id": None,
            "question": item["question"],
            "key_points": list(CV_KEY_POINTS),
            "source_url": None,
            "based_on": item["based_on"],
        }
        for item in _checked_questions(data, cv_text, "CV", count)
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

    data = _ask_groq_for_json(prompt, "job posting questions")

    questions = []
    for item in _checked_questions(data, posting_text, "job posting", count):
        _check_key_points(item.get("key_points"))
        questions.append({
            "type": "job_posting",
            "skill_id": None,
            "question": item["question"],
            "key_points": item["key_points"],
            "source_url": None,
            "based_on": item["based_on"],
        })
    return questions



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

    data = _ask_groq_for_json(prompt, f"{skill['name']} questions")

    items = data.get("questions")
    if not isinstance(items, list) or not items:
        raise ValueError(f"Groq's reply for {skill['name']} questions had no list: {str(data)[:200]}")

    questions = []
    for item in items[:count]:
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Groq gave a {skill['name']} question with no text: {item!r}")
        # A URL only counts if Tavily really returned it -- otherwise the
        # "real question from the web" label would be made up.
        source_url = item.get("source_url")
        if source_url is not None and source_url not in real_urls:
            raise ValueError(f"Groq gave a URL that Tavily never returned: {source_url!r}")
        _check_key_points(item.get("key_points"))

        questions.append({
            "type": "skill",
            "skill_id": skill["id"],  # from our own roadmap, never from Groq
            "question": question,
            "key_points": item["key_points"],
            "source_url": source_url,
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
