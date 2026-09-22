"""Tests for grading an answer against its question's key points.

Groq only judges whether each key point was covered. Everything else --
checking its reply, the key point text, the score -- is plain Python,
and that's what these pin down.
"""

import pytest
from personaltracker import interview
from personaltracker import llm

QUESTION = {
    "id": "qst_002",
    "type": "skill",
    "skill_id": "skl_001",
    "question": "What is chunking in RAG, and why does chunk size matter?",
    "key_points": [
        "Splits documents into pieces",
        "Size is a trade-off",
        "Overlap keeps context",
    ],
    "source_url": "https://example.com/rag-interview-questions",
    "based_on": None,
}


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_an_empty_answer_misses_every_key_point(blank):
    grade = interview.grade_answer(QUESTION, blank)

    assert [m["mark"] for m in grade["marks"]] == ["missed", "missed", "missed"]
    assert grade["score"] == 0
    assert grade["max_score"] == 3


def test_key_point_text_comes_from_the_question():
    grade = interview.grade_answer(QUESTION, "")

    assert [m["key_point"] for m in grade["marks"]] == QUESTION["key_points"]


def test_a_missed_key_point_has_no_evidence():
    grade = interview.grade_answer(QUESTION, "")

    assert all(m["evidence"] is None for m in grade["marks"])


def test_the_grade_says_which_question_it_belongs_to():
    grade = interview.grade_answer(QUESTION, "")

    assert grade["question_id"] == "qst_002"


# ---------------------------------------------------------------------------
# A written answer -- Groq marks it, Python does the rest
# ---------------------------------------------------------------------------

def mark_for(point, mark, evidence=None):
    return {"point": point, "mark": mark, "evidence": evidence}


ANSWER = ("Chunking splits long documents into smaller pieces so they fit the model. "
          "Too big and search gets vague.")

GOOD_REPLY = {
    "marks": [
        mark_for(1, "covered", "splits long documents into smaller pieces"),
        mark_for(2, "partly", "Too big and search gets vague"),
        mark_for(3, "missed"),
    ],
    "feedback": "Say what goes wrong when chunks are too small, and mention overlap.",
}


def test_a_written_answer_gets_groqs_marks(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq(GOOD_REPLY))

    grade = interview.grade_answer(QUESTION, ANSWER)

    assert [m["mark"] for m in grade["marks"]] == ["covered", "partly", "missed"]
    assert grade["score"] == 1.5
    assert grade["feedback"] == GOOD_REPLY["feedback"]


def test_the_score_comes_from_python_not_from_groq(monkeypatch,fake_groq):
    """A score Groq volunteers on its own changes nothing."""
    monkeypatch.setattr(llm, "groq_client", fake_groq({**GOOD_REPLY, "score": 3}))

    grade = interview.grade_answer(QUESTION, ANSWER)

    assert grade["score"] == 1.5


def test_marks_can_come_back_in_any_order(monkeypatch,fake_groq):
    reply = {**GOOD_REPLY, "marks": list(reversed(GOOD_REPLY["marks"]))}
    monkeypatch.setattr(llm, "groq_client", fake_groq(reply))

    grade = interview.grade_answer(QUESTION, ANSWER)

    assert [m["point"] for m in grade["marks"]] == [1, 2, 3]
    assert [m["key_point"] for m in grade["marks"]] == QUESTION["key_points"]


def test_an_empty_answer_never_calls_groq(monkeypatch):
    monkeypatch.setattr(llm, "groq_client", None)  # any call would crash

    grade = interview.grade_answer(QUESTION, "")

    assert grade["score"] == 0


def test_a_reply_that_is_not_json_raises(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq('{"marks": [{"poi'))

    with pytest.raises(ValueError):
        interview.grade_answer(QUESTION, ANSWER)


@pytest.mark.parametrize("marks", [None, [], "all covered"])
def test_a_reply_without_a_list_of_marks_raises(monkeypatch, marks,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({**GOOD_REPLY, "marks": marks}))

    with pytest.raises(ValueError):
        interview.grade_answer(QUESTION, ANSWER)


def test_a_reply_without_feedback_raises(monkeypatch,fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"marks": GOOD_REPLY["marks"]}))

    with pytest.raises(ValueError):
        interview.grade_answer(QUESTION, ANSWER)


@pytest.mark.parametrize("points", [
    [1, 2],        # skipped 3
    [1, 2, 2, 3],  # repeated 2
    [1, 2, 4],     # made up 4
    [0, 1, 2],     # 0 would quietly become the LAST key point
])
def test_each_key_point_must_be_marked_exactly_once(monkeypatch, points,fake_groq):
    reply = {**GOOD_REPLY, "marks": [mark_for(p, "missed") for p in points]}
    monkeypatch.setattr(llm, "groq_client", fake_groq(reply))

    with pytest.raises(ValueError):
        interview.grade_answer(QUESTION, ANSWER)


@pytest.mark.parametrize("bad_mark", ["good", "yes", None])
def test_a_mark_that_is_not_one_of_the_three_raises(monkeypatch, bad_mark,fake_groq):
    reply = {**GOOD_REPLY, "marks": [
        # A real quote, so only the mark-word check can catch this.
        mark_for(1, bad_mark, "splits long documents into smaller pieces"),
        mark_for(2, "missed"), mark_for(3, "missed"),
    ]}
    monkeypatch.setattr(llm, "groq_client", fake_groq(reply))

    with pytest.raises(ValueError):
        interview.grade_answer(QUESTION, ANSWER)



def test_a_quote_may_differ_in_capitals_and_spaces(monkeypatch,fake_groq):
    reply = {**GOOD_REPLY, "marks": [
        mark_for(1, "covered", "SPLITS long   documents"),
        mark_for(2, "missed"), mark_for(3, "missed"),
    ]}
    monkeypatch.setattr(llm, "groq_client", fake_groq(reply))

    grade = interview.grade_answer(QUESTION, ANSWER)

    assert grade["score"] == 1


@pytest.mark.parametrize("mark, evidence", [
    ("partly", None),                             # no quote at all
    ("covered", "   "),                           # an empty quote is "in" every text
    ("missed", "Too big and search gets vague"),  # missed must have no quote
])
def test_evidence_must_follow_the_rules(monkeypatch, mark, evidence,fake_groq):
    reply = {**GOOD_REPLY, "marks": [
        mark_for(1, mark, evidence), mark_for(2, "missed"), mark_for(3, "missed"),
    ]}
    monkeypatch.setattr(llm, "groq_client", fake_groq(reply))

    with pytest.raises(ValueError):
        interview.grade_answer(QUESTION, ANSWER)



# ---------------------------------------------------------------------------
# CV questions -- written by Groq, checked by Python
# ---------------------------------------------------------------------------

CV_TEXT = """Projects
- Built NextRole, an AI career coach, with Groq, Tavily and SQLite.
- Added an OCR fallback for scanned PDF CVs.
Experience
- Data intern: cleaned sales data with pandas.
"""


def cv_question(question, based_on):
    return {"question": question, "based_on": based_on}


def test_cv_questions_are_graded_on_the_four_explanation_points(monkeypatch, fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [
        cv_question("Why did you choose SQLite?", "with Groq, Tavily and SQLite"),
        cv_question("How does the app spot a scanned PDF?", "Added an OCR fallback for scanned PDF CVs"),
    ]}))

    questions = interview.generate_cv_questions(CV_TEXT, "Generative AI Engineer")

    assert len(questions) == 2
    for q in questions:
        assert q["type"] == "cv"
        assert q["key_points"] == interview.CV_KEY_POINTS
        assert q["skill_id"] is None
        assert q["source_url"] is None
    assert questions[0]["based_on"] == "with Groq, Tavily and SQLite"


def test_based_on_may_differ_in_capitals_and_spaces(monkeypatch, fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [
        cv_question("What did you clean?", "CLEANED sales   data"),
    ]}))

    questions = interview.generate_cv_questions(CV_TEXT, "Data Analyst")

    assert len(questions) == 1


def test_a_question_about_something_not_in_the_cv_raises(monkeypatch, fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [
        cv_question("How did you lead the team?", "Led a team of 50 engineers"),
    ]}))

    with pytest.raises(ValueError):
        interview.generate_cv_questions(CV_TEXT, "Generative AI Engineer")


@pytest.mark.parametrize("reply", [
    '{"questions": [{"quest',            # not JSON
    {"interview": []},                   # wrong key
    {"questions": []},                   # empty list
    {"questions": [cv_question("", "cleaned sales data")]},        # no question text
    {"questions": [cv_question("Why pandas?", None)]},             # no based_on
])
def test_a_reply_with_the_wrong_shape_raises(monkeypatch, fake_groq, reply):
    monkeypatch.setattr(llm, "groq_client", fake_groq(reply))

    with pytest.raises(ValueError):
        interview.generate_cv_questions(CV_TEXT, "Data Analyst")


def test_extra_questions_are_cut_to_the_count(monkeypatch, fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [
        cv_question("Why SQLite?", "SQLite"),
        cv_question("Why OCR?", "OCR fallback"),
        cv_question("Why pandas?", "pandas"),
    ]}))

    questions = interview.generate_cv_questions(CV_TEXT, "Data Analyst", count=2)

    assert len(questions) == 2



# ---------------------------------------------------------------------------
# Job-posting questions -- written by Groq, checked by Python
# ---------------------------------------------------------------------------

POSTING_TEXT = """Machine Learning Engineer
You will:
- Deploy models with Docker and Kubernetes
- Build RAG pipelines on top of our document store
Nice to have: experience with LangGraph.
"""

KUBERNETES_POINTS = [
    "Package the model in a container",
    "Kubernetes runs and scales it",
    "Health checks or rollbacks",
]


def posting_question(question, based_on, key_points=KUBERNETES_POINTS):
    return {"question": question, "based_on": based_on, "key_points": key_points}


def test_posting_questions_keep_groqs_key_points(monkeypatch, fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [
        posting_question("How would you deploy a model on Kubernetes?",
                         "Deploy models with Docker and Kubernetes"),
    ]}))

    questions = interview.generate_posting_questions(POSTING_TEXT)

    assert len(questions) == 1
    assert questions[0]["type"] == "job_posting"
    assert questions[0]["key_points"] == KUBERNETES_POINTS
    assert questions[0]["skill_id"] is None
    assert questions[0]["source_url"] is None
    assert questions[0]["based_on"] == "Deploy models with Docker and Kubernetes"


def test_a_question_about_something_not_in_the_posting_raises(monkeypatch, fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [
        posting_question("How would you tune Spark jobs?", "Optimise Spark jobs"),
    ]}))

    with pytest.raises(ValueError):
        interview.generate_posting_questions(POSTING_TEXT)


@pytest.mark.parametrize("key_points", [
    None,                                      # missing
    ["Containers", "Scaling"],                 # too few
    ["a", "b", "c", "d", "e", "f"],            # too many
    ["Containers", "", "Scaling"],             # an empty one
])
def test_a_posting_question_needs_3_to_5_key_points(monkeypatch, fake_groq, key_points):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [
        posting_question("How would you use LangGraph?", "experience with LangGraph", key_points),
    ]}))

    with pytest.raises(ValueError):
        interview.generate_posting_questions(POSTING_TEXT)


@pytest.mark.parametrize("reply", [
    '{"questions": [{"quest',   # not JSON
    {"questions": []},          # empty list
])
def test_a_posting_reply_with_the_wrong_shape_raises(monkeypatch, fake_groq, reply):
    monkeypatch.setattr(llm, "groq_client", fake_groq(reply))

    with pytest.raises(ValueError):
        interview.generate_posting_questions(POSTING_TEXT)



# ---------------------------------------------------------------------------
# Skill questions -- Tavily searches first, then one Groq call
# ---------------------------------------------------------------------------

class FakeTavily:
    """Returns the same search results every time."""

    def __init__(self, results):
        self.results = results

    def search(self, **kwargs):
        return {"results": self.results}


SKILL = {
    "id": "skl_001",
    "name": "RAG pipelines",
    "description": "Combine retrieval with generation.",
    "level_required": "advanced",
}

PAGE_URL = "https://example.com/rag-interview-questions"
PAGE = {"url": PAGE_URL, "content": "Q1. What is chunking in RAG, and why does chunk size matter?"}


def skill_question(question, source_url, key_points=QUESTION["key_points"]):
    return {"question": question, "source_url": source_url, "key_points": key_points}


def test_a_real_question_keeps_its_url_and_our_skill_id(monkeypatch, fake_groq):
    monkeypatch.setattr(interview, "tavily_client", FakeTavily([PAGE]))
    reply_item = skill_question("What is chunking in RAG?", PAGE_URL)
    reply_item["skill_id"] = "skl_999"  # Groq's guess must be ignored
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [reply_item]}))

    questions = interview.generate_skill_questions(SKILL)

    assert len(questions) == 1
    assert questions[0]["type"] == "skill"
    assert questions[0]["skill_id"] == "skl_001"
    assert questions[0]["source_url"] == PAGE_URL
    assert questions[0]["based_on"] is None


def test_groq_may_write_its_own_question_with_no_url(monkeypatch, fake_groq):
    monkeypatch.setattr(interview, "tavily_client", FakeTavily([PAGE]))
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [
        skill_question("How would you evaluate a RAG pipeline?", None),
    ]}))

    questions = interview.generate_skill_questions(SKILL)

    assert questions[0]["source_url"] is None


def test_with_no_search_results_groq_still_writes_questions(monkeypatch, fake_groq):
    monkeypatch.setattr(interview, "tavily_client", FakeTavily([]))
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [
        skill_question("How would you evaluate a RAG pipeline?", None),
    ]}))

    questions = interview.generate_skill_questions(SKILL)

    assert len(questions) == 1


def test_a_url_tavily_never_returned_raises(monkeypatch, fake_groq):
    monkeypatch.setattr(interview, "tavily_client", FakeTavily([PAGE]))
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [
        skill_question("What is chunking?", "https://made-up.example.com/questions"),
    ]}))

    with pytest.raises(ValueError):
        interview.generate_skill_questions(SKILL)


@pytest.mark.parametrize("reply", [
    '{"questions": [{"quest',                                   # not JSON
    {"questions": []},                                          # empty list
    {"questions": [skill_question("", None)]},                  # no question text
    {"questions": [skill_question("What is RAG?", None, ["Retrieval", "Generation"])]},  # 2 key points
])
def test_a_skill_reply_with_the_wrong_shape_raises(monkeypatch, fake_groq, reply):
    monkeypatch.setattr(interview, "tavily_client", FakeTavily([PAGE]))
    monkeypatch.setattr(llm, "groq_client", fake_groq(reply))

    with pytest.raises(ValueError):
        interview.generate_skill_questions(SKILL)


# ---------------------------------------------------------------------------
# Putting the interview together -- order, numbering, skipping
# ---------------------------------------------------------------------------

def fake_generator(kind):
    """Stands in for one of the three generate_* functions."""
    def generate(*args, **kwargs):
        return [{"type": kind, "question": f"a {kind} question"}]
    return generate


def must_not_be_called(*args, **kwargs):
    raise AssertionError("this generator should have been skipped")


@pytest.fixture
def fake_generators(monkeypatch):
    monkeypatch.setattr(interview, "generate_cv_questions", fake_generator("cv"))
    monkeypatch.setattr(interview, "generate_posting_questions", fake_generator("job_posting"))
    monkeypatch.setattr(interview, "generate_skill_questions", fake_generator("skill"))


def test_cv_questions_come_first_then_the_posting_then_skills(fake_generators):
    questions = interview.build_interview(CV_TEXT, "ML Engineer", POSTING_TEXT, [SKILL, SKILL])

    assert [q["type"] for q in questions] == ["cv", "job_posting", "skill", "skill"]


def test_questions_are_numbered_in_the_order_they_are_asked(fake_generators):
    questions = interview.build_interview(CV_TEXT, "ML Engineer", POSTING_TEXT, [SKILL])

    assert [q["id"] for q in questions] == ["qst_001", "qst_002", "qst_003"]


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_a_missing_cv_and_posting_are_skipped_without_asking_groq(
        monkeypatch, fake_generators, missing):
    monkeypatch.setattr(interview, "generate_cv_questions", must_not_be_called)
    monkeypatch.setattr(interview, "generate_posting_questions", must_not_be_called)

    questions = interview.build_interview(missing, "ML Engineer", missing, [SKILL])

    assert [q["type"] for q in questions] == ["skill"]


def test_nothing_to_ask_about_raises(monkeypatch):
    monkeypatch.setattr(interview, "generate_skill_questions", must_not_be_called)

    with pytest.raises(ValueError):
        interview.build_interview("", "ML Engineer", "", [])


# ---------------------------------------------------------------------------
# Quotes shortened with "..." -- seen in a real run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dots", ["...", "…"])
def test_a_quote_shortened_with_dots_is_accepted(monkeypatch, fake_groq, dots):
    """Groq skipped a messy middle part and cut the end, like in a real run."""
    quote = f"Chunking splits long documents {dots} so they fit the model {dots}"
    monkeypatch.setattr(llm, "groq_client", fake_groq({**GOOD_REPLY, "marks": [
        mark_for(1, "covered", quote), mark_for(2, "missed"), mark_for(3, "missed"),
    ]}))

    grade = interview.grade_answer(QUESTION, ANSWER)

    assert grade["score"] == 1


@pytest.mark.parametrize("quote", [
    "Chunking splits long documents ... we used a vector database",  # 2nd piece made up
    "so they fit the model ... Chunking splits long documents",       # pieces out of order
    "...",                                                            # nothing but dots
])
def test_a_shortened_quote_still_has_to_be_real(monkeypatch, fake_groq, quote):
    """A made-up quote earns nothing.

    It no longer costs the whole answer -- see
    test_a_quote_that_isnt_in_the_answer_only_costs_that_point -- but the
    point it was claiming is still refused, which is what this check was
    always for.
    """
    monkeypatch.setattr(llm, "groq_client", fake_groq({**GOOD_REPLY, "marks": [
        mark_for(1, "covered", quote), mark_for(2, "missed"), mark_for(3, "missed"),
    ]}))

    grade = interview.grade_answer(QUESTION, ANSWER)

    assert grade["marks"][0]["mark"] == "missed"
    assert grade["marks"][0]["evidence"] is None
    assert grade["score"] == 0


def test_a_based_on_shortened_with_dots_is_accepted(monkeypatch, fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq({"questions": [
        cv_question("Why SQLite?", "Built NextRole ... with Groq, Tavily and SQLite"),
    ]}))

    questions = interview.generate_cv_questions(CV_TEXT, "Generative AI Engineer")

    assert len(questions) == 1



# ---------------------------------------------------------------------------
# The difficulty ladder -- pure Python, no Groq call at all
# ---------------------------------------------------------------------------

def test_a_strong_answer_makes_the_next_question_harder():
    assert interview.next_level(2, score=3, max_score=4) == 3


def test_a_weak_answer_makes_the_next_question_easier():
    assert interview.next_level(3, score=1, max_score=4) == 2


def test_a_middling_answer_keeps_the_same_level():
    assert interview.next_level(3, score=2, max_score=4) == 3


def test_the_level_never_goes_above_the_top():
    assert interview.next_level(interview.MAX_LEVEL, score=4, max_score=4) == interview.MAX_LEVEL


def test_the_level_never_goes_below_the_bottom():
    assert interview.next_level(interview.MIN_LEVEL, score=0, max_score=4) == interview.MIN_LEVEL


def test_half_marks_count_towards_moving_up():
    # covered + partly + partly = 2 of 3 = 0.67 -- not enough
    assert interview.next_level(2, score=2, max_score=3) == 2
    # covered + covered + partly = 2.5 of 3 = 0.83 -- enough
    assert interview.next_level(2, score=2.5, max_score=3) == 3


def test_a_level_outside_the_ladder_is_an_error():
    with pytest.raises(ValueError, match="level must be"):
        interview.next_level(9, score=1, max_score=4)


def test_a_question_with_no_key_points_is_an_error():
    with pytest.raises(ValueError, match="can't be scored"):
        interview.next_level(2, score=0, max_score=0)


# ---------------------------------------------------------------------------
# The quote check must survive ordinary typing
# ---------------------------------------------------------------------------

REAL_ANSWER = (
    "Because graph neural networks makes a map and connects every order in "
    "real time ,  logistic regression basically we used for classification "
    "that will order will be delivered till this date or not , and gnn "
    "compares from previous orders and connects every piece . that's why we "
    "used ensemble learning ."
)


def test_a_quote_matches_even_when_the_spacing_around_punctuation_differs():
    """The real failure that sent this to the user as a Groq error.

    They typed "ensemble learning ." with a space before the full stop;
    Groq quoted it back as "ensemble learning." without one. One space
    apart, and a correct answer could not be graded at all.
    """
    quote = (
        "Because graph neural networks makes a map and connects every order "
        "in real time , logistic regression basically we used for "
        "classification ... that's why we used ensemble learning."
    )

    assert interview._quote_is_in(quote, REAL_ANSWER)


def test_a_curly_apostrophe_is_the_same_word():
    """Groq swaps ' and ’ freely, and neither is evidence of anything."""
    assert interview._quote_is_in("that’s why we used ensemble learning", REAL_ANSWER)


@pytest.mark.parametrize("quote", [
    "we used a transformer model",                     # never said
    "Because graph neural networks ... we used BERT",  # invented after the dots
    "ensemble learning ... graph neural networks",     # real words, wrong order
    "earn",                                            # part of "learning"
    "   ",                                             # empty
    "... ...",                                         # punctuation only
])
def test_ignoring_punctuation_does_not_let_a_fake_quote_through(quote):
    """Looser about commas, exactly as strict about words."""
    assert not interview._quote_is_in(quote, REAL_ANSWER)


# ---------------------------------------------------------------------------
# A misquote costs one point, not the whole answer
# ---------------------------------------------------------------------------

MISQUOTED = {**GOOD_REPLY, "marks": [
    mark_for(1, "covered", "chunks overlap by 10 percent"),      # never said
    mark_for(2, "covered", "Chunking splits long documents"),    # really said
    mark_for(3, "missed"),
]}


def test_a_quote_that_isnt_in_the_answer_only_costs_that_point(monkeypatch, fake_groq):
    """The real failure, and it happened on a spoken answer.

    Groq marked a key point covered and quoted words the candidate never
    said. That used to raise, and the whole answer went with it -- an
    answer they had just spoken out loud and now had to repeat.
    """
    monkeypatch.setattr(llm, "groq_client", fake_groq(MISQUOTED))

    grade = interview.grade_answer(QUESTION, ANSWER)

    assert [m["mark"] for m in grade["marks"]] == ["missed", "covered", "missed"]


def test_an_unverifiable_quote_is_thrown_away_with_its_credit(monkeypatch, fake_groq):
    """No credit AND no quote: showing it would repeat Groq's invention."""
    monkeypatch.setattr(llm, "groq_client", fake_groq(MISQUOTED))

    grade = interview.grade_answer(QUESTION, ANSWER)

    assert grade["marks"][0]["evidence"] is None
    assert grade["score"] == 1


def test_a_dropped_point_is_said_out_loud(monkeypatch, fake_groq):
    """A mark Python took away must not vanish silently."""
    monkeypatch.setattr(llm, "groq_client", fake_groq(MISQUOTED))

    grade = interview.grade_answer(QUESTION, ANSWER)

    assert "one point" in grade["feedback"]
    assert "aren't in your answer" in grade["feedback"]


# ---------------------------------------------------------------------------
# The follow-up -- a question about the answer just given
# ---------------------------------------------------------------------------

FOLLOW_UP_REPLY = {
    "question": "Why did that make the search vague?",
    "based_on": "Too big and search gets vague",
    "key_points": ["a point", "b point", "c point"],
}


def test_a_follow_up_is_tied_to_the_question_it_follows(monkeypatch, fake_groq):
    monkeypatch.setattr(llm, "groq_client", fake_groq(FOLLOW_UP_REPLY))

    follow_up = interview.generate_follow_up(QUESTION, ANSWER, interview.START_LEVEL)

    assert follow_up["type"] == "follow_up"
    assert follow_up["follows"] == QUESTION["id"]
    assert follow_up["id"] == "qst_002_up"


def test_a_follow_up_is_grounded_in_the_answer_not_the_cv(monkeypatch, fake_groq):
    """The quote has to come from what they SAID -- that is what this
    question is about."""
    monkeypatch.setattr(llm, "groq_client", fake_groq(FOLLOW_UP_REPLY))

    follow_up = interview.generate_follow_up(QUESTION, ANSWER, interview.START_LEVEL)

    assert follow_up["based_on"] in ANSWER


def test_a_follow_up_about_words_never_said_is_refused(monkeypatch, fake_groq):
    monkeypatch.setattr(llm, "groq_client",
                        fake_groq({**FOLLOW_UP_REPLY, "based_on": "we used a vector database"}))

    with pytest.raises(ValueError, match="aren't in the answer"):
        interview.generate_follow_up(QUESTION, ANSWER, interview.START_LEVEL)


def test_a_follow_up_keeps_the_skill_it_came_from(monkeypatch, fake_groq):
    """So the app can still say which skill this question is about."""
    monkeypatch.setattr(llm, "groq_client", fake_groq(FOLLOW_UP_REPLY))

    follow_up = interview.generate_follow_up(QUESTION, ANSWER, interview.START_LEVEL)

    assert follow_up["skill_id"] == QUESTION["skill_id"]
