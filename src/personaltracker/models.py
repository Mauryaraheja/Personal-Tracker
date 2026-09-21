"""
The shapes we expect back from the model.

Groq's JSON mode promises valid JSON, not the right shape. Each reply's
shape is written here once, as a Pydantic model, instead of a handful of
isinstance checks at every call site. A reply that doesn't fit raises
pydantic.ValidationError, which IS a kind of ValueError -- the same error
the app already knows how to show.

Two rules keep this from spreading through the whole app:

1. Only *replies* live here. What the app passes around afterwards stays
   a plain dict, so tracker.py, app.py and the tests don't change.
2. Models check shape, not meaning. "A gap has a skill_id" is shape.
   "There is exactly one gap per roadmap skill", "this quote really
   appears in the answer", "this URL is one we actually fetched" --
   those need the rest of the picture, so they stay in the feature's own
   code, where the reason for them is visible.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

# Text that must actually say something -- "" and "   " are rejected.
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# A question's checklist: 3 to 5 key points, none of them blank.
KeyPoints = Annotated[list[Text], Field(min_length=3, max_length=5)]


class Skill(BaseModel):
    """One roadmap skill. Only the name is required: the roadmap prompt
    asks for the rest, and a missing one shows up as an empty field
    rather than losing the whole roadmap."""

    name: Text
    description: str | None = None
    why_it_matters: str | None = None
    priority: str | None = None
    level_required: str | None = None
    source_url: str | None = None


class RoadmapReply(BaseModel):
    skills: Annotated[list[Skill], Field(min_length=1)]


class Gap(BaseModel):
    skill_id: Text
    status: str
    evidence_from_cv: str | None = None
    suggestion: str | None = None


class GapsReply(BaseModel):
    gaps: Annotated[list[Gap], Field(min_length=1)]


class Mark(BaseModel):
    """One key point's mark. The three words are the ones the score knows
    how to add up, so anything else is a wrong-shaped reply."""

    point: int
    mark: Literal["covered", "partly", "missed"]
    evidence: str | None = None


class MarksReply(BaseModel):
    marks: Annotated[list[Mark], Field(min_length=1)]
    feedback: Text


class CvQuestion(BaseModel):
    question: Text
    based_on: Text


class CvQuestionsReply(BaseModel):
    questions: Annotated[list[CvQuestion], Field(min_length=1)]


class PostingQuestion(BaseModel):
    question: Text
    based_on: Text
    key_points: KeyPoints


class PostingQuestionsReply(BaseModel):
    questions: Annotated[list[PostingQuestion], Field(min_length=1)]


class SkillQuestion(BaseModel):
    """A skill question has no based_on: it comes from the web, not from
    the candidate's CV or the posting they pasted."""

    question: Text
    key_points: KeyPoints
    source_url: str | None = None


class SkillQuestionsReply(BaseModel):
    questions: Annotated[list[SkillQuestion], Field(min_length=1)]


class BatchPostingSkillsReply(BaseModel):
    """The skills found in SEVERAL postings at once, keyed by the number
    each posting was given in the prompt.

    Numbers, not URLs. Groq has mangled a cited URL before (dropping the
    domain), and a mangled key would attach a posting's skills to nothing
    -- or, worse, to the wrong posting, which quietly changes every
    mention_count. A number the model only has to copy back is much
    harder to get wrong, and Python keeps the number -> URL map.

    Shape is checked here: a string key, a list of strings. Whether the
    reply covers every posting that was actually sent is checked in
    market_validation, where that list is in scope.

    An empty list is fine -- a posting can genuinely name nothing we can
    use -- but every entry has to be a string, because the market check
    lowercases and groups these names later.
    """

    postings: dict[str, list[str]] = {}


class SkillGroup(BaseModel):
    canonical_name: Text
    aliases_seen: list[str] = []


class SkillGroupsReply(BaseModel):
    skill_groups: list[SkillGroup] = []
