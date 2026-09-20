"""
Small text helpers that more than one feature needs.

Three files used to carry their own copy of "lowercase it and squash the
spaces", and two had their own copy of the URL version. They live here
now, so one change fixes all of them -- and the reason for each is
written down once instead of three times.
"""


def normalize_text(text: str) -> str:
    """Lowercase, and turn any run of spaces, tabs or newlines into one
    space, so two spellings of the same words compare as equal:
    "  Too   BIG " becomes "too big".
    """
    return " ".join(text.split()).lower()


def normalize_url(url: str) -> str:
    """Drop the parts that make one page look like two: http vs https, a
    query string, a trailing slash, capitals.

    Tavily returns the same posting under http:// and https://, and with
    different query strings. Counting those as two postings would inflate
    every number the market check reports.
    """
    return url.split("://", 1)[-1].split("?")[0].rstrip("/").lower()
