"""
Streamlit UI for the AI Career Coach (Personal Tracker).

Flow: Choose a Role -> Your Required Skills (with progress tracking) ->
Upload Your CV -> Next Steps -> Market Insights -> Mock Interview

Uses st.session_state because Streamlit reruns this entire script top-to-bottom
on every interaction (typing, clicking, uploading). Without state, the skills
list generated in step 1 would vanish by the time the user reaches step 3.

Skills and progress are persisted in SQLite (see tracker.py) -- the first time
a role is tracked it costs three API requests (two Groq, one Tavily); every
visit after that is a local database read.
"""

import streamlit as st
from personaltracker import (
    build_interview,
    extract_text_from_pdf,
    grade_answer,
    get_skill_gaps,
    get_or_create_roadmap,
    get_market_validation,
    get_refined_title,
    get_tracker_items,
    update_tracker_status,
)

st.set_page_config(page_title="NextRole", layout="wide")

STATUS_OPTIONS = ["not_started", "in_progress", "completed"]
STATUS_LABELS = {
    "not_started": "Not started",
    "in_progress": "In progress",
    "completed": "Completed",
}

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
# One entry per piece of data that needs to survive a rerun. `uploader_key` is
# not user data -- it's a counter we bump on "Start Over" so the file_uploader
# widget below gets a fresh key and actually clears the previously uploaded
# file. Without changing the key, the widget keeps its own internal state and
# will keep showing the old file even after cv_text/gaps are wiped.
DEFAULTS = {
    "role": None,
    "skills": None,
    "tracker": None,
    "cv_text": None,
    "gaps": None,
    "market_insights": None,
    # The interview's questions, and a dict of grades keyed by question id.
    # Grades start as None, not {}: reset_all() would hand the SAME dict
    # back every time, so old grades would survive "Start Over".
    "interview_questions": None,
    "interview_grades": None,
    "uploader_key": 0,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_all():
    """Wipes every piece of session state so the user starts a clean pass."""
    for key, value in DEFAULTS.items():
        if key == "uploader_key":
            st.session_state[key] += 1  # force a new file_uploader widget
        else:
            st.session_state[key] = value


def _on_status_change(role, skill_id, widget_key):
    """
    Fires when a skill's progress selectbox changes. Writes the new status to
    the database, then updates the in-memory copy in session_state so the UI
    reflects the change immediately without a fresh database read.
    """
    new_status = st.session_state[widget_key]
    update_tracker_status(role, skill_id, new_status)
    for item in st.session_state.tracker:
        if item["skill_id"] == skill_id:
            item["status"] = new_status
            break


def mention_band(count: int) -> str:
    """Describe how widely a skill was mentioned, without quoting a total.

    The denominator was never trustworthy. Some postings come back from
    Tavily with their requirement bullets stripped, so a phrase like
    "3 of 12" silently counted pages we never actually read. A band keeps
    the signal that matters -- common versus rare -- and drops precision
    the data doesn't support.
    """
    if count >= 4:
        return "mentioned across many postings"
    if count >= 2:
        return "mentioned in a few postings"
    if count == 1:
        return "mentioned once"
    # No usable count. Say nothing about frequency rather than invent a
    # number -- a missing field must not render as "mentioned once".
    return "mentioned in these postings"


st.title("NextRole")

with st.sidebar:
    st.button("Start Over", on_click=reset_all, use_container_width=True)

# ---------------------------------------------------------------------------
# Section 1: Choose a Role
# ---------------------------------------------------------------------------
# Every section below this point only renders once its prerequisite data
# exists in session_state -- so the page naturally grows top-to-bottom as
# the user progresses, instead of showing empty/disabled sections upfront.
st.header("Choose a Role")
role_input = st.text_input(
    "Target role",
    value=st.session_state.role or "",
    placeholder="e.g. Backend Engineer",
)

if st.button("Find Required Skills", type="primary"):
    if not role_input.strip():
        st.warning("Enter a role first.")
    else:
        try:
            with st.spinner("Loading required skills for this role..."):
                role = role_input.strip()
                # get_or_create_roadmap only hits Groq/Tavily the first time
                # this role is ever tracked -- after that it reads from the
                # database, which is why this can be slow once and instant
                # every time after.
                skills = get_or_create_roadmap(role)
                tracker = get_tracker_items(role)
            # Only commit to session_state once both calls have actually
            # succeeded -- we don't want a role set with no matching skills,
            # or skills with no matching tracker rows.
            st.session_state.role = role
            st.session_state.skills = skills
            st.session_state.tracker = tracker
            # A new role invalidates anything worked out from the old skills.
            st.session_state.cv_text = None
            st.session_state.gaps = None
            st.session_state.market_insights = None
            st.session_state.interview_questions = None
            st.session_state.interview_grades = None
        except Exception as e:
            st.error(
                "Couldn't load the skill roadmap. If this is the first time "
                "tracking this role, it's usually a Groq or Tavily API issue "
                "(rate limit, network, or bad key). If it's a role you've "
                "tracked before, it's more likely a database problem."
            )
            with st.expander("Technical details"):
                st.exception(e)

# ---------------------------------------------------------------------------
# Section 2: Your Required Skills
# ---------------------------------------------------------------------------
if st.session_state.skills:
    st.divider()
    st.header("Your Required Skills")
    st.caption(f"For: {st.session_state.role}")
    # Looked up by skill_id so each skill card can show/update its own
    # progress without scanning the whole tracker list per skill.
    tracker_by_skill = {t["skill_id"]: t for t in st.session_state.tracker}
    for skill in st.session_state.skills:
        with st.container(border=True):
            col_info, col_status = st.columns([3, 1])
            with col_info:
                st.markdown(f"**{skill.get('name', skill.get('id'))}**")
                if skill.get("description"):
                    st.caption(skill["description"])
            with col_status:
                current_status = tracker_by_skill.get(skill["id"], {}).get(
                    "status", "not_started"
                )
                widget_key = f"status_{skill['id']}"
                st.selectbox(
                    "Progress",
                    STATUS_OPTIONS,
                    index=STATUS_OPTIONS.index(current_status),
                    format_func=lambda s: STATUS_LABELS[s],
                    key=widget_key,
                    on_change=_on_status_change,
                    args=(st.session_state.role, skill["id"], widget_key),
                    label_visibility="collapsed",
                )

    # -----------------------------------------------------------------------
    # Section 3: Upload Your CV
    # -----------------------------------------------------------------------
    st.divider()
    st.header("Upload Your CV")
    uploaded_file = st.file_uploader(
        "PDF only, for now",
        type=["pdf"],
        key=f"cv_uploader_{st.session_state.uploader_key}",
    )

    if uploaded_file is not None and st.button("Analyze Gaps", type="primary"):
        # Extraction happens silently by design: no "Extracting text..."
        # message, and the raw CV text is never displayed in the UI --
        # it was only ever a CLI debugging aid, not a real feature.
        try:
            with st.spinner("Reading your CV..."):
                cv_text = extract_text_from_pdf(uploaded_file)
        except Exception as e:
            st.error(
                "Couldn't read that PDF. If it's a scanned document, make sure "
                "Tesseract and Poppler are installed and on PATH -- otherwise "
                "the file may be corrupted."
            )
            with st.expander("Technical details"):
                st.exception(e)
        else:
            try:
                with st.spinner("Comparing your CV against the required skills..."):
                    gaps = get_skill_gaps(st.session_state.skills, cv_text)
                st.session_state.cv_text = cv_text
                st.session_state.gaps = gaps
            except Exception as e:
                st.error(
                    "Couldn't complete the gap analysis. This is usually a Groq "
                    "API issue (rate limit, network, or bad key) -- try again in "
                    "a moment."
                )
                with st.expander("Technical details"):
                    st.exception(e)

# ---------------------------------------------------------------------------
# Section 4: Your Gaps
# ---------------------------------------------------------------------------
if st.session_state.gaps:
    st.divider()
    st.header("Next Steps")
    status_labels = {
        "met": "✅ Met",
        "partial": "⚠️ Partial",
        "missing": "❌ Missing",
    }
    skills_by_id = {s["id"]: s.get("name", s["id"]) for s in st.session_state.skills}

    for gap in st.session_state.gaps:
        skill_name = skills_by_id.get(gap["skill_id"], gap["skill_id"])
        label = status_labels.get(gap["status"], gap["status"])
        with st.container(border=True):
            st.markdown(f"**{skill_name}** — {label}")
            if gap.get("evidence_from_cv"):
                st.caption(f"From your CV: {gap['evidence_from_cv']}")
            if gap.get("suggestion"):
                st.write(gap["suggestion"])


# ---------------------------------------------------------------------------
# Section 5: Market Insights
# ---------------------------------------------------------------------------
# Independent of the CV/Gaps flow above -- only needs a roadmap to exist, not
# a completed gap analysis. Deliberately not auto-run: one check costs 5
# Tavily searches plus up to 14 Groq calls, and job postings go stale fast,
# so it's an explicit on-demand check rather than baked into
# get_or_create_roadmap's cache flow.
if st.session_state.skills:
    st.divider()
    st.header("Market Insights")
    st.caption("Check your roadmap against real, live job postings")

    if st.button("Check against real job postings"):
        try:
            with st.status("Scanning real job postings...", expanded=True) as status:
                def update_progress(current, total):
                    status.update(label=f"Reading posting {current} of {total}...")

                insights = get_market_validation(
                    get_refined_title(st.session_state.role),
                    st.session_state.skills,
                    progress_callback=update_progress,
                )
                status.update(label="Done", state="complete")
            st.session_state.market_insights = insights
        except Exception as e:
            st.error(
            "Couldn't complete the market check. This is usually a Groq "
            "or Tavily API issue (rate limit, network, or bad key) -- try "
            "again in a moment."
        )
            with st.expander("Technical details"):
                st.exception(e)

    if st.session_state.market_insights:
        insights = st.session_state.market_insights
        # Kept for the empty-state check below, deliberately never shown.
        # A posting total reads as evidence ("3 of 12"), but some of those
        # pages arrive from Tavily with their requirement bullets stripped,
        # so the total counts pages we failed to read. See mention_band().
        scanned = insights.get("total_postings_scanned", 0)

        if scanned == 0:
            st.info("No job postings found for this role -- try again later.")
        else:
            if insights.get("confirmed"):
                st.subheader("✅ Confirmed by real postings")
                for item in insights["confirmed"]:
                    matched = ", ".join(item.get("matched_market_skills", []))
                    st.write(
                        f"**{item['roadmap_skill_name']}** — matched {matched}, "
                        f"{mention_band(item.get('mention_count', 0))}"
                    )

            if insights.get("suggested_additions"):
                st.subheader("➕ Suggested additions")
                st.caption("Skills real postings ask for that aren't in your roadmap yet")

                additions = insights["suggested_additions"]
                strong = [a for a in additions if a.get("mention_count", 0) >= 2]
                weak = [a for a in additions if a.get("mention_count", 0) < 2]

                for item in sorted(strong, key=lambda a: -a.get("mention_count", 0)):
                    st.write(
                        f"**{item['skill_name']}** — "
                        f"{mention_band(item['mention_count'])}"
                    )

                if weak:
                    with st.expander(f"Mentioned in only one posting ({len(weak)})"):
                        for item in weak:
                            st.write(item["skill_name"])

            st.caption(
                "A directional signal from live job postings, not a "
                "comprehensive market survey."
            )



# ---------------------------------------------------------------------------
# Section 6: Mock Interview
# ---------------------------------------------------------------------------
# Asks questions the way a real interviewer does -- about your CV first, then
# the job you're applying for, then the role's skills -- and grades each
# answer against a checklist of key points. One question at a time: the next
# one only appears once the current one has been graded.
MARK_ICONS = {"covered": "✅", "partly": "⚠️", "missed": "❌"}

if st.session_state.skills:
    st.divider()
    st.header("Mock Interview")

    if not st.session_state.cv_text:
        st.caption("Tip: upload your CV and click Analyze Gaps above to get questions about your CV too.")

    posting_text = st.text_area(
        "Job description (optional)",
        placeholder="Paste the job posting you're applying for",
    )
    skills_by_name = {s["name"]: s for s in st.session_state.skills}
    chosen_names = st.multiselect(
        "Skills to be asked about",
        list(skills_by_name),
        default=list(skills_by_name)[:2],
    )
    st.caption("Each skill costs one web search and one Groq call.")

    if st.button("Start interview", type="primary"):
        try:
            with st.spinner("Preparing your questions..."):
                questions = build_interview(
                    st.session_state.cv_text,
                    get_refined_title(st.session_state.role),
                    posting_text,
                    [skills_by_name[name] for name in chosen_names],
                )
        except Exception as e:
            st.error(
                "Couldn't prepare the interview. This is usually a Groq or Tavily "
                "API issue (rate limit, network, or bad key) -- try again in a moment."
            )
            with st.expander("Technical details"):
                st.exception(e)
        else:
            st.session_state.interview_questions = questions
            st.session_state.interview_grades = {}

    if st.session_state.interview_questions:
        questions = st.session_state.interview_questions
        grades = st.session_state.interview_grades
        skill_names = {s["id"]: s["name"] for s in st.session_state.skills}

        for question in questions:
            with st.container(border=True):
                # Where the question came from -- every question can show it.
                if question["type"] == "cv":
                    st.caption(f'About your CV: "{question["based_on"]}"')
                elif question["type"] == "job_posting":
                    st.caption(f'About the job posting: "{question["based_on"]}"')
                elif question["source_url"]:
                    st.caption(f"{skill_names.get(question['skill_id'], 'Skill')} -- "
                               f"a real interview question from {question['source_url']}")
                else:
                    st.caption(f"{skill_names.get(question['skill_id'], 'Skill')} -- "
                               "written by Groq, no matching question found online")
                st.markdown(f"**{question['question']}**")

                grade = grades.get(question["id"])
                if grade is None:
                    with st.form(key=f"form_{question['id']}"):
                        answer = st.text_area("Your answer", key=f"answer_{question['id']}")
                        submitted = st.form_submit_button("Submit answer")
                    if submitted:
                        try:
                            with st.spinner("Grading your answer..."):
                                new_grade = grade_answer(question, answer)
                        except Exception as e:
                            st.error(
                                "Couldn't grade that answer. This is usually a Groq API "
                                "issue -- try submitting again in a moment."
                            )
                            with st.expander("Technical details"):
                                st.exception(e)
                        else:
                            grades[question["id"]] = new_grade
                            st.rerun()  # show the grade and the next question
                    break  # later questions stay hidden until this one is graded

                st.markdown(f"> {grade['answer'] or '(no answer)'}")
                st.markdown(f"**Score: {grade['score']:g} / {grade['max_score']}**")
                for mark in grade["marks"]:
                    line = f"{MARK_ICONS[mark['mark']]} {mark['key_point']}"
                    if mark["evidence"]:
                        line += f' -- you said: "{mark["evidence"]}"'
                    st.write(line)
                st.info(grade["feedback"])

        if len(grades) == len(questions):
            total = sum(g["score"] for g in grades.values())
            out_of = sum(g["max_score"] for g in grades.values())
            st.success(f"Interview finished -- total score {total:g} / {out_of}")
