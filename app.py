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

import pathlib

import streamlit as st
import streamlit.components.v1 as components

from personaltracker import (
    LEVEL_DOWN_RATIO,
    START_LEVEL,
    build_interview,
    generate_follow_up,
    next_level,
    extract_text_from_pdf,
    grade_answer,
    get_skill_gaps,
    get_or_create_roadmap,
    get_market_validation,
    get_refined_title,
    get_tracker_items,
    rebuild_roadmap,
    update_tracker_status,
)

st.set_page_config(page_title="NextRole", layout="wide")

# The voice interviewer is its own little web page, loaded here once.
# declare_component, not components.html, for two reasons: only a declared
# component can send a value BACK to Python, and Streamlit serves it from
# its own address -- a components.html page has no real origin, and a
# browser will not hand the microphone to a page it cannot identify.
VOICE_DIR = pathlib.Path(__file__).parent / "voice_component"
voice_component = components.declare_component("nextrole_voice", path=str(VOICE_DIR))

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
    # How hard the next question should be (1-5), and why a
    # follow-up could not be written, if that happened.
    "interview_level": None,
    "follow_up_error": None,
    "uploader_key": 0,
    # Same trick as uploader_key, for the progress dropdowns. A rebuild
    # reuses skill ids (skl_001, skl_002...) for different skills, and a
    # Streamlit widget keyed "status_skl_001" keeps whatever was selected
    # before -- so a brand new skill would show the old skill's progress.
    # Bumping this changes every key, forcing fresh dropdowns.
    "skills_key": 0,
    # True only while the rebuild confirmation is on screen.
    "confirming_rebuild": False,
    "voice_mode": False,
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

def closing_line(questions, grades) -> str:
    """What the interviewer says after the LAST answer.

    Without this the interviewer hears the final answer, says nothing at
    all, and the interview ends on a bare number. Every other answer gets
    a reply, because the reply is carried in front of the next question
    -- and the last answer has no next question to carry it.
    """
    last = grades.get(questions[-1]["id"]) or {}
    return f"{last.get('reaction', '')} That's everything from me. Thanks for your time.".strip()


def interviewer_line(questions, grades, index) -> str:
    """What the interviewer says at question `index`.

    A reaction to the previous answer, then the question itself -- so it
    sounds like one conversation instead of a list of questions read out
    loud. Worked out here, never stored: the reaction is already part of
    the grade saved for the previous answer.
    """
    line = questions[index]["question"]
    if index > 0:
        previous = grades.get(questions[index - 1]["id"])
        if previous and previous.get("reaction"):
            line = f"{previous['reaction']} {line}"
    return line


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
                widget_key = f"status_{st.session_state.skills_key}_{skill['id']}"
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
    # Rebuild this roadmap
    # -----------------------------------------------------------------------
    # Behind a confirm step on purpose. Every other button either reads the
    # database or acts on something the user just supplied; this one spends
    # Groq and Tavily credits to replace data they already have, so a
    # misclick has a real cost and is not undoable.
    if st.session_state.confirming_rebuild:
        st.warning(
            "This throws away the saved roadmap for "
            f"**{st.session_state.role}** and generates a new one "
            "(2 Groq calls + 2 Tavily searches). Progress is kept for every "
            "skill that comes back with the same name. Notes on a skill the "
            "new roadmap drops are lost."
        )
        col_yes, col_no, _ = st.columns([1, 1, 3])
        if col_no.button("Cancel", use_container_width=True):
            st.session_state.confirming_rebuild = False
            st.rerun()
        if col_yes.button("Yes, rebuild", type="primary", use_container_width=True):
            try:
                with st.spinner("Building a new roadmap for this role..."):
                    skills = rebuild_roadmap(st.session_state.role)
                    tracker = get_tracker_items(st.session_state.role)
            except Exception as e:
                st.error(
                    "Couldn't rebuild the roadmap. Your existing one is "
                    "untouched -- nothing was deleted. This is usually a Groq "
                    "or Tavily issue (rate limit, network, or bad key)."
                )
                with st.expander("Technical details"):
                    st.exception(e)
            else:
                st.session_state.skills = skills
                st.session_state.tracker = tracker
                # Everything below was worked out from the OLD skill list, so
                # it now describes skills that may no longer exist. The CV
                # stays -- that is the user's file, not a result.
                st.session_state.gaps = None
                st.session_state.market_insights = None
                st.session_state.interview_questions = None
                st.session_state.interview_grades = None
                st.session_state.skills_key += 1
                st.session_state.confirming_rebuild = False
                st.rerun()
    elif st.button("Rebuild this roadmap"):
        st.session_state.confirming_rebuild = True
        st.rerun()


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

    st.session_state.voice_mode = st.toggle(
        "Voice mode",
        value=st.session_state.voice_mode,
        help="The interviewer reads each question out loud, and you answer by speaking.",
    )
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
            st.session_state.interview_level = START_LEVEL
            st.session_state.follow_up_error = None

    if st.session_state.interview_questions:
        questions = st.session_state.interview_questions
        grades = st.session_state.interview_grades
        skill_names = {s["id"]: s["name"] for s in st.session_state.skills}

        if st.session_state.voice_mode:
            # -------------------------------------------------------------
            # The conversational path
            # -------------------------------------------------------------
            # Everything that has to feel instant -- speaking, hearing you,
            # the words arriving on screen as you talk, noticing that you
            # have stopped -- happens in the browser, inside
            # voice_component. Python is reached once per turn, when you
            # finish answering. It could never have held a live microphone
            # itself: Streamlit reruns this entire script top to bottom on
            # every interaction.
            #
            # The transcript below stores nothing new. It is worked out
            # from the grades we already have, so every line on screen can
            # be pointed at a real question or a real graded answer.
            answered = len(grades)

            # Three containers, made in a fixed order before anything is
            # written into them.
            #
            # This is not tidiness. Streamlit rebuilds a component's iframe
            # when the component moves in the page, and a rebuilt iframe
            # never receives its question: Streamlit has already marked
            # this component ready, so the new iframe saying hello is
            # ignored. The interview then stops dead with no error at all.
            # Growing the conversation above the component moved it every
            # single turn. Made this way, the component is always the third
            # thing here no matter how long the conversation gets, and the
            # history is written into a box that was already in place.
            history_box = st.container()
            question_box = st.container()
            live_box = st.container()
            marks_box = st.container()

            with history_box:
                for i in range(answered):
                    with st.chat_message("assistant"):
                        st.write(interviewer_line(questions, grades, i))
                    with st.chat_message("user"):
                        st.write(grades[questions[i]["id"]]["answer"] or "(said nothing)")

            if answered < len(questions):
                with question_box:
                    with st.chat_message("assistant"):
                        st.write(interviewer_line(questions, grades, answered))

                with live_box:
                    reply = voice_component(
                        say=interviewer_line(questions, grades, answered),
                        turn=answered,
                        listen=True,
                        default=None,
                        key="voice_turn",
                    )

                    if st.session_state.follow_up_error:
                        st.caption(
                            "Couldn't write a follow-up to your last answer, so "
                            "we moved on. "
                            f"({st.session_state.follow_up_error})"
                        )

                    # The component hands back its last answer again on
                    # every rerun. The turn number is what tells a fresh
                    # answer from one already graded -- without it,
                    # question 1 would be graded over and over in a loop.
                    said = None
                    if reply and reply.get("turn") == answered:
                        said = (reply.get("transcript") or "").strip()

                    # A way through that needs no microphone at all. A
                    # browser with no speech recognition, a refused
                    # permission, or a component that fails to connect
                    # would otherwise leave the interview stuck with
                    # nothing the candidate can do about it. Both routes
                    # feed the same grading call below.
                    with st.expander("Type this answer instead"):
                        with st.form(key=f"typed_{answered}"):
                            typed = st.text_area("Your answer", label_visibility="collapsed")
                            if st.form_submit_button("Submit") and typed.strip():
                                said = typed.strip()

                if said:
                    try:
                        with st.spinner("Thinking..."):
                            new_grade = grade_answer(questions[answered], said)
                    except Exception as e:
                        st.error(
                            "Couldn't grade that answer. This is usually a Groq API "
                            "issue -- try again in a moment."
                        )
                        with st.expander("Technical details"):
                            st.exception(e)
                    else:
                        asked = questions[answered]
                        grades[asked["id"]] = new_grade
                        st.session_state.follow_up_error = None

                        # What a real interviewer does with a thin answer:
                        # push on what you just said, rather than read out
                        # the model answer and change the subject. Only
                        # once per question -- a follow-up that goes badly
                        # does not earn another follow-up, or a struggling
                        # candidate would never get off the topic.
                        level = st.session_state.interview_level
                        ratio = new_grade["score"] / new_grade["max_score"]
                        if ratio <= LEVEL_DOWN_RATIO and asked["type"] != "follow_up":
                            try:
                                questions.insert(
                                    answered + 1,
                                    generate_follow_up(asked, said, level),
                                )
                            except Exception as follow_up_failed:
                                # Not worth losing a graded answer over.
                                # The interview moves on -- and says so,
                                # rather than quietly skipping a step.
                                st.session_state.follow_up_error = str(follow_up_failed)

                        st.session_state.interview_level = next_level(
                            level, new_grade["score"], new_grade["max_score"]
                        )
                        st.rerun()

            else:
                # Every answer gets a reply, including the last one.
                with question_box:
                    with st.chat_message("assistant"):
                        st.write(closing_line(questions, grades))
                with live_box:
                    voice_component(
                        say=closing_line(questions, grades),
                        turn=answered,
                        listen=False,      # speak, then stop -- nothing left to ask
                        default=None,
                        key="voice_turn",
                    )

            if answered:
                with marks_box:
                    with st.expander("Marks so far"):
                        for i in range(answered):
                            graded = grades[questions[i]["id"]]
                            st.markdown(
                                f"**Q{i + 1} -- {graded['score']:g} / {graded['max_score']}**"
                            )
                            for mark in graded["marks"]:
                                st.write(f"{MARK_ICONS[mark['mark']]} {mark['key_point']}")
                            st.caption(graded["feedback"])
        else:
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
                        qid = question["id"]
                        with st.form(key=f"form_{qid}"):
                            answer = st.text_area("Your answer", key=f"answer_{qid}")
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
