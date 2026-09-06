"""
Streamlit UI for the AI Career Coach (Personal Tracker).

Flow: Choose a Role -> Your Required Skills -> Upload Your CV -> Your Gaps

Uses st.session_state because Streamlit reruns this entire script top-to-bottom
on every interaction (typing, clicking, uploading). Without state, the skills
list generated in step 1 would vanish by the time the user reaches step 3.
"""

import streamlit as st
from personaltracker import get_skill_roadmap, extract_text_from_pdf, get_skill_gaps

st.set_page_config(page_title="NextRole", layout="wide")

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
    "cv_text": None,
    "gaps": None,
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
            with st.spinner("Researching current requirements for this role..."):
                role = role_input.strip()
                skills = get_skill_roadmap(role)
            # Only commit to session_state once the call has actually succeeded --
            # if get_skill_roadmap raises partway through, we don't want a role
            # set with no matching skills.
            st.session_state.role = role
            st.session_state.skills = skills
            # A new role invalidates any CV analysis run against the old skills.
            st.session_state.cv_text = None
            st.session_state.gaps = None
        except Exception as e:
            st.error(
                "Couldn't generate the skill roadmap. This is usually a Groq or "
                "Tavily API issue (rate limit, network, or bad key) rather than "
                "a problem with your input -- try again in a moment."
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
    for skill in st.session_state.skills:
        with st.container(border=True):
            st.markdown(f"**{skill.get('name', skill.get('id'))}**")
            if skill.get("description"):
                st.caption(skill["description"])

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