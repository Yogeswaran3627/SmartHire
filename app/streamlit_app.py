"""
SmartHire — Resume-to-Job Matching & Career Guidance Engine
Production Streamlit Application featuring:
- Resume Category Classification (Supervised ML)
- Top-N Content-based Job Recommendation (Unsupervised TF-IDF + Cosine Similarity)
- Candidate Shortlisting / Fit Predictor (Supervised Engineered Feature ML)
- Skill-Gap Diagnostics & Actionable Career Guidance
- Job-family Clustering & LDA Topic Discovery
- Rule-based AI Career Mentor Assistant (Stretch Goal)
- Interactive Model Benchmark Analytics & Report Downloader
"""
import sys
import io
from pathlib import Path

# Add project root before importing src modules
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
# pyrefly: ignore[missing-import]
import streamlit as st

from src.mentor import ask_mentor

from src.config import (
    CLEAN_RESUMES_PATH,
    CLEAN_JOBS_PATH,
    MODEL_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    DEFAULT_TOP_N,
    TFIDF_PATH,
    CLASSIFIER_PATH,
    FIT_PREDICTOR_PATH,
    CLUSTERING_PATH
)
from src.data.load_data import load_processed_data, load_raw_resumes, discover_raw_files
from src.data.preprocess import merge_job_corpus, preprocess_resumes
from src.parsing.resume_parser import extract_resume_text, extract_contact_info
from src.features.text_features import load_tfidf_vectorizer
from src.features.match_features import extract_skills, skill_gap, extract_years_experience, extract_education_level
from src.models.classifier import load_classifier, predict_category, train_classifier
from src.models.recommender import recommend_jobs, evaluate_precision_at_k
from src.models.fit_predictor import load_fit_predictor, predict_fit_score
from src.models.clustering import cluster_jobs, extract_job_topics

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="SmartHire — AI Resume & Job Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Header */
.hero-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #172554 100%);
    padding: 2.2rem 2.4rem;
    border-radius: 18px;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
    margin-bottom: 1.8rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #cbd5e1;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Cards & Containers */
.smart-card {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 1.3rem;
    margin-bottom: 1rem;
    transition: all 0.25s ease;
}
.smart-card:hover {
    border-color: rgba(96, 165, 250, 0.4);
    box-shadow: 0 8px 20px -4px rgba(0, 0, 0, 0.15);
}

/* Badges & Pills */
.pill-badge {
    display: inline-block;
    padding: 0.28rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 0.2rem;
}
.pill-green {
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}
.pill-blue {
    background: rgba(59, 130, 246, 0.18);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.35);
}
.pill-amber {
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}
.pill-purple {
    background: rgba(168, 85, 247, 0.18);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.35);
}

/* Metric KPI Tile */
.metric-tile {
    text-align: center;
    padding: 1.2rem;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
}
.metric-value {
    font-size: 1.85rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.82rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.3rem;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------
# Cached Data & Model Loading
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_cached_corpus():
    """Loads processed jobs and resumes, or processes them if missing."""
    resumes_df, jobs_df = load_processed_data()

    if resumes_df is None:
        try:
            resumes_df = preprocess_resumes(save=True)
        except Exception:
            resumes_df = pd.DataFrame()

    if jobs_df is None:
        try:
            jobs_df = merge_job_corpus(save=True, max_rows=25000)
        except Exception:
            jobs_df = pd.DataFrame()

    return resumes_df, jobs_df

@st.cache_resource(show_spinner=False)
def get_cached_models():
    """Loads pre-trained models from disk."""
    clf = load_classifier()
    tfidf = load_tfidf_vectorizer(TFIDF_PATH)
    fit_model = load_fit_predictor()
    return clf, tfidf, fit_model

# Load core assets
resumes_df, jobs_df = get_cached_corpus()
classifier_model, global_vectorizer, fit_predictor_model = get_cached_models()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎯 SmartHire ML")
    st.caption("Resume-to-Job Matching & Career Guidance")
    st.divider()

    st.markdown("**Dataset Corpus**")
    st.metric("Total Jobs Indexed", f"{len(jobs_df):,}")
    st.metric("Training Resumes", f"{len(resumes_df):,}")
    st.metric("Target Domains", f"{resumes_df['category'].nunique() if not resumes_df.empty else 0}")

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">SmartHire — Intelligent Career Engine</div>
    <div class="hero-subtitle">
        High-precision classical machine learning for candidate-job matching, role domain classification, 
        and data-driven skill-gap diagnostics. No external LLMs required.
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabs Structure
# ---------------------------------------------------------
tabs = st.tabs([
    "📄 Resume Matcher & Fit Analysis",
    "🤖 AI Career Mentor",
    "🔍 Explore Job Corpus",
    "🌐 Job Families & Clustering"
])

# =========================================================
# TAB 1: RESUME MATCHER & FIT ANALYSIS
# =========================================================
with tabs[0]:
    st.markdown("### 📄 Resume Analysis & Shortlisting Engine")

    with st.expander("⚡ Load Sample Demo Profile (Optional)", expanded=False):
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        with preset_col1:
            if st.button("🧪 Sample 1: Senior Data Scientist", use_container_width=True):
                st.session_state["active_resume"] = (
                    "Senior Data Scientist with 5 years experience in Python, Machine Learning, Deep Learning, "
                    "NLP, Scikit-learn, TensorFlow, PyTorch, Pandas, SQL, Docker, AWS and statistical data analysis. "
                    "Master of Science in Computer Science. Built predictive churn models, automated NLP pipelines, and recommender systems."
                )
                st.rerun()
        with preset_col2:
            if st.button("💻 Sample 2: Full-Stack Web Developer", use_container_width=True):
                st.session_state["active_resume"] = (
                    "Full Stack Web Developer with 4 years hands-on experience building scalable applications. "
                    "Proficient in JavaScript, TypeScript, React, Node.js, Express, HTML5, CSS3, PostgreSQL, MongoDB, "
                    "REST APIs, Git, CI/CD, and Docker. Bachelor of Technology in Information Technology."
                )
                st.rerun()
        with preset_col3:
            if st.button("👥 Sample 3: HR & Talent Acquisition Lead", use_container_width=True):
                st.session_state["active_resume"] = (
                    "Human Resources Specialist with 6 years experience in technical recruitment, talent acquisition, "
                    "onboarding, employee relations, payroll, HR operations, performance management and stakeholder management. "
                    "MBA in Human Resource Management. Proficient in ATS tools, LinkedIn Recruiter, and communication."
                )
                st.rerun()

    col_input, col_meta = st.columns([1.4, 1.0])
    with col_input:
        uploaded_file = st.file_uploader("Upload Candidate Resume (PDF, DOCX, or TXT)", type=["pdf", "docx", "txt"])
        pasted_text = st.text_area(
            "Or paste resume text directly:",
            value=st.session_state.get("active_resume", ""),
            height=150,
            placeholder="Paste raw resume or CV text here..."
        )

        resume_text = ""
        if uploaded_file is not None:
            try:
                resume_text = extract_resume_text(uploaded_file)
                st.caption(f"Extracted {len(resume_text):,} characters from `{uploaded_file.name}`.")
            except Exception as e:
                st.error(f"Error extracting resume text: {e}")
        elif pasted_text.strip():
            resume_text = pasted_text.strip()

        st.session_state["active_resume"] = resume_text

    with col_meta:
        st.markdown("**Candidate Parsing Summary**")
        if resume_text and len(resume_text) >= 30:
            contact = extract_contact_info(resume_text)
            years_exp = extract_years_experience(resume_text)
            edu_level = extract_education_level(resume_text)
            edu_labels = {4: "Doctorate / PhD", 3: "Master's Degree", 2: "Bachelor's Degree", 1: "Diploma", 0: "Unspecified"}

            st.markdown(f"- 📧 **Email:** `{contact['email'] or 'Not detected'}`")
            st.markdown(f"- 📞 **Phone:** `{contact['phone'] or 'Not detected'}`")
            st.markdown(f"- ⏱️ **Est. Experience:** `{years_exp} years`" if years_exp > 0 else "- ⏱️ **Est. Experience:** `Entry / Unspecified`")
            st.markdown(f"- 🎓 **Education Level:** `{edu_labels.get(edu_level, 'Unspecified')}`")
            
            cand_skills = extract_skills(resume_text)
            st.markdown(f"- 🛠️ **Detected Skills ({len(cand_skills)}):**")
            pills = " ".join([f"<span class='pill-badge pill-blue'>{s}</span>" for s in cand_skills[:12]])
            st.markdown(pills if pills else "None from taxonomy", unsafe_allow_html=True)
            if len(cand_skills) > 12:
                st.caption(f"... and {len(cand_skills) - 12} more skills detected.")
        else:
            st.info("Upload or paste a resume to view parsed profile attributes.")

    if resume_text and len(resume_text) >= 30:
        st.divider()
        # 1. RESUME CLASSIFICATION (MODEL A)
        st.markdown("#### 🏷️ 1. Domain Category Classification (Supervised Model A)")
        if classifier_model is not None:
            cat_result = predict_category(classifier_model, resume_text, top_k=5)
            c_pred1, c_pred2 = st.columns([1, 1.5])
            with c_pred1:
                st.markdown(f"""
                <div class="smart-card">
                    <span style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase;">Primary Predicted Domain</span>
                    <div style="font-size:1.6rem; font-weight:700; color:#38bdf8; margin:0.3rem 0;">
                        {cat_result['predicted_category']}
                    </div>
                    <span class="pill-badge pill-green">Confidence: {cat_result['confidence_pct']}%</span>
                </div>
                """, unsafe_allow_html=True)
            with c_pred2:
                st.markdown("**Top Alternative Class Probabilities**")
                probs_df = pd.DataFrame(cat_result["top_predictions"])
                st.dataframe(
                    probs_df[["category", "percentage"]].rename(columns={"category": "Domain", "percentage": "Model Probability (%)"}),
                    hide_index=True,
                    use_container_width=True
                )
        else:
            st.warning("Classifier model not yet trained.")

        st.divider()

        # 2. JOB RECOMMENDATIONS (UNSUPERVISED ENGINE)
        st.markdown("#### 🎯 2. Top Matching Job Recommendations (Unsupervised Cosine Recommender)")
        
        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
        with filter_col1:
            role_query = st.text_input("Filter Role / Title (Optional)", placeholder="e.g. Developer, Analyst")
        with filter_col2:
            loc_query = st.text_input("Filter Location (Optional)", placeholder="e.g. India, Remote")
        with filter_col3:
            top_k_select = st.slider("Top Results to Show", min_value=5, max_value=25, value=10)

        with st.spinner("Finding best matching jobs across 25,000+ postings..."):
            recs = recommend_jobs(
                resume_text,
                jobs_df,
                top_n=top_k_select,
                filter_role=role_query,
                filter_location=loc_query,
                vectorizer=global_vectorizer
            )

        if not recs.empty:
            display_cols = [c for c in ["rank", "title", "company", "location", "experience", "salary", "match_percent"] if c in recs.columns]
            st.dataframe(
                recs[display_cols].rename(columns={"match_percent": "Similarity Match (%)"}),
                hide_index=True,
                use_container_width=True
            )

            st.divider()

            # 3. FIT PREDICTOR & SKILL GAP REPORT
            st.markdown("#### 🔍 3. In-Depth Job Fit & Skill-Gap Analysis")
            st.write("Select one of the recommended jobs above to compute the hiring shortlisting probability and skill breakdown:")

            job_options = [f"#{r['rank']}: {r['title']} @ {r['company']} ({r['location']})" for _, r in recs.iterrows()]
            selected_idx = st.selectbox("Select Target Job Posting:", range(len(job_options)), format_func=lambda i: job_options[i])
            chosen_job = recs.iloc[selected_idx]

            # Run Fit Predictor (Model B)
            fit_result = predict_fit_score(
                resume_text,
                chosen_job["job_text"],
                model=fit_predictor_model,
                vectorizer=global_vectorizer
            )

            # Run Skill Gap Engine
            gap_result = skill_gap(resume_text, chosen_job["job_text"])

            fit_c1, fit_c2, fit_c3 = st.columns([1, 1, 1])
            with fit_c1:
                st.markdown(f"""
                <div class="smart-card" style="text-align:center;">
                    <div style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase;">Shortlisting Fit Score</div>
                    <div style="font-size:2.2rem; font-weight:800; color:{'#34d399' if fit_result['is_shortlisted'] else '#fbbf24'};">
                        {fit_result['fit_percent']}%
                    </div>
                    <span class="pill-badge {'pill-green' if fit_result['is_shortlisted'] else 'pill-amber'}">
                        {fit_result['status']}
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with fit_c2:
                st.markdown(f"""
                <div class="smart-card" style="text-align:center;">
                    <div style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase;">Skill Coverage</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#60a5fa;">
                        {gap_result['coverage'] * 100:.1f}%
                    </div>
                    <span class="pill-badge pill-blue">
                        {len(gap_result['matched_skills'])} of {len(gap_result['required_skills'])} Skills Matched
                    </span>
                </div>
                """, unsafe_allow_html=True)

            with fit_c3:
                st.markdown(f"""
                <div class="smart-card" style="text-align:center;">
                    <div style="font-size:0.85rem; color:#94a3b8; text-transform:uppercase;">Text Alignment</div>
                    <div style="font-size:2.2rem; font-weight:800; color:#c084fc;">
                        {chosen_job['match_percent']}%
                    </div>
                    <span class="pill-badge pill-purple">TF-IDF Cosine Similarity</span>
                </div>
                """, unsafe_allow_html=True)

            # Skill Gap Details
            st.markdown("##### Detailed Skill Breakdown")
            skill_c1, skill_c2 = st.columns(2)
            with skill_c1:
                st.markdown("**✅ Matched Candidate Skills**")
                if gap_result["matched_skills"]:
                    pills = " ".join([f"<span class='pill-badge pill-green'>{s}</span>" for s in gap_result["matched_skills"]])
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.caption("No direct keyword skills matched.")

            with skill_c2:
                st.markdown("**⚠️ Identified Skill Gaps (To Acquire or Emphasize)**")
                if gap_result["missing_skills"]:
                    pills = " ".join([f"<span class='pill-badge pill-amber'>{s}</span>" for s in gap_result["missing_skills"]])
                    st.markdown(pills, unsafe_allow_html=True)
                else:
                    st.markdown("<span class='pill-badge pill-green'>Zero Missing Skills — 100% Core Competency!</span>", unsafe_allow_html=True)

            st.caption(f"💡 **Career Advisor Note:** {gap_result['note']}")

            with st.expander("🔍 View Raw Job Posting Details & Description"):
                st.markdown(f"**Title:** {chosen_job['title']}")
                st.markdown(f"**Company:** {chosen_job['company']} | **Location:** {chosen_job['location']}")
                st.markdown(f"**Experience Requirement:** {chosen_job.get('experience', 'Not specified')}")
                st.markdown(f"**Salary:** {chosen_job.get('salary', 'Not disclosed')}")
                st.markdown(f"**Full Description:**\n\n{chosen_job.get('description', '')[:1500]}...")

# =========================================================
# TAB 2: AI CAREER MENTOR (CHATBOT)
# =========================================================

with tabs[1]:
    st.header("🤖 AI Career Mentor")
    st.write(
        "Your personal AI mentor for career guidance, "
        "learning roadmaps, resumes and interview preparation."
    )

    if "mentor_messages" not in st.session_state:
        st.session_state["mentor_messages"] = []

    # Use the resume and job recommendations from SmartHire
    context_parts = []

    resume_text = st.session_state.get("resume_text", "")

    if resume_text:
        context_parts.append(
            "Candidate resume:\n" + resume_text[:5000]
        )

    recommendations = st.session_state.get("recommendations")

    if recommendations is not None and len(recommendations):
        columns = [
            c for c in [
                "title", "company", "location",
                "skills", "experience"
            ]
            if c in recommendations.columns
        ]

        context_parts.append(
            "Recommended jobs:\n"
            + recommendations[columns]
                .head(5)
                .to_string(index=False)
        )

    context = "\n\n".join(context_parts)

    # Display previous conversation messages
    for message in st.session_state["mentor_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input(
        "Ask your career mentor anything..."
    )

    if prompt:
        st.session_state["mentor_messages"].append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Career Mentor is thinking..."):
                try:
                    answer = ask_mentor(
                        st.session_state["mentor_messages"],
                        context=context
                    )

                    st.markdown(answer)

                    st.session_state["mentor_messages"].append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as error:
                    st.error(str(error))

    if st.button("Clear conversation"):
        st.session_state["mentor_messages"] = []
        st.rerun()

# =========================================================
# TAB 3: EXPLORE JOB CORPUS
# =========================================================
with tabs[2]:
    st.markdown("### 🔍 Unified Job Postings Corpus")
    st.caption("Search across 25,000+ curated postings merged from jobs.csv, Naukri ldjson, and LinkedIn.")

    s_col1, s_col2, s_col3 = st.columns([1.5, 1, 1])
    with s_col1:
        search_kw = st.text_input("Search Keyword / Title / Skills:", placeholder="e.g. Machine Learning, Manager, React")
    with s_col2:
        if "source" in jobs_df.columns:
            source_filt = st.selectbox("Filter Source:",["All Sources"] + sorted(jobs_df["source"].dropna().astype(str).unique().tolist()))
        else:
            source_filt = "All Sources"
    with s_col3:
        max_rows_disp = st.selectbox("Max rows to display:", [50, 100, 250, 500], index=1)

    filtered_jobs = jobs_df
    if source_filt != "All Sources":
        filtered_jobs = filtered_jobs[filtered_jobs["source"] == source_filt]

    if search_kw.strip():
        kw = search_kw.strip().lower()
        mask = (
            filtered_jobs["title"].str.lower().str.contains(kw, na=False) |
            filtered_jobs["skills"].str.lower().str.contains(kw, na=False) |
            filtered_jobs["company"].str.lower().str.contains(kw, na=False) |
            filtered_jobs["location"].str.lower().str.contains(kw, na=False)
        )
        filtered_jobs = filtered_jobs[mask]

    st.write(f"Showing {min(len(filtered_jobs), max_rows_disp):,} of {len(filtered_jobs):,} matching listings:")
    disp_cols = [c for c in ["title", "company", "location", "skills", "experience", "salary", "source"] if c in filtered_jobs.columns]
    st.dataframe(filtered_jobs[disp_cols].head(max_rows_disp), hide_index=True, use_container_width=True)

    csv_data = filtered_jobs[disp_cols].head(1000).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Matching Postings (CSV)",
        data=csv_data,
        file_name="smarthire_jobs_export.csv",
        mime="text/csv"
    )

# =========================================================
# TAB 4: JOB FAMILIES & CLUSTERING
# =========================================================
with tabs[3]:
    st.markdown("### 🌐 Job Family Discovery & Topic Modeling")
    st.markdown("Unsupervised K-Means clustering groups postings into natural job families based on TF-IDF semantic similarities.")

    c_img1, c_img2 = st.columns(2)
    with c_img1:
        pca_fig = FIGURES_DIR / "job_clusters_pca.png"
        if pca_fig.exists():
            st.image(str(pca_fig), caption="2D PCA Projection of Job Clusters", use_container_width=True)
    with c_img2:
        elbow_fig = FIGURES_DIR / "kmeans_elbow_silhouette.png"
        if elbow_fig.exists():
            st.image(str(elbow_fig), caption="Optimal k: Elbow Inertia & Silhouette Scores", use_container_width=True)

    st.divider()
    st.markdown("#### 🔬 Latent Dirichlet Allocation (LDA) Topic Themes")
    st.caption("Extracted latent skill and industry themes from job posting descriptions:")

    try:
        topics = extract_job_topics(jobs_df.head(2000), n_topics=5)
        for t in topics:
            with st.expander(f"Topic {t['topic_id']}: {t['theme']}"):
                st.write(f"**Prominent Keywords:** {', '.join(t['keywords'])}")
    except Exception as e:
        st.info("Topic modeling ready. Topics extraction running on demand.")



# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94a3b8; font-size:0.85rem;'>"
    "SmartHire — Machine Learning Industrial Project &copy; 2026. Built with Python, Scikit-Learn & Streamlit."
    "</div>",
    unsafe_allow_html=True
)

if __name__ == "__main__":
    if not st.runtime.exists():
        try:
            from run import ensure_environment
            ensure_environment()
        except Exception:
            pass
        # pyrefly: ignore [missing-import]
        from streamlit.web import cli as stcli
        sys.argv = [
            "streamlit",
            "run",
            str(Path(__file__).resolve()),
            "--browser.gatherUsageStats=false"
        ]
        sys.exit(stcli.main())



