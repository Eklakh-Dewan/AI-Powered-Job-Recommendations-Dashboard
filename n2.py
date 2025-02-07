import streamlit as st
import pandas as pd
import requests
import PyPDF2
import re
import random
from sentence_transformers import SentenceTransformer, util

st.set_page_config(page_title="AI-Powered Job Recommendations Dashboard", layout="wide")

# Load NLP Model for Semantic Matching
model = SentenceTransformer('all-MiniLM-L6-v2')

# Helper function to extract keywords from resume text
def extract_keywords(text):
    words = re.findall(r'\b\w{3,}\b', text.lower())
    keywords = set([word for word in words if word not in {"and", "the", "with", "for", "from", "that", "this", "have"}])
    return keywords

st.title("AI-Powered Job Recommendations System Dashboard")
st.write("This dashboard uses AI to provide personalized job recommendations based on your resume.")

st.header("Upload Resume")
resume_file = st.file_uploader("Upload your resume (PDF format)", type=["pdf"])

if resume_file:
    st.success("Resume uploaded successfully! Extracting details...")
    try:
        # Extract text from the uploaded PDF
        pdf_reader = PyPDF2.PdfReader(resume_file)
        resume_text = "".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        if not resume_text:
            st.warning("No text could be extracted from the uploaded PDF.")
        else:
            st.text_area("Extracted Resume Text", resume_text, height=200)

            # Extract keywords from the resume
            resume_keywords = extract_keywords(resume_text)

            # Semantic Matching with Job Data
            api_url = "https://remotive.io/api/remote-jobs"
            response = requests.get(api_url)

            if response.status_code == 200:
                jobs_data = response.json().get("jobs", [])
                st.header("Top Job Recommendations")

                # Batch encode job texts for faster similarity computation
                job_texts = [f"{job['title']} {job['description']}" for job in jobs_data]
                job_embeddings = model.encode(job_texts, convert_to_tensor=True)
                resume_embedding = model.encode(resume_text, convert_to_tensor=True)

                # Compute similarities in a single operation
                similarities = util.cos_sim(resume_embedding, job_embeddings).squeeze().tolist()

                # Attach similarity scores to jobs and sort
                job_scores = list(zip(jobs_data, similarities))
                ranked_jobs = sorted(job_scores, key=lambda x: x[1], reverse=True)[:5]

                for job, score in ranked_jobs:
                    st.write(f"- **{job['title']}** at {job['company_name']}, Score: {score:.2f}")

                    # Fetch additional companies for the job's category
                    related_jobs = [j for j in jobs_data if j['category'] == job['category']][:2]

                    st.write("Other companies hiring for similar roles:")
                    for related_job in related_jobs:
                        st.markdown(f"**Company:** {related_job['company_name']} - **Job Title:** {related_job['title']} - [Apply Here]({related_job['url']})")
                    st.markdown("---")

                # Country Selection for job filtering
                st.header("Filter Jobs by Country")
                country = st.selectbox("Select a country", ["All"] + list(set([job['candidate_required_location'] for job in jobs_data])))

                if country != "All":
                    st.subheader(f"Top Companies Providing Recommended Jobs in {country}")
                    country_jobs = [job for job in jobs_data if job['candidate_required_location'] == country]

                    # Extract top  companies and their job titles
                    top_companies_jobs = country_jobs[:10]

                    for job in top_companies_jobs:
                        st.markdown(f"**Company Name:** {job['company_name']}")
                        st.markdown(f"**Location:** {job['candidate_required_location']}")
                        st.markdown(f"**Job Title:** {job['title']}")
                        st.markdown(f"[Apply Here]({job['url']})")

                        # dynamic trend data for visualization
                        trend_years = [2019, 2020, 2021, 2022, 2023]
                        trend_counts = [random.randint(20, 50), random.randint(50, 70), random.randint(70, 90), random.randint(90, 120), random.randint(120, 150)]
                        trend_data = pd.DataFrame({"Year": trend_years, "Job Count": trend_counts})
                        st.subheader(f"Trend for {job['title']} at {job['company_name']}")
                        st.line_chart(trend_data.set_index("Year"))
                        st.markdown("---")
            else:
                st.error("Failed to fetch job data. Please try again later.")
    except Exception as e:
        st.error(f"An error occurred while processing the resume: {str(e)}")

# Footer Section
st.markdown("---")
st.write("Thank you for using this AI-powered job recommendations dashboard!")

st.markdown("Developed by Mr. Eklakh with ❤️")
