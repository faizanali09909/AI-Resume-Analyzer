import os
import sys

# Fix for Streamlit Cloud SQLite version requirement by ChromaDB
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import tempfile
import time
import re

# Load environment variables
load_dotenv()

# Sync API Keys
api_key = os.getenv("GEMINI_API_KEY")
if api_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = api_key

# Page configuration
st.set_page_config(
    page_title="AI Resume Analyzer Pro",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4);
    }
    
    .result-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    </style>
""", unsafe_allow_html=True)

from typing import Any

from crewai.tools import tool

def setup_resume_rag(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    chunks = splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

def analyze_resume(pdf_path, job_description):
    vectorstore = setup_resume_rag(pdf_path)
    
    @tool("Search Resume")
    def resume_tool(query: str):
        """Search the candidate's resume for specific skills, experience, or education details."""
        results = vectorstore.similarity_search(query, k=2)
        return "\n".join([r.page_content for r in results])

    # Use Gemini Flash for reliability and higher limits
    llm = LLM(model="gemini/gemini-flash-latest", api_key=os.getenv("GEMINI_API_KEY"))

    extractor = Agent(
        role="Resume Extractor",
        goal="Extract all relevant skills, experience, and education from the resume using search tools.",
        backstory="Expert at deep-reading resumes and extracting key information with precision.",
        llm=llm,
        tools=[resume_tool]
    )
    
    matcher = Agent(
        role="Job Matcher",
        goal="Compare resume details with the job description and pinpoint specific gaps or matches.",
        backstory="Technical recruiter with years of experience matching candidates to specialized roles.",
        llm=llm,
        tools=[resume_tool]
    )
    
    coach = Agent(
        role="Career Coach",
        goal="Provide a strategic match score and 3-5 high-impact improvement tips.",
        backstory="Career strategist who knows exactly how to make a resume stand out to hiring managers.",
        llm=llm
    )

    task1 = Task(
        description="Search the candidate's resume and extract a structured list of skills, professional experience, and educational background.",
        agent=extractor,
        expected_output="A detailed summary of the candidate's profile based on the resume."
    )
    
    task2 = Task(
        description=f"Compare the candidate's profile with this job description:\n{job_description}. Use the search tool if you need to clarify specific experience points.",
        agent=matcher,
        expected_output="An analysis of matches, missing skills, and potential deal-breakers."
    )
    
    task3 = Task(
        description="Based on the analysis, provide a match score from 0 to 100 and 3-5 specific, actionable tips to improve the resume for this specific role.",
        agent=coach,
        expected_output="A final report containing the score and a list of improvement suggestions."
    )

    crew = Crew(
        agents=[extractor, matcher, coach],
        tasks=[task1, task2, task3],
        verbose=True
    )

    return run_crew_with_retry(crew)

def run_crew_with_retry(crew, inputs=None, max_retries=5):
    """Run crew with retry logic for handling rate limit and 503 errors"""
    for attempt in range(max_retries):
        try:
            return crew.kickoff(inputs=inputs) if inputs else crew.kickoff()
        except Exception as e:
            error_str = str(e)
            # Handle 503 - Service Unavailable
            if "503" in error_str or "UNAVAILABLE" in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    st.warning(f"🤖 Model busy, retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise e
            # Handle 429 - Rate Limit Exceeded
            elif "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                if attempt < max_retries - 1:
                    # Extract retry delay from error if available, otherwise use default
                    wait_time = 20  # Default 20 seconds for rate limits
                    if "retryDelay" in error_str:
                        try:
                            # Try to parse the retry delay from error
                            match = re.search(r'(\d+)s', error_str)
                            if match:
                                wait_time = int(match.group(1)) + 2  # Add buffer
                        except:
                            pass
                    st.warning(f"⏱️ Rate limit hit! Waiting {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    raise e
            else:
                raise e

# App UI
st.markdown("<h1 class='main-header'>AI Resume Analyzer Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Optimize your resume for any job description using Multi-Agent AI</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 🎯 Job Details")
    job_desc = st.text_area(
        "Paste the Job Description here:",
        placeholder="e.g., We are looking for a Python Developer with 2+ years of experience...",
        height=300
    )
    
    st.markdown("### 📄 Upload Resume")
    uploaded_file = st.file_uploader("Choose your resume (PDF)", type="pdf")

if col2:
    st.markdown("### 🔍 Analysis Result")
    if st.button("🚀 Run Deep Analysis"):
        if not uploaded_file:
            st.error("Please upload a resume first.")
        elif not job_desc:
            st.error("Please provide a job description.")
        else:
            with st.spinner("🤖 Our AI Agents are analyzing your resume..."):
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    result = analyze_resume(tmp_path, job_desc)
                    st.success("✨ Analysis Complete!")
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown(result)
                    st.markdown("</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
    else:
        st.info("Results will appear here after you run the analysis.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: #64748b;'>Built with CrewAI & Streamlit 🚀</p>", unsafe_allow_html=True)
