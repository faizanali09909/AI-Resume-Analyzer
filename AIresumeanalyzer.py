from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from crewai import Agent, Task, Crew, LLM
import os
from dotenv import load_dotenv

load_dotenv()



from crewai_tools import BaseTool

class SearchResumeTool(BaseTool):
    name: str = "Search Resume"
    description: str = "Search the candidate's resume for specific skills, experience, or education details."
    vectorstore: object = None

    def _run(self, query: str) -> str:
        results = self.vectorstore.similarity_search(query, k=3)
        return "\n".join([r.page_content for r in results])

def setup_resume_rag(pdf_path, persist_directory="./chroma_db"):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", 
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Use persistent storage
    vectorstore = Chroma.from_documents(
        chunks, 
        embeddings, 
        persist_directory=persist_directory
    )
    return vectorstore

llm = LLM(model="groq/llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

def analyze_resume(pdf_path, job_description):
    vectorstore = setup_resume_rag(pdf_path)
    resume_tool = SearchResumeTool(vectorstore=vectorstore)

    extractor = Agent(
        role="Resume Extractor",
        goal="Extract all skills, experience, and education using the Search Resume tool.",
        backstory="Expert at deep-diving into resumes to find specific qualifications.",
        llm=llm, 
        tools=[resume_tool],
        verbose=True
    )
    
    matcher = Agent(
        role="Job Matcher",
        goal="Compare resume details with job requirements and identify gaps.",
        backstory="Recruiter with a sharp eye for matching talent to requirements.",
        llm=llm, 
        tools=[resume_tool],
        verbose=True
    )
    
    coach = Agent(
        role="Career Coach",
        goal="Provide a match score (0-100) and actionable improvement tips.",
        backstory="Experienced career mentor helping candidates optimize their profile.",
        llm=llm, 
        verbose=True
    )

    task1 = Task(
        description="Search the resume to extract all relevant skills, experience, and education details.",
        agent=extractor,
        expected_output="A structured list of skills, experience, and education found in the resume."
    )
    
    task2 = Task(
        description=f"Compare the extracted information with the following job requirements:\n{job_description}. Search the resume if you need more details to confirm a match.",
        agent=matcher,
        expected_output="A gap analysis highlighting direct matches and missing qualifications."
    )
    
    task3 = Task(
        description="Based on the comparison, give a match score out of 100 and 3-5 specific tips to improve the resume for this role.",
        agent=coach,
        expected_output="Final report with score and improvement suggestions."
    )

    crew = Crew(
        agents=[extractor, matcher, coach],
        tasks=[task1, task2, task3],
        verbose=True
    )
    return crew.kickoff()

if __name__ == "__main__":
    job_desc = """
    We are hiring a Python Developer with:
    - 2+ years Python experience
    - Knowledge of Django or Flask
    - Experience with REST APIs
    - Familiarity with SQL databases
    """
    result = analyze_resume("resumer.pdf", job_desc)
    print("\n===== RESUME ANALYSIS REPORT =====")
    print(result)
