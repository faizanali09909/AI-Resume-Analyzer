from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from crewai import Agent, Task, Crew, LLM
import os
from dotenv import load_dotenv

load_dotenv()



def setup_resume_rag(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

def get_resume_context(vectorstore, job_description):
    results = vectorstore.similarity_search(job_description, k=4)
    return "\n".join([r.page_content for r in results])

llm = LLM(model="groq/llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

extractor = Agent(
    role="Resume Extractor",
    goal="Extract all skills, experience, education, achievement",
    backstory="You are an expert at reading resumes and pulling key info",
    llm=llm, verbose=True
)
matcher = Agent(
    role="Job Matcher",
    goal="Compare resume details with job requirements and find gaps",
    backstory="You are a recruiter who knows what employers want.",
    llm=llm, verbose=True
)
coach = Agent(
    role="Career Coach",
    goal="Give a match score out of 100 and suggest improvements",
    backstory="You help candidates improve their resumes.",
    llm=llm, verbose=True
)


def analyze_resume(pdf_path, job_description):
    vectorstore = setup_resume_rag(pdf_path)
    resume_context = get_resume_context(vectorstore, job_description)
    task1 = Task(
        description=f"""Extract skills, experience, education
from:\n{resume_context}""",
        agent=extractor,
        expected_output="Bulleted list of skills, experience, education."
    )
    task2 = Task(
        description=f"""Compare the extracted info with this
job:\n{job_description}""",
        agent=matcher,
        expected_output="Matches, missing skills, and gaps."
    )
    task3 = Task(
        description="Give a score out of 100 and 3-5 improvement tips.",
        agent=coach,
        expected_output="Match score and improvement suggestions."
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
