# Smart Study Buddy

**Smart Study Buddy** is an AI-powered personal tutor designed to help students master their course materials. Built with modern GenAI technologies, this web application allows users to upload PDF lectures, chat with their documents, generate structured summaries, and test their knowledge through auto-generated quizzes.

---

## Features

- **Interactive Chat (RAG)**: Ask complex questions about your course material and get highly accurate answers cited directly from the PDF excerpts.
- **Auto-Summary**: Instantly generate structured overviews, key concepts, and study tips from any uploaded course.
- **AI Quizzes**: Generate customized Multiple-Choice Questions (MCQs) tailored to the specific content of your PDF to test your knowledge.
- **Progress Tracking**: Keep an eye on your quiz scores, track history, and measure your improvement over time.
- **Multilingual Assistant**: Fully supports interactions, summaries, and quizzes in English, French, and Arabic—regardless of the PDF's original language.

## Technology Stack

- **Frontend & UI**: [Streamlit](https://streamlit.io/)
- **LLM & Orchestration**: [LangChain](https://python.langchain.com/) & [Groq](https://groq.com/) (using `llama-3.3-70b-versatile` for lightning-fast inference)
- **Document Processing**: `pypdf` + RecursiveCharacterTextSplitter
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (In-memory ephemeral configuration for robust multi-user scaling)

---

## How to Run Locally

Follow these steps to run Smart Study Buddy on your own machine.

### 1. Prerequisites
- Python 3.10+ installed
- Git installed
- A free [Groq API Key](https://console.groq.com/keys)

### 2. Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/smart_study_buddy.git
cd smart_study_buddy

# Create a virtual environment
python -m venv venv

# Activate it (Windows)
venv\Scripts\activate
# Or (Mac/Linux)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory (or rename the provided `.env.example`) and add your Groq API Key:

```env
GROQ_API_KEY="your_groq_api_key_here"
```

### 4. Start the Application
Run the Streamlit server:

```bash
streamlit run app.py
```
*The application will automatically launch in your default web browser at `http://localhost:8501`.*

---

## How to Deploy (Streamlit Community Cloud)

This app is optimized for seamless deployment on free hosting platforms like Streamlit Community Cloud.

1. Push your code to a public or private GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New app**.
3. Select your repository, branch (`main`), and main file path (`app.py`).
4. **Crucial Step**: Before clicking Deploy, click on **Advanced settings...** (or go to **Settings > Secrets** later).
5. Add your Groq API inside the Secrets block using TOML format:
   ```toml
   GROQ_API_KEY="your_groq_api_key_here"
   ```
6. Click **Deploy!**

> **Note**: The application has been patched to run the vector database purely in RAM (ephemeral mode) using `st.session_state`. This ensures multiple users can upload different PDFs and use the app simultaneously without database locks or cross-contamination.

---

## Author
Built by Ferdaouss Bouchennou to showcase skills in RAG (Retrieval-Augmented Generation), Prompt Engineering, and Full-Stack Python.
