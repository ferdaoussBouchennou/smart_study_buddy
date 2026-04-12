# rag_engine.py
# Gère le chargement PDF et le pipeline RAG complet

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Charger les variables d'environnement
load_dotenv()

class RAGEngine:
    """
    Gère tout le pipeline RAG:
    PDF → Chunks → Embeddings → VectorDB → Réponses
    """
    
    def __init__(self):
        #  GRATUIT: HuggingFace Embeddings
        # Convertit le texte en vecteurs - pas besoin d'API!
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )
        
        #  GRATUIT: Groq LLM
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.4,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=1024
        )
        
        # Vector database - sera créé après upload PDF
        self.vector_db = None
        self.is_loaded = False
        
        print(" RAG Engine ready!")
    
    # ════════════════════════════════════════
    # ÉTAPE 1: CHARGER ET TRAITER LE PDF
    # ════════════════════════════════════════
    def load_pdf(self, pdf_path):
        """
        Pipeline d'ingestion complet:
        PDF → Pages → Chunks → Vecteurs → Base de données
        
        Returns: nombre de chunks créés
        """
        
        print(f" Loading PDF: {pdf_path}")
        
        #  1.1: Lire le PDF ──
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"   {len(pages)} pages loaded")
        
        # ── 1.2: Découper en chunks ──
        # chunk_size: max 500 caractères par chunk
        # chunk_overlap: 50 caractères partagés entre chunks
        # (l'overlap évite de perdre le contexte aux frontières)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(pages)
        print(f"    {len(chunks)} chunks created")
        
        # ── 1.3: Créer les embeddings et stocker ──
        # from_documents fait tout automatiquement:
        # texte → vecteur → stockage dans ChromaDB
        print("   Creating embeddings (this may take a minute)...")
        
        # Supprimer l'ancienne DB si elle existe
        import shutil
        if os.path.exists("./study_db"):
            shutil.rmtree("./study_db")
        
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory="./study_db"
        )
        
        self.is_loaded = True
        print(f"   Vector database created!")
        
        return len(chunks)
    
    # ════════════════════════════════════════
    # ÉTAPE 2: RÉPONDRE AUX QUESTIONS (RAG)
    # ════════════════════════════════════════
    def ask_question(self, question, language="English"):
        """
        Pipeline RAG complet pour une question:
        Question → Chercher chunks → LLM → Réponse
        
        Returns: (answer, source_documents)
        """
        
        if not self.is_loaded:
            return "Please upload a PDF first!", []
        
        # Template de prompt - instructions pour le LLM
        prompt_template = f"""You are a helpful and friendly study assistant.
Your job is to help students understand their course material.

IMPORTANT RULES:
- Answer ONLY based on the context provided below
- If the answer is not in the context, say "I couldn't find this in your course material"
- Be clear, simple and educational
- Answer in {language}
- Use bullet points when listing multiple items

Context from the course:
{{context}}

Student's question: {{question}}

Your helpful answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Créer la chaîne RAG
        # retriever: cherche les 3 chunks les plus similaires
        # chain_type "stuff": met tous les chunks dans 1 prompt
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={
                "prompt": prompt
            },
            return_source_documents=True
        )
        
        # Exécuter la chaîne
        result = qa_chain.invoke({"query": question})
        
        answer = result["result"]
        sources = result["source_documents"]
        
        return answer, sources
    
    # ════════════════════════════════════════
    # GÉNÉRER UN RÉSUMÉ DU COURS
    # ════════════════════════════════════════
    def get_summary(self, language="English"):
        """
        Génère un résumé complet du cours uploadé
        
        Returns: summary string
        """
        
        if not self.is_loaded:
            return "Please upload a PDF first!"
        
        # Récupérer un échantillon large du contenu
        retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 10}
        )
        docs = retriever.invoke("main topics key concepts definitions")
        
        # Combiner tous les chunks en un seul contexte
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Limiter le contexte (éviter de dépasser la limite de tokens)
        if len(context) > 3000:
            context = context[:3000] + "..."
        
        prompt = f"""You are an expert teacher.
Create a comprehensive study summary in {language}.

Course content:
{context}

Please provide:
1.  MAIN TOPICS (list the main subjects covered)
2.  KEY CONCEPTS (important terms and definitions)
3.  KEY TAKEAWAYS (most important things to remember)
4.  STUDY TIPS (what to focus on)

Make it clear and easy to study from.

Summary:"""
        
        response = self.llm.invoke(prompt)
        return response.content
    
    # ════════════════════════════════════════
    # RÉCUPÉRER DU CONTENU POUR LE QUIZ
    # ════════════════════════════════════════
    def get_content_for_quiz(self, topic=None):
        """
        Récupère le contenu pertinent pour générer un quiz
        
        Returns: context string
        """
        
        if not self.is_loaded:
            return ""
        
        # Chercher selon le topic ou en général
        if topic and topic.strip():
            query = f"key concepts facts definitions about {topic}"
        else:
            query = "important concepts definitions facts examples"
        
        retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 8}
        )
        docs = retriever.invoke(query)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Limiter pour éviter de dépasser les tokens
        if len(context) > 4000:
            context = context[:4000]
        
        return context