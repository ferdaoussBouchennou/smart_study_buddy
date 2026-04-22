# rag_engine.py
# Gère le chargement PDF et le pipeline RAG complet

import os
import re
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
            max_tokens=2048  # augmenté pour des réponses complètes
        )
        
        # Vector database - sera créé après upload PDF
        self.vector_db = None
        self.is_loaded = False
        
        print(" RAG Engine ready!")
    
    # ════════════════════════════════════════
    # NETTOYAGE DU TEXTE PDF
    # ════════════════════════════════════════
    def _clean_text(self, text: str) -> str:
        """
        Corrige les artefacts courants de l'extraction PDF :
        - mots collés sans espaces (ex: 'SpringBoot' au lieu de 'Spring Boot')
        - sauts de ligne multiples
        - espaces superflus
        - tirets de coupure de mot (-\n)
        """
        # Supprimer les tirets de coupure en fin de ligne
        text = re.sub(r'-\n', '', text)
        
        # Remplacer les sauts de ligne simples par un espace
        # (garder les double sauts = séparateurs de paragraphes)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
        
        # Insérer un espace avant une majuscule collée à une minuscule
        # ex: "ApplicationContext" → déjà bon, mais "SpringBoot" → "Spring Boot"
        # Cas: mot_minuscule + MAJUSCULE + minuscules collés ensemble
        text = re.sub(r'([a-z\u00e0-\u00ff])([A-Z][a-z])', r'\1 \2', text)
        
        # Normaliser les espaces multiples
        text = re.sub(r'[ \t]{2,}', ' ', text)
        
        # Supprimer les lignes qui sont uniquement des headers/footers
        # (lignes très courtes ou contenant seulement des chiffres / codes)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Garder la ligne sauf si c'est un header/footer typique:
            # - moins de 4 caractères
            # - uniquement un numéro de page
            # - pattern "Année/Université + numero de page"
            if len(stripped) < 4:
                continue
            if re.match(r'^[\d\s/\-]+$', stripped):
                continue
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        return text.strip()
    
    # ════════════════════════════════════════
    # ÉTAPE 1: CHARGER ET TRAITER LE PDF
    # ════════════════════════════════════════
    def load_pdf(self, pdf_path):
        """
        Pipeline d'ingestion complet:
        PDF → Pages → Nettoyage → Chunks → Vecteurs → Base de données
        
        Returns: nombre de chunks créés
        """
        
        print(f" Loading PDF: {pdf_path}")
        
        # ── 1.1: Lire le PDF ──
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"   {len(pages)} pages loaded")
        
        # ── 1.2: Nettoyer le texte de chaque page ──
        for page in pages:
            page.page_content = self._clean_text(page.page_content)
        
        # Filtrer les pages vides après nettoyage
        pages = [p for p in pages if len(p.page_content.strip()) > 50]
        print(f"   {len(pages)} pages après nettoyage")
        
        # ── 1.3: Découper en chunks ──
        # chunk_size plus grand = plus de contexte par chunk
        # chunk_overlap plus grand = meilleure continuité
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(pages)
        
        # Filtrer les chunks trop courts (bruit)
        chunks = [c for c in chunks if len(c.page_content.strip()) > 80]
        print(f"    {len(chunks)} chunks créés")
        
        # ── 1.4: Créer les embeddings et stocker ──
        print("   Creating embeddings (this may take a minute)...")
        
        # Create vector database in memory (ephemeral)
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
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
    # GÉNÉRER UN RÉSUMÉ COMPLET DU COURS
    # ════════════════════════════════════════
    def get_summary(self, language="English"):
        """
        Génère un résumé complet et détaillé du cours,
        structuré comme un vrai cours de professeur.
        
        Returns: summary string
        """
        
        if not self.is_loaded:
            return "Please upload a PDF first!"
        
        retriever = self.vector_db.as_retriever(
            search_kwargs={"k": 15}
        )
        
        # Plusieurs requêtes ciblées pour couvrir tout le cours
        queries = [
            "main topics introduction overview objectives",
            "key concepts definitions terminology",
            "important formulas methods algorithms steps procedure",
            "examples illustrations applications use cases",
            "conclusion results advantages disadvantages comparison"
        ]
        
        seen = set()
        all_docs = []
        for q in queries:
            for doc in retriever.invoke(q):
                # Dédupliquer par contenu
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    all_docs.append(doc)
        
        # Trier par numéro de page pour respecter l'ordre du cours
        all_docs.sort(
            key=lambda d: d.metadata.get("page", 0)
        )
        
        context = "\n\n".join([doc.page_content for doc in all_docs])
        
        # Limite généreuse pour un résumé détaillé
        if len(context) > 8000:
            context = context[:8000] + "..."
        
        prompt = f"""You are an experienced university professor.
Your task is to write a COMPLETE, DETAILED and WELL-STRUCTURED course summary in {language}.
The summary must be long, thorough and educational — like official lecture notes a professor
would hand out to students before an exam.

Course content to summarize:
{context}

---
Write the summary using EXACTLY this structure (use markdown formatting):

# Course Summary

## 1. Introduction & Overview
Brief introduction to the course topic, its importance and objectives.

## 2. Main Topics Covered
For EACH main topic found in the content:
### Topic Name
- Detailed explanation (3–5 sentences minimum)
- Sub-points if needed

## 3. Key Definitions & Concepts
List ALL important terms with their precise definitions:
- **Term**: definition

## 4. Important Formulas / Methods / Steps
List any formulas, algorithms, procedures or methods with explanations.
If none exist, describe the key processes step by step.

## 5. Concrete Examples & Applications
Provide real examples or use cases from the course content.

## 6. Key Points to Remember
Bullet list of the most critical points a student MUST know for an exam.

## 7. Study Tips
Specific advice for studying this material effectively.

---
RULES:
- Be thorough and detailed — this is a serious academic summary
- Use clear, simple language appropriate for students
- Preserve all technical terms from the course
- Do NOT skip sections — fill each one based on the content
- Answer in {language}
"""
        
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