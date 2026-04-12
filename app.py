# app.py
# Interface principale Streamlit

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from rag_engine import RAGEngine
from quiz_generator import QuizGenerator
from progress_tracker import ProgressTracker

load_dotenv()

# ════════════════════════════════════════
# CONFIGURATION DE LA PAGE
# ════════════════════════════════════════
st.set_page_config(
    page_title="Smart Study Buddy",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour un beau design
st.markdown("""
<style>
    .score-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5em;
        font-weight: bold;
    }
    .correct {background-color: #d4edda; color: #155724;}
    .wrong   {background-color: #f8d7da; color: #721c24;}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════
# INITIALISATION SESSION STATE
# ════════════════════════════════════════
@st.cache_resource
def init_engines():
    """
    Initialise les moteurs une seule fois
    (cache_resource évite de réinitialiser à chaque interaction)
    """
    return (
        RAGEngine(),
        QuizGenerator(),
        ProgressTracker()
    )

rag, quiz_gen, tracker = init_engines()

# Variables de session
defaults = {
    "pdf_loaded": False,
    "pdf_name": "",
    "chat_history": [],
    "current_quiz": [],
    "user_answers": {},
    "quiz_submitted": False,
    "quiz_score": None,
    "num_chunks": 0
}

for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    
    st.markdown("## Settings")
    
    # Langue
    language = st.selectbox(
        "Language",
        ["English", "Français", "العربية"],
        index=0
    )
    
    # Nom du cours
    subject = st.text_input(
        " Subject Name",
        placeholder="e.g., Machine Learning",
        value=""
    )
    
    st.divider()
    
    # ── Upload PDF ──
    st.markdown("## Upload Your Course")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload your course PDF (max 200MB)"
    )
    
    if uploaded_file is not None:
        # Vérifier si c'est un nouveau fichier
        if uploaded_file.name != st.session_state.pdf_name:
            
            with st.spinner("Processing PDF..."):
                
                # Sauvegarder temporairement
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name
                
                # Lancer le pipeline RAG
                try:
                    num_chunks = rag.load_pdf(tmp_path)
                    st.session_state.pdf_loaded = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.num_chunks = num_chunks
                    st.session_state.chat_history = []
                    
                    # Nettoyer fichier temp
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"Error loading PDF: {e}")
        
        if st.session_state.pdf_loaded:
            st.success(
                f"**{st.session_state.pdf_name}**\n\n"
            )
    
    st.divider()
    
    # ── Statistiques ──
    st.markdown("## My Progress")
    
    stats = tracker.get_stats()
    
    if stats["total_quizzes"] > 0:
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Quizzes",
                stats["total_quizzes"]
            )
        with col2:
            st.metric(
                "Average",
                f"{stats['avg_score']}%"
            )
        
        st.metric(
            "Best Score",
            f"{stats['best_score']}%"
        )
        
        # Barre de progression
        st.progress(stats["avg_score"] / 100)
        
    else:
        st.info("Take your first quiz to see stats!")
    
    st.divider()
    
    # Bouton reset
    if st.button("Reset Progress", type="secondary"):
        tracker.reset_progress()
        st.success("Progress reset!")
        st.rerun()

# ════════════════════════════════════════
# CONTENU PRINCIPAL
# ════════════════════════════════════════
st.markdown(
    "<h1 style='color: #B33630; font-weight: 600;text-align: center;'>Smart Study Buddy</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center'>Your AI-powered personal tutor</p>",
    unsafe_allow_html=True
)

if not st.session_state.pdf_loaded:
    # Message de bienvenue si pas de PDF
    st.markdown(
        '<div style="background-color: #f0f2f6; padding: 16px; border-radius: 8px; color: #555; text-align: left; margin-bottom: 20px;">'
        'Upload a PDF course in the sidebar to get started.'
        '</div>',
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #B33630; font-weight: 600;'>Capabilities</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: -10px;'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.container(border=True):
            st.subheader("Ask Questions")
            st.write("Interact directly with your course material. Ask complex questions and get precise answers.")
        
        with st.container(border=True):
            st.subheader("Get Summary")
            st.write("Instantly generate a complete overview including main topics, key concepts, and study tips.")
            
    with col2:
        with st.container(border=True):
            st.subheader("Take Quizzes")
            st.write("Test your knowledge through auto-generated MCQs tailored specifically to your PDF.")
            
        with st.container(border=True):
            st.subheader("Track Progress")
            st.write("Keep an eye on your scores, track history by difficulty, and measure your improvement.")
            
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #B33630; font-weight: 600;'>Quick Start Guide</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top: -10px;'>", unsafe_allow_html=True)
    
    step1, step2, step3 = st.columns(3)
    with step1:
        with st.container(border=True):
            st.markdown("<h4 style='color: #555;'>Step 1</h4>", unsafe_allow_html=True)
            st.write("**Upload** your PDF course via the sidebar")
    with step2:
        with st.container(border=True):
            st.markdown("<h4 style='color: #555;'>Step 2</h4>", unsafe_allow_html=True)
            st.write("**Choose** your preferred language")
    with step3:
        with st.container(border=True):
            st.markdown("<h4 style='color: #555;'>Step 3</h4>", unsafe_allow_html=True)
            st.write("**Start** learning and testing!")

else:
    # ── TABS PRINCIPAUX ──
    tab1, tab2, tab3, tab4 = st.tabs([
        "Ask Questions",
        "Quiz",
        "Summary",
        "History"
    ])
    
    # ══════════════════════════════════════
    # TAB 1: POSER DES QUESTIONS
    # ══════════════════════════════════════
    with tab1:
        st.header("Ask Questions About Your Course")
        
        # Afficher l'historique du chat
        chat_container = st.container()
        
        with chat_container:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    with st.chat_message("user"):
                        st.write(msg["content"])
                else:
                    with st.chat_message("assistant"):
                        st.write(msg["content"])
                        # Afficher les sources si disponibles
                        if "sources" in msg and msg["sources"]:
                            with st.expander("Sources"):
                                for i, src in enumerate(
                                    msg["sources"]
                                ):
                                    st.caption(
                                        f"Excerpt {i+1}: "
                                        f"{src.page_content[:200]}..."
                                    )
        
        # Input question
        question = st.chat_input(
            "Ask anything about your course..."
        )
        
        if question:
            with chat_container:
                # Afficher la question
                with st.chat_message("user"):
                    st.write(question)
                
                # Générer la réponse
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        answer, sources = rag.ask_question(
                            question, language
                        )
                    
                    st.write(answer)
                    
                    if sources:
                        with st.expander("Sources from your course"):
                            for i, src in enumerate(sources):
                                st.caption(
                                    f"**Excerpt {i+1}** "
                                    f"(Page {src.metadata.get('page', 'N/A')}):"
                                )
                                st.write(src.page_content[:300] + "...")
                                st.divider()
            
            # Sauvegarder dans l'historique
            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )
        
        # Bouton clear chat
        if st.session_state.chat_history:
            if st.button("Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()
    
    # ══════════════════════════════════════
    # TAB 2: QUIZ
    # ══════════════════════════════════════
    with tab2:
        st.header("Test Your Knowledge")
        
        # ── Configuration du Quiz ──
        if not st.session_state.quiz_submitted:
            
            st.subheader("Quiz Settings")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_q = st.selectbox(
                    "Number of Questions",
                    options=[3, 5, 10],
                    index=1
                )
            
            with col2:
                difficulty = st.selectbox(
                    "Difficulty",
                    options=["easy", "medium", "hard"],
                    index=1
                )
            
            with col3:
                topic = st.text_input(
                    "Specific Topic (optional)",
                    placeholder="e.g., backpropagation"
                )
            
            st.divider()
            
            # Bouton générer quiz
            if st.button(
                "Generate Quiz",
                type="primary",
                use_container_width=True
            ):
                with st.spinner(
                    "Generating your quiz... Please wait..."
                ):
                    # Récupérer le contenu
                    context = rag.get_content_for_quiz(
                        topic if topic else None
                    )
                    
                    if not context:
                        st.error("Could not retrieve course content!")
                    else:
                        # Générer les questions
                        questions = quiz_gen.generate_quiz(
                            context=context,
                            num_questions=num_q,
                            difficulty=difficulty,
                            language=language
                        )
                        
                        if questions:
                            st.session_state.current_quiz = questions
                            st.session_state.user_answers = {}
                            st.session_state.quiz_submitted = False
                            st.session_state.quiz_score = None
                            st.success(
                                f"{len(questions)} questions generated!"
                            )
                            st.rerun()
                        else:
                            st.error(
                                "Could not generate quiz. "
                                "Try again!"
                            )
        
        # ── Afficher les Questions ──
        if (st.session_state.current_quiz and
                not st.session_state.quiz_submitted):
            
            st.subheader(
                f"Answer all {len(st.session_state.current_quiz)}"
                f" questions:"
            )
            
            # Afficher chaque question
            for i, q in enumerate(st.session_state.current_quiz):
                
                st.markdown(
                    f"**Question {i+1} of "
                    f"{len(st.session_state.current_quiz)}:** "
                    f"{q['question']}"
                )
                
                # Les 4 options
                options_display = [
                    f"A — {q['options']['A']}",
                    f"B — {q['options']['B']}",
                    f"C — {q['options']['C']}",
                    f"D — {q['options']['D']}"
                ]
                
                selected = st.radio(
                    "Choose your answer:",
                    options=options_display,
                    key=f"question_{i}",
                    index=None,
                    label_visibility="collapsed"
                )
                
                # Sauvegarder la réponse (extraire A, B, C, ou D)
                if selected:
                    st.session_state.user_answers[i] = selected[0]
                
                st.divider()
            
            # Compter les réponses données
            answered = len(st.session_state.user_answers)
            total_q = len(st.session_state.current_quiz)
            
            st.write(
                f"✏️ Answered: {answered}/{total_q} questions"
            )
            
            # Bouton soumettre (actif seulement si tout répondu)
            submit_disabled = answered < total_q
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(
                    "Submit Answers",
                    type="primary",
                    disabled=submit_disabled,
                    use_container_width=True
                ):
                    # Calculer le score
                    score_data = quiz_gen.calculate_score(
                        st.session_state.current_quiz,
                        st.session_state.user_answers
                    )
                    
                    st.session_state.quiz_score = score_data
                    st.session_state.quiz_submitted = True
                    
                    # Sauvegarder dans la DB
                    tracker.save_quiz(
                        subject=subject or "General",
                        difficulty=difficulty,
                        language=language,
                        score_data=score_data
                    )
                    
                    st.rerun()
            
            with col2:
                if st.button(
                    " New Quiz",
                    use_container_width=True
                ):
                    st.session_state.current_quiz = []
                    st.session_state.user_answers = {}
                    st.session_state.quiz_submitted = False
                    st.rerun()
        
        # ── Afficher les Résultats ──
        if (st.session_state.quiz_submitted and
                st.session_state.quiz_score):
            
            score = st.session_state.quiz_score
            pct = score["percentage"]
            
            st.subheader("Your Results")
            
            # Score global avec couleur
            if pct >= 80:
                st.balloons()
                st.success(
                    f" Excellent! {score['correct']}/{score['total']}"
                    f" correct = **{pct}%**"
                )
            elif pct >= 60:
                st.warning(
                    f" Good job! {score['correct']}/{score['total']}"
                    f" correct = **{pct}%**"
                )
            else:
                st.error(
                    f" Keep studying! {score['correct']}/"
                    f"{score['total']} correct = **{pct}%**"
                )
            
            # Barre de progression du score
            st.progress(pct / 100)
            
            st.divider()
            st.subheader("Detailed Review:")
            
            # Détail de chaque question
            for result in score["results"]:
                
                if result["is_correct"]:
                    st.success(
                        f" **Q{result['question_num']}:** "
                        f"{result['question']}\n\n"
                        f"Your answer: **{result['user_answer']}** "
                        f"- {result['user_answer_text']}"
                    )
                else:
                    st.error(
                        f" **Q{result['question_num']}:** "
                        f"{result['question']}\n\n"
                        f"Your answer: **{result['user_answer']}** "
                        f"- {result['user_answer_text']}\n\n"
                        f" Correct: **{result['correct_answer']}** "
                        f"- {result['correct_answer_text']}"
                    )
                
                with st.expander(
                    f" Explanation for Q{result['question_num']}"
                ):
                    st.write(result["explanation"])
            
            st.divider()
            
            # Boutons après résultats
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(
                    " New Quiz",
                    type="primary",
                    use_container_width=True
                ):
                    st.session_state.current_quiz = []
                    st.session_state.user_answers = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.quiz_score = None
                    st.rerun()
            
            with col2:
                if st.button(
                    " Review Course",
                    use_container_width=True
                ):
                    st.info(
                        "Go to 'Ask Questions' tab to review "
                        "topics you missed!"
                    )
    
    # ══════════════════════════════════════
    # TAB 3: RÉSUMÉ
    # ══════════════════════════════════════
    with tab3:
        st.header(" Course Summary")
        
        st.write(
            "Generate a complete summary of your uploaded course."
        )
        
        if st.button(
            " Generate Summary",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Creating your summary..."):
                summary = rag.get_summary(language)
            
            st.markdown("---")
            st.markdown(summary)
            st.markdown("---")
            
            # Bouton télécharger
            st.download_button(
                label="⬇ Download Summary as .txt",
                data=summary,
                file_name=f"summary_{subject or 'course'}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # ══════════════════════════════════════
    # TAB 4: HISTORIQUE
    # ══════════════════════════════════════
    with tab4:
        st.header("Quiz History & Progress")
        
        stats = tracker.get_stats()
        
        if stats["total_quizzes"] == 0:
            st.info("No quiz history yet. Take your first quiz!")
        
        else:
            # ── Stats Globales ──
            st.subheader("Overall Statistics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Quizzes",
                    stats["total_quizzes"]
                )
            with col2:
                st.metric(
                    "Average Score",
                    f"{stats['avg_score']}%"
                )
            with col3:
                st.metric(
                    "Best Score",
                    f"{stats['best_score']}%"
                )
            with col4:
                st.metric(
                    "Last Score",
                    f"{stats['last_score']}%"
                )
            
            # Barre de progression globale
            st.write("**Overall Progress:**")
            st.progress(stats["avg_score"] / 100)
            
            st.divider()
            
            # ── Performance par Difficulté ──
            if stats["by_difficulty"]:
                st.subheader("📈 Performance by Difficulty")
                
                for diff, avg, count in stats["by_difficulty"]:
                    emoji = (
                        "🟢" if avg >= 80
                        else "🟡" if avg >= 60
                        else "🔴"
                    )
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.write(
                            f"{emoji} **{diff.capitalize()}** "
                            f"({count} quizzes)"
                        )
                    with col2:
                        st.progress(avg / 100)
                        st.caption(f"{round(avg, 1)}%")
            
            st.divider()
            
            # ── Historique Récent ──
            st.subheader(" Recent Quizzes")
            
            for session in stats["recent_sessions"]:
                date, subj, diff, score, total, correct = session
                
                with st.expander(
                    f"{date} | {subj} | "
                    f"{diff} | {score:.0f}%"
                ):
                    st.write(f" Date: {date}")
                    st.write(f" Subject: {subj}")
                    st.write(f" Difficulty: {diff}")
                    st.write(f" Score: {correct}/{total} ({score:.0f}%)")
                    st.progress(score / 100)