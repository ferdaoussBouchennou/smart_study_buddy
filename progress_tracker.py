# progress_tracker.py
# Sauvegarde et récupère la progression de l'étudiant

import sqlite3
import os
from datetime import datetime

class ProgressTracker:
    """
    Gère la base de données SQLite pour
    sauvegarder la progression de l'étudiant
    """
    
    def __init__(self, db_path="study_data.db"):
        self.db_path = db_path
        self._create_tables()
    
    # ════════════════════════════════════════
    # CRÉER LES TABLES
    # ════════════════════════════════════════
    def _create_tables(self):
        """
        Crée les tables si elles n'existent pas encore
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table principale: sessions de quiz
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                date        TEXT    NOT NULL,
                subject     TEXT    NOT NULL,
                difficulty  TEXT    NOT NULL,
                language    TEXT    DEFAULT 'English',
                total_q     INTEGER NOT NULL,
                correct_q   INTEGER NOT NULL,
                score_pct   REAL    NOT NULL
            )
        """)
        
        # Table détail: chaque réponse individuelle
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quiz_answers (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id      INTEGER NOT NULL,
                question_num    INTEGER NOT NULL,
                question_text   TEXT    NOT NULL,
                user_answer     TEXT,
                correct_answer  TEXT    NOT NULL,
                is_correct      BOOLEAN NOT NULL,
                FOREIGN KEY (session_id) 
                    REFERENCES quiz_sessions(id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    # ════════════════════════════════════════
    # SAUVEGARDER UN QUIZ
    # ════════════════════════════════════════
    def save_quiz(self, subject, difficulty, language,
                  score_data):
        """
        Sauvegarde les résultats d'un quiz complet
        
        Args:
            subject: nom du cours/matière
            difficulty: easy/medium/hard
            language: langue utilisée
            score_data: dict retourné par calculate_score()
        
        Returns: session_id
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Sauvegarder la session principale
        cursor.execute("""
            INSERT INTO quiz_sessions
            (date, subject, difficulty, language,
             total_q, correct_q, score_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            subject or "General",
            difficulty,
            language,
            score_data["total"],
            score_data["correct"],
            score_data["percentage"]
        ))
        
        session_id = cursor.lastrowid
        
        # Sauvegarder chaque réponse
        for result in score_data["results"]:
            cursor.execute("""
                INSERT INTO quiz_answers
                (session_id, question_num, question_text,
                 user_answer, correct_answer, is_correct)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id,
                result["question_num"],
                result["question"],
                result["user_answer"],
                result["correct_answer"],
                result["is_correct"]
            ))
        
        conn.commit()
        conn.close()
        
        return session_id
    
    # ════════════════════════════════════════
    # RÉCUPÉRER LES STATISTIQUES
    # ════════════════════════════════════════
    def get_stats(self):
        """
        Retourne les statistiques globales de progression
        
        Returns: dict avec toutes les stats
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Nombre total de quiz
        cursor.execute("SELECT COUNT(*) FROM quiz_sessions")
        total_quizzes = cursor.fetchone()[0]
        
        # Score moyen
        cursor.execute(
            "SELECT AVG(score_pct) FROM quiz_sessions"
        )
        avg_score = cursor.fetchone()[0] or 0
        
        # Meilleur score
        cursor.execute(
            "SELECT MAX(score_pct) FROM quiz_sessions"
        )
        best_score = cursor.fetchone()[0] or 0
        
        # Score le plus récent
        cursor.execute("""
            SELECT score_pct FROM quiz_sessions
            ORDER BY id DESC LIMIT 1
        """)
        row = cursor.fetchone()
        last_score = row[0] if row else 0
        
        # Historique récent (5 derniers)
        cursor.execute("""
            SELECT date, subject, difficulty, 
                   score_pct, total_q, correct_q
            FROM quiz_sessions
            ORDER BY id DESC
            LIMIT 5
        """)
        recent = cursor.fetchall()
        
        # Progression par difficulté
        cursor.execute("""
            SELECT difficulty, AVG(score_pct), COUNT(*)
            FROM quiz_sessions
            GROUP BY difficulty
        """)
        by_difficulty = cursor.fetchall()
        
        conn.close()
        
        return {
            "total_quizzes": total_quizzes,
            "avg_score": round(avg_score, 1),
            "best_score": round(best_score, 1),
            "last_score": round(last_score, 1),
            "recent_sessions": recent,
            "by_difficulty": by_difficulty
        }
    
    # ════════════════════════════════════════
    # RÉINITIALISER LA PROGRESSION
    # ════════════════════════════════════════
    def reset_progress(self):
        """
        Efface toute la progression (remet à zéro)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM quiz_answers")
        cursor.execute("DELETE FROM quiz_sessions")
        conn.commit()
        conn.close()