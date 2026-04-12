# quiz_generator.py
# Génère des quiz automatiquement depuis le contenu du cours

import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

class QuizGenerator:
    """
    Génère des QCM (Multiple Choice Questions)
    à partir du contenu du cours
    """
    
    def __init__(self):
        # Température plus haute = questions plus variées
        self.llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.5,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=2048
        )
    
    # ════════════════════════════════════════
    # GÉNÉRER LE QUIZ
    # ════════════════════════════════════════
    def generate_quiz(self, context,
                      num_questions=5,
                      difficulty="medium",
                      language="English"):
        """
        Génère un quiz QCM depuis le contenu du cours
        
        Args:
            context: texte du cours (vient du RAG)
            num_questions: nombre de questions (3, 5, ou 10)
            difficulty: "easy", "medium", ou "hard"
            language: langue des questions
        
        Returns: liste de dictionnaires (questions)
        """
        
        # Instructions selon la difficulté
        difficulty_instructions = {
            "easy": "Simple factual questions. "
                   "Wrong options should be obviously different.",
            "medium": "Mix of factual and conceptual questions. "
                     "Wrong options should be plausible.",
            "hard": "Deep conceptual questions requiring understanding. "
                   "All options should seem plausible."
        }
        
        diff_instruction = difficulty_instructions.get(
            difficulty, difficulty_instructions["medium"]
        )
        
        prompt = f"""You are an expert teacher creating a quiz.

Course content to base questions on:
{context}

Create exactly {num_questions} multiple choice questions.
Difficulty: {difficulty} - {diff_instruction}
Language for questions and options: {language}

STRICT RULES:
1. Questions must be based ONLY on the content above
2. Each question has exactly 4 options (A, B, C, D)
3. Only ONE option is correct
4. Wrong options must be related but incorrect
5. Include a clear explanation for the correct answer

Return ONLY a valid JSON array. No introduction, no explanation, ONLY JSON:
[
  {{
    "question": "Write the question here?",
    "options": {{
      "A": "First option",
      "B": "Second option",
      "C": "Third option",
      "D": "Fourth option"
    }},
    "correct_answer": "A",
    "explanation": "The answer is A because..."
  }}
]"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Nettoyer la réponse
            # Supprimer les backticks markdown si présents
            if "```json" in content:
                content = content.split("```json")[1]
                content = content.split("```")[0]
            elif "```" in content:
                content = content.split("```")[1]
                content = content.split("```")[0]
            
            content = content.strip()
            
            # Parser le JSON
            questions = json.loads(content)
            
            # Valider la structure
            validated = []
            for q in questions:
                if all(key in q for key in 
                      ["question", "options", 
                       "correct_answer", "explanation"]):
                    if all(opt in q["options"] 
                          for opt in ["A", "B", "C", "D"]):
                        validated.append(q)
            
            return validated
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response was: {content}")
            return []
        except Exception as e:
            print(f"Error generating quiz: {e}")
            return []
    
    # ════════════════════════════════════════
    # VÉRIFIER UNE RÉPONSE
    # ════════════════════════════════════════
    def check_answer(self, question_data, user_answer):
        """
        Vérifie si la réponse de l'utilisateur est correcte
        
        Returns: (is_correct, correct_answer, explanation)
        """
        correct = question_data["correct_answer"].upper()
        user = user_answer.upper().strip()
        
        is_correct = (user == correct)
        explanation = question_data["explanation"]
        
        return is_correct, correct, explanation
    
    # ════════════════════════════════════════
    # CALCULER LE SCORE
    # ════════════════════════════════════════
    def calculate_score(self, questions, user_answers):
        """
        Calcule le score final du quiz
        
        Returns: dict avec résultats détaillés
        """
        results = []
        correct_count = 0
        
        for i, question in enumerate(questions):
            user_ans = user_answers.get(i, "")
            is_correct, correct_ans, explanation = \
                self.check_answer(question, user_ans)
            
            if is_correct:
                correct_count += 1
            
            results.append({
                "question_num": i + 1,
                "question": question["question"],
                "user_answer": user_ans,
                "correct_answer": correct_ans,
                "user_answer_text": question["options"].get(
                    user_ans, "Not answered"
                ),
                "correct_answer_text": question["options"].get(
                    correct_ans, ""
                ),
                "is_correct": is_correct,
                "explanation": explanation
            })
        
        total = len(questions)
        score_pct = (correct_count / total * 100) if total > 0 else 0
        
        return {
            "correct": correct_count,
            "total": total,
            "percentage": round(score_pct, 1),
            "results": results
        }