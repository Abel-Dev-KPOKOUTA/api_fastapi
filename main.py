import json
import os
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import random
from difflib import SequenceMatcher

from dotenv import load_dotenv

# ============================================
# CONFIGURATION ET LOGGING
# ============================================

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('anontchigan.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("ANONTCHIGAN")

class Config:
    SIMILARITY_THRESHOLD = 0.65  # Abaissé car recherche textuelle moins précise
    MAX_HISTORY_LENGTH = 8
    MAX_CONTEXT_LENGTH = 1000
    MAX_ANSWER_LENGTH = 600
    MIN_ANSWER_LENGTH = 30
    RESULTS_COUNT = 3

# ============================================
# MODELES DE DONNEES
# ============================================

class ChatQuery(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    user_id: str = Field(..., min_length=1)

class ChatResponse(BaseModel):
    answer: str
    status: str
    method: str
    score: Optional[float] = None
    matched_question: Optional[str] = None
    context_used: Optional[int] = None

# ============================================
# GROQ SERVICE
# ============================================

class GroqService:
    def __init__(self):
        self.client = None
        self.available = False
        self._initialize_groq()

    def _initialize_groq(self):
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                logger.warning("⚠️ Clé API Groq manquante")
                return
            self.client = Groq(api_key=api_key)
            test_response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )
            if test_response:
                self.available = True
                logger.info("✅ Service Groq initialisé")
        except Exception as e:
            self.available = False
            logger.warning(f"❌ Service Groq non disponible: {e}")

    def generate_response(self, question: str, context: str, history: List[Dict]) -> str:
        if not self.available:
            raise RuntimeError("Service Groq non disponible")
        try:
            system_prompt = f"""Tu es ANONTCHIGAN, assistante IA professionnelle spécialisée dans la sensibilisation au cancer du sein au Bénin.

CONTEXTE À UTILISER :
{context}

RÈGLES :
- Réponses COMPLÈTES et naturelles
- Professionnel, clair, empathique
- Emojis : 💗 🌸 😊 🇧🇯
- N'invente PAS d'informations
- Recommande un professionnel si nécessaire"""

            messages = [{"role": "system", "content": system_prompt}]
            for msg in history[-4:]:
                messages.append(msg)
            messages.append({"role": "user", "content": question})
            
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=600,
                temperature=0.7,
                top_p=0.9
            )
            
            answer = response.choices[0].message.content.strip()
            if len(answer) < Config.MIN_ANSWER_LENGTH:
                answer += "."
            return answer
        except Exception as e:
            logger.error(f"Erreur Groq: {e}")
            return "Pour des informations précises, consultez un professionnel de santé. 💗"

# ============================================
# RAG SERVICE ULTRA-LÉGER (sans embeddings)
# ============================================

class RAGServiceUltraLight:
    """Version ultra-légère sans PyTorch ni FAISS - seulement recherche textuelle"""
    
    def __init__(self, data_file: str = 'cancer_sein.json'):
        self.questions_data = []
        self._load_data(data_file)

    def _load_data(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            self.questions_data.append({
                'question_originale': item['question'],
                'question_normalisee': item['question'].lower().strip(),
                'answer': item['answer']
            })
        logger.info(f"✓ {len(self.questions_data)} questions chargées (mode ultra-léger)")

    def _calculate_similarity(self, query: str, question: str) -> float:
        """Calcule la similarité entre deux textes"""
        query_lower = query.lower()
        question_lower = question.lower()
        
        # 1. Similarité de séquence (SequenceMatcher)
        seq_similarity = SequenceMatcher(None, query_lower, question_lower).ratio()
        
        # 2. Mots en commun (Jaccard)
        query_words = set(query_lower.split())
        question_words = set(question_lower.split())
        
        if not query_words or not question_words:
            return seq_similarity
        
        intersection = len(query_words & question_words)
        union = len(query_words | question_words)
        jaccard = intersection / union if union > 0 else 0
        
        # 3. Correspondance de mots-clés importants
        important_words = ['cancer', 'sein', 'symptôme', 'dépistage', 'mammographie', 
                          'auto-examen', 'prévention', 'traitement', 'risque']
        
        keyword_match = 0
        for word in important_words:
            if word in query_lower and word in question_lower:
                keyword_match += 0.05
        
        # Combinaison pondérée
        final_score = (seq_similarity * 0.4) + (jaccard * 0.4) + min(keyword_match, 0.2)
        
        return min(final_score, 1.0)

    def search(self, query: str, k: int = Config.RESULTS_COUNT) -> List[Dict]:
        """Recherche par similarité textuelle"""
        results = []
        
        for item in self.questions_data:
            similarity = self._calculate_similarity(query, item['question_originale'])
            
            results.append({
                'question': item['question_originale'],
                'answer': item['answer'],
                'similarity': similarity,
                'distance': 1 - similarity
            })
        
        # Trier par similarité décroissante
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:k]

# ============================================
# CONVERSATION MANAGER
# ============================================

class ConversationManager:
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}

    def get_history(self, user_id: str) -> List[Dict]:
        return self.conversations.get(user_id, [])

    def add_message(self, user_id: str, role: str, content: str):
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        self.conversations[user_id].append({"role": role, "content": content})
        if len(self.conversations[user_id]) > Config.MAX_HISTORY_LENGTH * 2:
            self.conversations[user_id] = self.conversations[user_id][-Config.MAX_HISTORY_LENGTH * 2:]

# ============================================
# INITIALISATION ET ENDPOINTS
# ============================================

groq_service = GroqService()
rag_service = RAGServiceUltraLight()
conversation_manager = ConversationManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Démarrage ANONTCHIGAN (version ultra-légère)...")
    yield
    logger.info("🛑 Arrêt ANONTCHIGAN...")

app = FastAPI(
    title="ANONTCHIGAN API",
    description="Assistant IA pour la sensibilisation au cancer du sein (version optimisée 512 MB)",
    version="2.3.0-light",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {
        "service": "ANONTCHIGAN API",
        "version": "2.3.0-light",
        "status": "operational",
        "features": ["text_search", "groq_generation"],
        "memory": "optimized for 512MB"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.3.0-light",
        "groq_available": groq_service.available,
        "search_method": "text_similarity",
        "memory_optimized": True
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(query: ChatQuery):
    logger.info(f"📥 Question: {query.question}")
    
    try:
        history = conversation_manager.get_history(query.user_id)
        
        # Gestion des salutations
        salutations = ["cc", "bonjour", "salut", "coucou", "hello", "akwe", "yo", "bonsoir", "hi"]
        if query.question.lower().strip() in salutations:
            responses = [
                "Je suis ANONTCHIGAN, assistante pour la sensibilisation au cancer du sein. Comment puis-je vous aider ? 💗",
                "Bonjour ! Je suis ANONTCHIGAN. Que souhaitez-vous savoir sur le cancer du sein ? 🌸",
                "ANONTCHIGAN à votre service. Posez-moi vos questions sur la prévention du cancer du sein. 😊"
            ]
            answer = random.choice(responses)
            conversation_manager.add_message(query.user_id, "user", query.question)
            conversation_manager.add_message(query.user_id, "assistant", answer)
            return ChatResponse(answer=answer, status="success", method="salutation")
        
        # Recherche textuelle
        logger.info("🔍 Recherche textuelle...")
        results = rag_service.search(query.question)
        
        if not results:
            answer = "Les informations disponibles ne couvrent pas ce point spécifique. Je vous recommande de consulter un professionnel de santé au Bénin pour des conseils adaptés. 💗"
            conversation_manager.add_message(query.user_id, "user", query.question)
            conversation_manager.add_message(query.user_id, "assistant", answer)
            return ChatResponse(answer=answer, status="info", method="no_result")
        
        best_result = results[0]
        similarity = best_result['similarity']
        
        logger.info(f"📊 Meilleure similarité: {similarity:.3f}")
        
        # Décision : Réponse directe vs Génération
        if similarity >= Config.SIMILARITY_THRESHOLD:
            logger.info("✅ Haute similarité → Réponse directe")
            answer = best_result['answer']
            
            if len(answer) > Config.MAX_ANSWER_LENGTH:
                answer = answer[:Config.MAX_ANSWER_LENGTH-3] + "..."
            
            conversation_manager.add_message(query.user_id, "user", query.question)
            conversation_manager.add_message(query.user_id, "assistant", answer)
            
            return ChatResponse(
                answer=answer,
                status="success",
                method="text_search_direct",
                score=float(similarity),
                matched_question=best_result['question']
            )
        else:
            logger.info("🤖 Similarité modérée → Génération Groq")
            
            # Préparer le contexte
            context_parts = []
            for i, result in enumerate(results[:3], 1):
                answer_truncated = result['answer'][:200]
                if len(result['answer']) > 200:
                    answer_truncated += "..."
                context_parts.append(f"{i}. Q: {result['question']}\n   R: {answer_truncated}")
            
            context = "\n\n".join(context_parts)
            
            # Génération avec Groq
            if groq_service.available:
                generated_answer = groq_service.generate_response(query.question, context, history)
            else:
                generated_answer = "Je vous recommande de consulter un professionnel de santé pour cette question spécifique. La prévention précoce est essentielle. 💗"
            
            conversation_manager.add_message(query.user_id, "user", query.question)
            conversation_manager.add_message(query.user_id, "assistant", generated_answer)
            
            return ChatResponse(
                answer=generated_answer,
                status="success",
                method="groq_generated",
                score=float(similarity),
                context_used=len(results[:3])
            )
            
    except Exception as e:
        logger.error(f"❌ Erreur: {str(e)}")
        error_message = "Désolé, une erreur s'est produite. Veuillez réessayer."
        
        conversation_manager.add_message(query.user_id, "user", query.question)
        conversation_manager.add_message(query.user_id, "assistant", error_message)
        
        return ChatResponse(
            answer=error_message,
            status="error",
            method="error"
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("\n" + "="*50)
    logger.info("✓ ANONTCHIGAN ULTRA-LÉGER - Prêt!")
    logger.info("  - Recherche: Textuelle (sans ML)")
    logger.info("  - Mémoire: < 200 MB")
    logger.info(f"  - Génération: {'Groq ⚡' if groq_service.available else 'Fallback'}")
    logger.info("="*50 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)