import json
import os
import logging
from typing import Dict, List, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import random
from difflib import SequenceMatcher
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from dotenv import load_dotenv
# ============================================
# CONFIGURATION ET LOGGING
# ============================================

load_dotenv()
print("GROQ_API_KEY =", os.getenv("GROQ_API_KEY"))

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
    """Configuration optimisée pour éviter les coupures"""
    SIMILARITY_THRESHOLD = 0.75
    MAX_HISTORY_LENGTH = 8
    MAX_CONTEXT_LENGTH = 1000
    MAX_ANSWER_LENGTH = 600  # Augmenté significativement
    FAISS_RESULTS_COUNT = 3
    MIN_ANSWER_LENGTH = 30

# ============================================
# MODÈLES DE DONNÉES
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
# SERVICE GROQ CORRIGÉ
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
                logger.warning("⚠️ Clé API Groq manquante — vérifie la variable d'environnement GROQ_API_KEY")
                return

            # ✅ Initialisation sans argument 'proxies'
            self.client = Groq(api_key=api_key)

            # ✅ Petit test de requête pour confirmer la disponibilité
            test_response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=5
            )

            if test_response:
                self.available = True
                logger.info("✅ Service Groq initialisé et opérationnel")
            else:
                self.available = False
                logger.warning("⚠️ Test Groq échoué, client non disponible")

        except Exception as e:
            self.available = False
            logger.warning(f"❌ Service Groq non disponible : {str(e)}")








    def generate_response(self, question: str, context: str, history: List[Dict]) -> str:
        """Génère une réponse complète sans coupure"""
        if not self.available:
            raise RuntimeError("Service Groq non disponible")
        
        try:
            # Préparer le contexte optimisé
            context_short = self._prepare_context(context)
            
            # Préparer les messages
            messages = self._prepare_messages(question, context_short, history)
            
            logger.info("🤖 Génération avec Groq...")
            
            # AUGMENTER SIGNIFICATIVEMENT les tokens pour éviter les coupures
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                max_tokens=600,  # Augmenté pour éviter coupures
                temperature=0.7,
                top_p=0.9,
            )
            
            answer = response.choices[0].message.content.strip()
            answer = self._clean_response(answer)
            
            # Validation renforcée
            if not self._is_valid_answer(answer):
                raise ValueError("Réponse trop courte")
                
            # Vérification et correction des coupures
            answer = self._ensure_complete_response(answer)
            
            logger.info(f"✓ Réponse générée ({len(answer)} caractères)")
            return answer
            
        except Exception as e:
            logger.error(f"Erreur Groq: {str(e)}")
            raise
    
    def _prepare_context(self, context: str) -> str:
        """Prépare un contexte concis"""
        lines = context.split('\n')[:5]  # Réduit pour focus
        context_short = '\n'.join(lines)
        if len(context_short) > Config.MAX_CONTEXT_LENGTH:
            context_short = context_short[:Config.MAX_CONTEXT_LENGTH-3] + "..."
        return context_short
    
    def _prepare_messages(self, question: str, context: str, history: List[Dict]) -> List[Dict]:
        """Prépare les messages avec instructions claires contre les coupures"""
        
        system_prompt = f"""Tu es ANONTCHIGAN, assistante IA professionnelle spécialisée dans la sensibilisation au cancer du sein au Bénin.

CONTEXTE À UTILISER :
{context}

RÈGLES CRITIQUES :
1. FOURNIR DES RÉPONSES COMPLÈTES - ne JAMAIS couper une phrase en milieu de mot
2. Si tu donnes un conseil de prévention, aller deux fois à la ligne AVANT de le donner
3. Tes réponses doivent se terminer naturellement par un point final

STYLE :
- Professionnel, clair, empathique
- Réponses directes sans formules introductives
- CONCIS mais COMPLET
- Humour humain léger 1 fois sur 3 réponses
- Emojis à utiliser fréquemment : 💗 🌸 😊 🇧🇯

STRUCTURE DES RÉPONSES :
1. Réponse principale basée sur le contexte
2- N'invente PAS d'informations
3- Si le contexte ne couvre pas la question, recommande de consulter un professionnel de santé au Bénin
4- ENSGMM se définit comme l'Ecole Nationale Supérieure de Génie Mathématique et Modélisation 
5- Utilise des connecteurs logiques pour fluidifier et produire des réponses naturelles et cohérentes

DIRECTIVES ANTI-CONPUTURE :
- Vérifie que ta réponse est complète avant de terminer
- Ne coupe PAS en milieu de phrase ou de mot
- Utilise "Atassa!" ou "Atassaaaaa!" en debut de phrase en cas dh'humour et d'étonnemnt extrême
- Termine par un point final approprié
- Si tu mentionnes des noms (créateurs, etc.), assure-toi qu'ils sont COMPLETS

Conseils de prévention : seulement si pertinents et si demandés."""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Historique récent
        for msg in history[-4:]:
            messages.append(msg)
        
        # Question actuelle avec instruction explicite
        messages.append({
            "role": "user", 
            "content": f"QUESTION: {question}\n\nIMPORTANT : Réponds de façon COMPLÈTE sans couper ta réponse. Termine par un point final. Si conseil de prévention, va à la ligne avant."
        })
        
        return messages
    
    def _clean_response(self, answer: str) -> str:
        """Nettoie la réponse en gardant la personnalité"""
        
        # Supprimer les introductions verbeuses
        unwanted_intros = [
            'bonjour', 'salut', 'coucou', 'hello', 'akwè', 'yo', 'bonsoir', 'hi',
            'excellente question', 'je suis ravi', 'permettez-moi', 'tout d abord',
            'premièrement', 'pour commencer', 'en tant qu', 'je suis anontchigan'
        ]
        
        answer_lower = answer.lower()
        for phrase in unwanted_intros:
            if answer_lower.startswith(phrase):
                sentences = answer.split('.')
                if len(sentences) > 1:
                    answer = '.'.join(sentences[1:]).strip()
                    if answer:
                        answer = answer[0].upper() + answer[1:]
                break
        
        return answer.strip()
    
    def _is_valid_answer(self, answer: str) -> bool:
        """Valide que la réponse est acceptable"""
        return (len(answer) >= Config.MIN_ANSWER_LENGTH and 
                not answer.lower().startswith(('je ne sais pas', 'désolé', 'sorry')))
    
    def _ensure_complete_response(self, answer: str) -> str:
        """Garantit que la réponse est complète et non coupée"""
        if not answer:
            return answer
            
        # Détecter les signes de coupure
        cut_indicators = [
            answer.endswith('...'),
            answer.endswith(','),
            answer.endswith(';'),
            answer.endswith(' '),
            any(word in answer.lower() for word in ['http', 'www.', '.com']),  # URLs coupées
            '...' in answer[-10:]  # Points de suspension vers la fin
        ]
        
        if any(cut_indicators):
            logger.warning("⚠️  Détection possible de réponse coupée")
            
            # Trouver la dernière phrase complète
            last_period = answer.rfind('.')
            last_exclamation = answer.rfind('!')
            last_question = answer.rfind('?')
            
            sentence_end = max(last_period, last_exclamation, last_question)
            
            if sentence_end > 0 and sentence_end >= len(answer) - 5:
                # Garder jusqu'à la dernière phrase complète
                answer = answer[:sentence_end + 1]
            else:
                # Si pas de ponctuation claire, nettoyer la fin
                answer = answer.rstrip(' ,;...')
                if not answer.endswith(('.', '!', '?')):
                    answer += '.'
        
        # Formater les conseils de prévention avec saut de ligne
        prevention_phrases = [
            'conseil de prévention',
            'pour prévenir',
            'je recommande',
            'il est important de',
            'n oubliez pas de'
        ]
        
        # Vérifier si un conseil de prévention est présent
        has_prevention_advice = any(phrase in answer.lower() for phrase in prevention_phrases)
        
        if has_prevention_advice:
            # Essayer d'insérer un saut de ligne avant le conseil
            lines = answer.split('. ')
            if len(lines) > 1:
                # Trouver la ligne qui contient le conseil
                for i, line in enumerate(lines[1:], 1):
                    if any(phrase in line.lower() for phrase in prevention_phrases):
                        # Insérer un saut de ligne avant cette ligne
                        lines[i] = '\n' + lines[i]
                        answer = '. '.join(lines)
                        break
        
        return answer

# ============================================
# SERVICES RAG ET CONVERSATION
# ============================================

class RAGService:
    """Service RAG avec recherche améliorée"""
    
    def __init__(self, data_file: str = 'cancer_sein.json'):
        self.questions_data = []
        self.embedding_model = None
        self.index = None
        self.embeddings = None
        self._load_data(data_file)
        self._initialize_embeddings()
    
    def _load_data(self, data_file: str):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for item in data:
                self.questions_data.append({
                    'question_originale': item['question'],
                    'question_normalisee': item['question'].lower().strip(),
                    'answer': item['answer']
                })
            
            logger.info(f"✓ {len(self.questions_data)} questions chargées")
            
        except Exception as e:
            logger.error(f"Erreur chargement données: {str(e)}")
            raise
    
    def _initialize_embeddings(self):
        try:
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            
            all_texts = [
                f"Q: {item['question_originale']} R: {item['answer']}"
                for item in self.questions_data
            ]
            
            self.embeddings = self.embedding_model.encode(all_texts, show_progress_bar=False)
            self.embeddings = np.array(self.embeddings).astype('float32')
            
            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(self.embeddings)
            
            logger.info(f"✓ Index FAISS créé ({len(self.embeddings)} vecteurs)")
            
        except Exception as e:
            logger.error(f"Erreur initialisation embeddings: {str(e)}")
            raise
    
    def search(self, query: str, k: int = Config.FAISS_RESULTS_COUNT) -> List[Dict]:
        """Recherche optimisée dans FAISS"""
        try:
            query_embedding = self.embedding_model.encode([query])
            query_embedding = np.array(query_embedding).astype('float32')
            
            distances, indices = self.index.search(query_embedding, k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.questions_data):
                    similarity = 1 / (1 + distances[0][i])
                    results.append({
                        'question': self.questions_data[idx]['question_originale'],
                        'answer': self.questions_data[idx]['answer'],
                        'similarity': similarity,
                        'distance': distances[0][i]
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Erreur recherche FAISS: {str(e)}")
            return []

class ConversationManager:
    """Gestionnaire de conversations"""
    
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
rag_service = RAGService()
conversation_manager = ConversationManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Démarrage d'ANONTCHIGAN anti-coupure...")
    yield
    logger.info("🛑 Arrêt d'ANONTCHIGAN...")

app = FastAPI(
    title="ANONTCHIGAN API",
    description="Assistant IA pour la sensibilisation au cancer du sein",
    version="2.2.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def serve_home():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "2.2.0",
        "groq_available": groq_service.available,
        "optimizations": ["anti_coupure", "conseils_formates", "reponses_completes"]
    }

@app.post("/chat")
async def chat(query: ChatQuery):
    logger.info(f"📥 Question: {query.question}")
    
    try:
        history = conversation_manager.get_history(query.user_id)
        
        # Gestion des salutations
        salutations = ["cc","bonjour", "salut", "coucou", "hello", "akwe", "yo", "bonsoir", "hi"]
        question_lower = query.question.lower().strip()
        
        if any(salut == question_lower for salut in salutations):
            responses = [
                "Je suis ANONTCHIGAN, assistante pour la sensibilisation au cancer du sein. Comment puis-je vous aider ? 💗",
                "Bonjour ! Je suis ANONTCHIGAN. Que souhaitez-vous savoir sur le cancer du sein ? 🌸",
                "ANONTCHIGAN à votre service. Posez-moi vos questions sur la prévention du cancer du sein. 😊"
            ]
            answer = random.choice(responses)
            
            conversation_manager.add_message(query.user_id, "user", query.question)
            conversation_manager.add_message(query.user_id, "assistant", answer)
            
            return ChatResponse(
                answer=answer,
                status="success",
                method="salutation"
            )
        
        # Recherche FAISS
        logger.info("🔍 Recherche FAISS...")
        faiss_results = rag_service.search(query.question)
        
        if not faiss_results:
            answer = "Les informations disponibles ne couvrent pas ce point spécifique. Je vous recommande de consulter un professionnel de santé au Bénin pour des conseils adaptés. 💗"
            conversation_manager.add_message(query.user_id, "user", query.question)
            conversation_manager.add_message(query.user_id, "assistant", answer)
            
            return ChatResponse(
                answer=answer,
                status="info",
                method="no_result"
            )
        
        best_result = faiss_results[0]
        similarity = best_result['similarity']
        
        logger.info(f"📊 Meilleure similarité: {similarity:.3f}")
        
        # Décision : Réponse directe vs Génération
        if similarity >= Config.SIMILARITY_THRESHOLD:
            logger.info(f"✅ Haute similarité → Réponse directe")
            answer = best_result['answer']
            
            # S'assurer que les réponses directes ne sont pas coupées non plus
            if len(answer) > Config.MAX_ANSWER_LENGTH:
                answer = answer[:Config.MAX_ANSWER_LENGTH-3] + "..."
            
            conversation_manager.add_message(query.user_id, "user", query.question)
            conversation_manager.add_message(query.user_id, "assistant", answer)
            
            return ChatResponse(
                answer=answer,
                status="success",
                method="json_direct",
                score=float(similarity),
                matched_question=best_result['question']
            )
        else:
            logger.info(f"🤖 Similarité modérée → Génération Groq")
            
            # Préparer le contexte
            context_parts = []
            for i, result in enumerate(faiss_results[:3], 1):
                answer_truncated = result['answer']
                if len(answer_truncated) > 200:  # Réduit pour laisser plus de place à la génération
                    answer_truncated = answer_truncated[:197] + "..."
                context_parts.append(f"{i}. Q: {result['question']}\n   R: {answer_truncated}")
            
            context = "\n\n".join(context_parts)
            
            # Génération avec Groq
            try:
                if groq_service.available:
                    generated_answer = groq_service.generate_response(query.question, context, history)
                else:
                    generated_answer = "Je vous recommande de consulter un professionnel de santé pour cette question spécifique. La prévention précoce est essentielle. 💗"
            except Exception as e:
                logger.warning(f"Génération échouée: {str(e)}")
                generated_answer = "Pour des informations précises sur ce sujet, veuillez consulter un médecin ou un centre de santé spécialisé au Bénin. 🌸"
            
            conversation_manager.add_message(query.user_id, "user", query.question)
            conversation_manager.add_message(query.user_id, "assistant", generated_answer)
            
            return ChatResponse(
                answer=generated_answer,
                status="success",
                method="groq_generated",
                score=float(similarity),
                context_used=len(faiss_results[:3])
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
    logger.info("✓ ANONTCHIGAN ANTI-COUPURE - Prêt!")
    logger.info("  - Max tokens: 550 (augmenté)")
    logger.info("  - Détection coupures: Activée")
    logger.info("  - Conseils formatés: Avec sauts de ligne")
    logger.info(f"  - Génération: {'Groq ⚡' if groq_service.available else 'Fallback'}")
    logger.info("="*50 + "\n")
    
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)