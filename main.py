import json
import os
import logging
from typing import Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import random
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
    SIMILARITY_THRESHOLD = 0.75
    MAX_HISTORY_LENGTH = 8
    MAX_CONTEXT_LENGTH = 1000
    MAX_ANSWER_LENGTH = 600
    MIN_ANSWER_LENGTH = 30
    FAISS_RESULTS_COUNT = 3

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
            else:
                self.available = False
        except Exception as e:
            self.available = False
            logger.warning(f"❌ Service Groq non disponible: {e}")

    def generate_response(self, question: str, context: str, history: List[Dict]) -> str:
        if not self.available:
            raise RuntimeError("Service Groq non disponible")
        try:
            messages = [{"role": "system", "content": f"Tu es ANONTCHIGAN. Contexte:\n{context}"}]
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
# RAG SERVICE (Optimisé pour faible RAM)
# ============================================

class RAGService:
    def __init__(self, data_file: str = 'cancer_sein.json'):
        self.questions_data = []
        self.embedding_model = None
        self.index = None
        self.embeddings = None
        self._load_data(data_file)
        self._initialize_embeddings()

    def _load_data(self, data_file: str):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for item in data:
            self.questions_data.append({
                'question_originale': item['question'],
                'question_normalisee': item['question'].lower().strip(),
                'answer': item['answer']
            })
        logger.info(f"✓ {len(self.questions_data)} questions chargées")

    def _initialize_embeddings(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        all_texts = [f"Q: {item['question_originale']} R: {item['answer']}" for item in self.questions_data]
        self.embeddings = self.embedding_model.encode(all_texts, show_progress_bar=False)
        self.embeddings = np.array(self.embeddings).astype('float16')
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        logger.info(f"✓ Index FAISS créé ({len(self.embeddings)} vecteurs)")

    def search(self, query: str, k: int = Config.FAISS_RESULTS_COUNT) -> List[Dict]:
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float16')
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.questions_data):
                similarity = 1 / (1 + distances[0][i])
                results.append({
                    'question': self.questions_data[idx]['question_originale'],
                    'answer': self.questions_data[idx]['answer'],
                    'similarity': similarity
                })
        return results

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
rag_service = RAGService()
conversation_manager = ConversationManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Démarrage ANONTCHIGAN...")
    yield
    logger.info("🛑 Arrêt ANONTCHIGAN...")

app = FastAPI(title="ANONTCHIGAN API", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def serve_home():
    return FileResponse("index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.2.0", "groq_available": groq_service.available}

@app.post("/chat")
async def chat(query: ChatQuery):
    history = conversation_manager.get_history(query.user_id)
    faiss_results = rag_service.search(query.question)
    if faiss_results and faiss_results[0]['similarity'] >= Config.SIMILARITY_THRESHOLD:
        answer = faiss_results[0]['answer']
        conversation_manager.add_message(query.user_id, "user", query.question)
        conversation_manager.add_message(query.user_id, "assistant", answer)
        return ChatResponse(answer=answer, status="success", method="json_direct",
                            score=float(faiss_results[0]['similarity']),
                            matched_question=faiss_results[0]['question'])
    else:
        context = "\n".join([f"{i+1}. Q:{r['question']}\n   R:{r['answer'][:200]}..." for i,r in enumerate(faiss_results[:3])])
        answer = groq_service.generate_response(query.question, context, history) if groq_service.available else \
                 "Veuillez consulter un professionnel de santé pour cette question. 💗"
        conversation_manager.add_message(query.user_id, "user", query.question)
        conversation_manager.add_message(query.user_id, "assistant", answer)
        return ChatResponse(answer=answer, status="success", method="groq_generated",
                            context_used=len(faiss_results[:3]))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
