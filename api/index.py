"""
Product Strategy Coach API

FastAPI backend that provides coaching endpoints.
Uses RAG with ChromaDB locally, falls back to direct OpenAI on Vercel.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import os
from typing import Optional

# Try to load dotenv for local development
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(
    title="Product Strategy Coach",
    description="AI-powered product strategy coaching based on insights from Lenny's Podcast",
    version="1.0.0"
)

# CORS so the frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Try to import RAG system (only available with full dependencies)
rag_system = None
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from rag import PodcastRAG
    
    persist_dir = Path(__file__).parent.parent / "chroma_db"
    if persist_dir.exists():
        rag_system = PodcastRAG(persist_directory=str(persist_dir))
        print("✓ RAG system loaded with ChromaDB")
except ImportError:
    print("⚠ RAG dependencies not available, using direct OpenAI mode")
except Exception as e:
    print(f"⚠ Could not load RAG system: {e}")


# ============== Request/Response Models ==============

class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    sources: list[dict] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


class CoachingStartRequest(BaseModel):
    challenge: str = Field(..., description="The product challenge or question")


class CoachingStartResponse(BaseModel):
    session_intro: str
    clarifying_questions: list[str]
    relevant_frameworks: list[str]


# ============== Coaching System Prompt ==============

COACH_SYSTEM_PROMPT = """You are an expert product strategy coach with deep knowledge from 
interviewing 50+ world-class product leaders including:
- Shreyas Doshi (ex-Stripe, Twitter, Google)
- Marty Cagan (Silicon Valley Product Group)
- Gibson Biddle (ex-Netflix)
- Lenny Rachitsky (ex-Airbnb)
- And many more top PMs from companies like Meta, Airbnb, Spotify, Notion, etc.

Your role is to:
1. Listen empathetically to product challenges
2. Ask clarifying questions to understand context
3. Share relevant frameworks and mental models
4. Provide actionable, specific advice
5. Reference insights from product leaders when relevant

Be conversational, warm, and supportive. Keep responses focused and concise (2-3 paragraphs max).
When you reference advice, mention the source (e.g., "As Shreyas Doshi often says...")"""


# ============== Endpoints ==============

@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok", 
        "service": "Product Strategy Coach API",
        "mode": "rag" if rag_system else "direct"
    }


@app.get("/api/stats")
def get_stats():
    """Get system statistics."""
    if rag_system:
        stats = rag_system.get_collection_stats()
        return {
            "status": "ready",
            "mode": "rag",
            "total_chunks": stats["total_chunks"],
            "collection_name": stats["collection_name"]
        }
    return {
        "status": "ready",
        "mode": "direct",
        "message": "Using direct OpenAI mode (no RAG)"
    }


@app.post("/api/coach/start", response_model=CoachingStartResponse)
def start_coaching_session(request: CoachingStartRequest):
    """Start a new coaching session."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    try:
        context = ""
        if rag_system:
            # Get relevant context from RAG
            rag_result = rag_system.query(
                f"Product strategy advice for: {request.challenge}",
                n_results=3,
                include_context=True
            )
            context = f"\n\nRelevant insights from podcast interviews:\n{rag_result.get('context', '')}"
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": COACH_SYSTEM_PROMPT},
                {"role": "user", "content": f"""A product professional needs help with this challenge:
"{request.challenge}"
{context}

Provide:
1. A brief, empathetic intro (2-3 sentences)
2. 2-3 clarifying questions to understand their situation better
3. 2-3 relevant frameworks or approaches

Format your response as JSON with keys: intro, questions (array), frameworks (array)"""}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        
        import json
        result = json.loads(response.choices[0].message.content)
        
        return CoachingStartResponse(
            session_intro=result.get("intro", "Let me help you with that challenge."),
            clarifying_questions=result.get("questions", []),
            relevant_frameworks=result.get("frameworks", [])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/coach/chat", response_model=ChatResponse)
def coaching_chat(request: ChatRequest):
    """Continue a coaching conversation."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    try:
        context = ""
        sources = []
        
        if rag_system:
            # Get relevant context from RAG
            rag_result = rag_system.query(
                request.message,
                n_results=3,
                include_context=True
            )
            context = f"\n\nRelevant insights from podcast interviews:\n{rag_result.get('context', '')}"
            sources = [
                {
                    "guest": s["guest"],
                    "episode": s["episode_title"],
                    "youtube_url": s["youtube_url"],
                    "timestamp": s.get("timestamp", ""),
                    "relevance": round(s["relevance_score"] * 100)
                }
                for s in rag_result.get("sources", [])[:3]
            ]
        
        messages = [{"role": "system", "content": COACH_SYSTEM_PROMPT}]
        
        # Add conversation history
        for msg in request.conversation_history[-10:]:
            messages.append(msg)
        
        # Add current message
        messages.append({
            "role": "user", 
            "content": f"{request.message}{context}"
        })
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=800
        )
        
        reply = response.choices[0].message.content
        
        # Generate follow-up questions
        followup_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Generate 2-3 brief follow-up questions. Return as JSON with key 'questions' (array)."},
                {"role": "user", "content": f"Based on: {request.message}\n\nResponse: {reply}"}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        
        import json
        try:
            followup_data = json.loads(followup_response.choices[0].message.content)
            follow_ups = followup_data.get("questions", [])[:3]
        except:
            follow_ups = []
        
        return ChatResponse(
            reply=reply,
            sources=sources,
            follow_up_questions=follow_ups
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


# Legacy endpoint
@app.post("/api/chat")
def chat(request: ChatRequest):
    """Legacy chat endpoint."""
    return coaching_chat(request)
