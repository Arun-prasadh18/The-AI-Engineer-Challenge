"""
Product Strategy Coach API

FastAPI backend that provides coaching endpoints powered by RAG
from Lenny's Podcast transcripts on product strategy.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import sys
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for RAG module import
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag import PodcastRAG

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

# Initialize RAG system with persistent storage
rag_system: Optional[PodcastRAG] = None


def get_rag_system() -> PodcastRAG:
    """Lazy initialization of RAG system."""
    global rag_system
    if rag_system is None:
        persist_dir = Path(__file__).parent.parent / "chroma_db"
        rag_system = PodcastRAG(persist_directory=str(persist_dir))
    return rag_system


# ============== Request/Response Models ==============

class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    sources: list[dict] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)


class CoachingStartRequest(BaseModel):
    challenge: str = Field(..., description="The product challenge or question the user wants help with")


class CoachingStartResponse(BaseModel):
    session_intro: str
    clarifying_questions: list[str]
    relevant_frameworks: list[str]


class GuestListResponse(BaseModel):
    guests: list[dict]
    total_count: int


# ============== Endpoints ==============

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Product Strategy Coach API"}


@app.get("/api/stats")
def get_stats():
    """Get RAG system statistics."""
    try:
        rag = get_rag_system()
        stats = rag.get_collection_stats()
        return {
            "status": "ready",
            "total_chunks": stats["total_chunks"],
            "collection_name": stats["collection_name"],
            "embedding_model": stats["embedding_model"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/coach/start", response_model=CoachingStartResponse)
def start_coaching_session(request: CoachingStartRequest):
    """
    Start a new coaching session by analyzing the user's product challenge.
    Returns clarifying questions and relevant frameworks to explore.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    try:
        rag = get_rag_system()
        
        # Query RAG for relevant context about the challenge
        rag_result = rag.query(
            f"Product strategy advice for: {request.challenge}",
            n_results=5,
            include_context=True
        )
        
        # Generate coaching response with clarifying questions
        system_prompt = """You are an expert product strategy coach drawing from insights 
shared by world-class product leaders on Lenny's Podcast.

Your role is to:
1. Acknowledge the user's challenge with empathy
2. Ask 2-3 clarifying questions to better understand their situation
3. Identify 2-3 relevant frameworks or mental models that might help

Be concise, warm, and actionable. Reference specific guests when appropriate."""

        user_prompt = f"""A product professional needs help with this challenge:
"{request.challenge}"

Context from podcast transcripts:
{rag_result.get('context', '')}

Provide:
1. A brief, empathetic intro (2-3 sentences)
2. 2-3 clarifying questions to understand their situation better
3. 2-3 relevant frameworks or approaches mentioned by podcast guests

Format your response as JSON with keys: intro, questions (array), frameworks (array)"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
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
        raise HTTPException(status_code=500, detail=f"Error starting coaching session: {str(e)}")


@app.post("/api/coach/chat", response_model=ChatResponse)
def coaching_chat(request: ChatRequest):
    """
    Continue a coaching conversation with RAG-powered responses.
    Maintains conversation context and provides sources.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    try:
        rag = get_rag_system()
        
        # Query RAG for relevant context
        rag_result = rag.query(
            request.message,
            n_results=5,
            include_context=True
        )
        
        # Build conversation messages
        system_prompt = """You are an expert product strategy coach. You provide advice based on 
insights from world-class product leaders featured on Lenny's Podcast.

Guidelines:
- Be conversational, warm, and supportive
- Cite specific guests and their advice when relevant
- Provide actionable recommendations
- Ask follow-up questions to go deeper
- Keep responses focused and concise (2-3 paragraphs max)
- When referencing podcast content, mention the guest's name and company"""

        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for msg in request.conversation_history[-10:]:  # Keep last 10 messages
            messages.append(msg)
        
        # Add current message with RAG context
        user_message_with_context = f"""User question: {request.message}

Relevant insights from Lenny's Podcast:
{rag_result.get('context', '')}

Provide a helpful coaching response based on these podcast insights."""
        
        messages.append({"role": "user", "content": user_message_with_context})
        
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
                {"role": "system", "content": "Generate 2-3 brief follow-up questions the user might want to ask next. Return as JSON array."},
                {"role": "user", "content": f"Based on this coaching exchange about: {request.message}\n\nCoach's response: {reply}"}
            ],
            temperature=0.8,
            response_format={"type": "json_object"}
        )
        
        import json
        try:
            followup_data = json.loads(followup_response.choices[0].message.content)
            follow_ups = followup_data.get("questions", followup_data.get("follow_up_questions", []))
        except:
            follow_ups = []
        
        # Format sources for frontend
        sources = [
            {
                "guest": s["guest"],
                "episode": s["episode_title"],
                "youtube_url": s["youtube_url"],
                "timestamp": s.get("timestamp", ""),
                "relevance": round(s["relevance_score"] * 100)
            }
            for s in rag_result.get("sources", [])[:3]  # Top 3 sources
        ]
        
        return ChatResponse(
            reply=reply,
            sources=sources,
            follow_up_questions=follow_ups[:3] if isinstance(follow_ups, list) else []
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in coaching chat: {str(e)}")


@app.get("/api/guests", response_model=GuestListResponse)
def get_guests():
    """Get list of all guests in the product strategy collection."""
    try:
        rag = get_rag_system()
        
        # Get all unique guests from the collection
        results = rag.collection.get(include=["metadatas"])
        
        guests_set = {}
        for metadata in results.get("metadatas", []):
            guest = metadata.get("guest", "Unknown")
            if guest not in guests_set:
                guests_set[guest] = {
                    "name": guest,
                    "episode_title": metadata.get("episode_title", ""),
                    "youtube_url": metadata.get("youtube_url", "")
                }
        
        guests_list = list(guests_set.values())
        guests_list.sort(key=lambda x: x["name"])
        
        return GuestListResponse(
            guests=guests_list,
            total_count=len(guests_list)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching guests: {str(e)}")


@app.post("/api/ask-guest")
def ask_specific_guest(guest_name: str, question: str):
    """
    Ask a question to a specific guest based on their episode content.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    
    try:
        rag = get_rag_system()
        
        # Query with guest filter
        question_with_guest = f"What does {guest_name} say about: {question}"
        rag_result = rag.query(question_with_guest, n_results=5, include_context=True)
        
        # Filter to only include chunks from this guest
        relevant_sources = [
            s for s in rag_result.get("sources", [])
            if guest_name.lower() in s.get("guest", "").lower()
        ]
        
        system_prompt = f"""You are channeling the perspective of {guest_name} based on their 
interview on Lenny's Podcast. Answer as if you are sharing {guest_name}'s views and experiences.

Guidelines:
- Stay true to what {guest_name} actually said in their interview
- Use first person when appropriate ("In my experience at [company]...")
- Be specific about frameworks and examples they shared
- If the context doesn't contain relevant info from this guest, acknowledge that"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {question}\n\nContext from {guest_name}'s interview:\n{rag_result.get('context', '')}"}
            ],
            temperature=0.7
        )
        
        return {
            "guest": guest_name,
            "answer": response.choices[0].message.content,
            "sources": relevant_sources[:3],
            "youtube_url": relevant_sources[0].get("youtube_url", "") if relevant_sources else ""
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error asking guest: {str(e)}")


# Legacy endpoint for backwards compatibility
@app.post("/api/chat")
def chat(request: ChatRequest):
    """Legacy chat endpoint - redirects to coaching chat."""
    return coaching_chat(request)
