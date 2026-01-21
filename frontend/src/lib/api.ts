/**
 * API client for the Product Strategy Coach backend
 */

// API URL: set NEXT_PUBLIC_API_URL in production, defaults to localhost for dev
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Source {
  guest: string;
  episode: string;
  youtube_url: string;
  timestamp: string;
  relevance: number;
}

export interface ChatResponse {
  reply: string;
  sources: Source[];
  follow_up_questions: string[];
}

export interface CoachingStartResponse {
  session_intro: string;
  clarifying_questions: string[];
  relevant_frameworks: string[];
}

export interface Guest {
  name: string;
  episode_title: string;
  youtube_url: string;
}

export interface StatsResponse {
  status: string;
  total_chunks: number;
  collection_name: string;
  embedding_model: string;
}

/**
 * Start a new coaching session with a product challenge
 */
export async function startCoachingSession(challenge: string): Promise<CoachingStartResponse> {
  const response = await fetch(`${API_BASE_URL}/api/coach/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ challenge }),
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Send a chat message in the coaching session
 */
export async function sendChatMessage(
  message: string,
  conversationHistory: Array<{ role: string; content: string }>
): Promise<ChatResponse> {
  const response = await fetch(`${API_BASE_URL}/api/coach/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      conversation_history: conversationHistory,
    }),
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Get list of all available guests
 */
export async function getGuests(): Promise<{ guests: Guest[]; total_count: number }> {
  const response = await fetch(`${API_BASE_URL}/api/guests`);
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Ask a specific guest a question
 */
export async function askGuest(guestName: string, question: string): Promise<{
  guest: string;
  answer: string;
  sources: Source[];
  youtube_url: string;
}> {
  const params = new URLSearchParams({ guest_name: guestName, question });
  const response = await fetch(`${API_BASE_URL}/api/ask-guest?${params}`, {
    method: 'POST',
  });
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return response.json();
}

/**
 * Get RAG system stats
 */
export async function getStats(): Promise<StatsResponse> {
  const response = await fetch(`${API_BASE_URL}/api/stats`);
  
  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }
  
  return response.json();
}
