/**
 * API client for the Product Strategy Coach backend
 */

// API URL: set NEXT_PUBLIC_API_URL in production, empty default uses relative URLs on Vercel
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || '';

// Log API configuration (only in browser)
if (typeof window !== 'undefined') {
  console.log('[API Config] Base URL:', API_BASE_URL);
  console.log('[API Config] Environment:', process.env.NODE_ENV);
}

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
  const url = `${API_BASE_URL}/api/coach/start`;
  console.log('[API] Starting coaching session:', url);
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ challenge }),
    });
    
    console.log('[API] Response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('[API] Error response:', errorText);
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }
    
    const data = await response.json();
    console.log('[API] Success:', data);
    return data;
  } catch (error) {
    console.error('[API] Fetch error:', error);
    throw error;
  }
}

/**
 * Send a chat message in the coaching session
 */
export async function sendChatMessage(
  message: string,
  conversationHistory: Array<{ role: string; content: string }>
): Promise<ChatResponse> {
  const url = `${API_BASE_URL}/api/coach/chat`;
  console.log('[API] Sending chat message:', url);
  
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        conversation_history: conversationHistory,
      }),
    });
    
    console.log('[API] Response status:', response.status);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error('[API] Error response:', errorText);
      throw new Error(`API error: ${response.status} - ${errorText}`);
    }
    
    const data = await response.json();
    console.log('[API] Success');
    return data;
  } catch (error) {
    console.error('[API] Fetch error:', error);
    throw error;
  }
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
