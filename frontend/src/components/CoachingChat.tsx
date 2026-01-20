'use client';

import { useState, useRef, useEffect } from 'react';
import { sendChatMessage, startCoachingSession, Source } from '@/lib/api';
import ChatMessage from './ChatMessage';
import SuggestedQuestions from './SuggestedQuestions';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
}

export default function CoachingChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [suggestedQuestions, setSuggestedQuestions] = useState<string[]>([]);
  const [sessionStarted, setSessionStarted] = useState(false);
  const [challenge, setChallenge] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleStartSession = async () => {
    if (!challenge.trim()) return;
    
    setIsLoading(true);
    setSessionStarted(true);
    
    try {
      const response = await startCoachingSession(challenge);
      
      // Add user's challenge as first message
      setMessages([
        { role: 'user', content: challenge },
        { 
          role: 'assistant', 
          content: `${response.session_intro}\n\n**Clarifying Questions:**\n${response.clarifying_questions.map((q, i) => `${i + 1}. ${q}`).join('\n')}\n\n**Relevant Frameworks to Explore:**\n${response.relevant_frameworks.map(f => `• ${f}`).join('\n')}`
        }
      ]);
      
      setSuggestedQuestions(response.clarifying_questions);
    } catch (error) {
      console.error('Error starting session:', error);
      setMessages([
        { role: 'user', content: challenge },
        { role: 'assistant', content: 'Sorry, I encountered an error starting the coaching session. Please try again.' }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async (messageText?: string) => {
    const text = messageText || input;
    if (!text.trim() || isLoading) return;
    
    const userMessage: Message = { role: 'user', content: text };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setSuggestedQuestions([]);
    
    try {
      // Build conversation history for context
      const conversationHistory = messages.map(m => ({
        role: m.role,
        content: m.content
      }));
      
      const response = await sendChatMessage(text, conversationHistory);
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.reply,
        sources: response.sources
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      setSuggestedQuestions(response.follow_up_questions);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: 'Sorry, I encountered an error. Please try again.' }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (sessionStarted) {
        handleSendMessage();
      } else {
        handleStartSession();
      }
    }
  };

  // Welcome screen before session starts
  if (!sessionStarted) {
    return (
      <div className="flex flex-col h-full">
        {/* Header */}
        <div className="text-center py-8 px-4">
          <div className="text-6xl mb-4">🎯</div>
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Product Strategy Coach
          </h1>
          <p className="text-gray-600 max-w-md mx-auto">
            Get personalized advice based on insights from 50+ product leaders 
            featured on Lenny&apos;s Podcast
          </p>
        </div>

        {/* Challenge input */}
        <div className="flex-1 flex flex-col items-center justify-center px-4">
          <div className="w-full max-w-2xl">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              What product challenge are you facing?
            </label>
            <textarea
              value={challenge}
              onChange={(e) => setChallenge(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="e.g., I'm struggling to prioritize my product roadmap with limited resources..."
              className="w-full p-4 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-800 placeholder-gray-400"
              rows={4}
            />
            <button
              onClick={handleStartSession}
              disabled={!challenge.trim() || isLoading}
              className="mt-4 w-full py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              {isLoading ? (
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Starting session...
                </span>
              ) : (
                'Start Coaching Session →'
              )}
            </button>
          </div>

          {/* Example challenges */}
          <div className="mt-8 w-full max-w-2xl">
            <p className="text-xs text-gray-500 mb-3">Example challenges:</p>
            <div className="flex flex-wrap gap-2">
              {[
                "How do I build a product roadmap?",
                "What metrics should I track?",
                "How do I get stakeholder buy-in?",
                "When should I pivot vs persist?"
              ].map((example, idx) => (
                <button
                  key={idx}
                  onClick={() => setChallenge(example)}
                  className="text-xs px-3 py-1.5 bg-gray-100 rounded-full text-gray-600 hover:bg-gray-200 transition-colors"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center py-4 text-xs text-gray-400">
          Powered by insights from Lenny&apos;s Podcast
        </div>
      </div>
    );
  }

  // Chat interface after session starts
  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-gray-200 px-4 py-3 flex items-center justify-between bg-white">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center text-white text-lg">
            🎯
          </div>
          <div>
            <h2 className="font-semibold text-gray-800">Product Strategy Coach</h2>
            <p className="text-xs text-gray-500">Powered by Lenny&apos;s Podcast insights</p>
          </div>
        </div>
        <button
          onClick={() => {
            setSessionStarted(false);
            setMessages([]);
            setChallenge('');
            setSuggestedQuestions([]);
          }}
          className="text-sm text-gray-500 hover:text-gray-700"
        >
          New Session
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 bg-gray-50">
        {messages.map((message, idx) => (
          <ChatMessage
            key={idx}
            role={message.role}
            content={message.content}
            sources={message.sources}
          />
        ))}
        
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-100 rounded-2xl rounded-bl-md px-4 py-3">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center text-white text-sm">
                  🎯
                </div>
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Suggested questions */}
      {suggestedQuestions.length > 0 && (
        <div className="px-4 py-2 bg-white border-t border-gray-100">
          <p className="text-xs text-gray-500 mb-2">Suggested follow-ups:</p>
          <SuggestedQuestions
            questions={suggestedQuestions}
            onSelect={(q) => handleSendMessage(q)}
          />
        </div>
      )}

      {/* Input */}
      <div className="border-t border-gray-200 p-4 bg-white">
        <div className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask a follow-up question..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-800 placeholder-gray-400"
            disabled={isLoading}
          />
          <button
            onClick={() => handleSendMessage()}
            disabled={!input.trim() || isLoading}
            className="px-6 py-3 bg-blue-600 text-white rounded-xl font-medium hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
}
