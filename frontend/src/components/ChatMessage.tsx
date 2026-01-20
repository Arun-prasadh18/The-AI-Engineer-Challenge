'use client';

import { Source } from '@/lib/api';

interface ChatMessageProps {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp?: string;
}

export default function ChatMessage({ role, content, sources }: ChatMessageProps) {
  const isUser = role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      <div
        className={`max-w-[85%] rounded-2xl px-4 py-3 ${
          isUser
            ? 'bg-blue-600 text-white rounded-br-md'
            : 'bg-gray-100 text-gray-800 rounded-bl-md'
        }`}
      >
        {/* Avatar */}
        <div className="flex items-start gap-3">
          {!isUser && (
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center text-white text-sm font-bold flex-shrink-0">
              🎯
            </div>
          )}
          
          <div className="flex-1">
            {/* Message content */}
            <div className="whitespace-pre-wrap text-sm leading-relaxed">
              {content}
            </div>
            
            {/* Sources */}
            {sources && sources.length > 0 && (
              <div className="mt-3 pt-3 border-t border-gray-200">
                <p className="text-xs font-semibold text-gray-500 mb-2">📚 Sources:</p>
                <div className="space-y-2">
                  {sources.map((source, idx) => (
                    <a
                      key={idx}
                      href={source.youtube_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block text-xs p-2 bg-white rounded-lg hover:bg-gray-50 transition-colors border border-gray-200"
                    >
                      <div className="font-medium text-gray-700">{source.guest}</div>
                      <div className="text-gray-500 truncate">{source.episode}</div>
                      <div className="text-blue-500 mt-1 flex items-center gap-1">
                        <span>▶️</span>
                        <span>Watch clip ({source.relevance}% relevant)</span>
                      </div>
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
          
          {isUser && (
            <div className="w-8 h-8 rounded-full bg-blue-700 flex items-center justify-center text-white text-sm font-bold flex-shrink-0">
              👤
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
