# Product Strategy Coach - Frontend

A modern Next.js frontend for the Product Strategy Coach application, providing an interactive coaching interface powered by insights from Lenny's Podcast.

## Features

- 🎯 **Interactive Coaching Sessions** - Start with your product challenge and get guided coaching
- 💬 **Conversational Interface** - Natural chat-based interaction with follow-up suggestions
- 📚 **Source Attribution** - Links to relevant podcast episodes with timestamps
- 📱 **Responsive Design** - Works on desktop and mobile devices

## Tech Stack

- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first styling
- **React Hooks** - State management

## Getting Started

### Prerequisites

- Node.js 18+ (or use nvm: `nvm use 20`)
- Backend API running on `http://localhost:8000`

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The app will be available at [http://localhost:3000](http://localhost:3000).

### Environment Variables

Create a `.env.local` file (already provided):

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

For production, update this to your deployed API URL.

## Running the Full Stack

1. **Start the Backend** (from project root):
   ```bash
   source .venv/bin/activate
   uvicorn api.index:app --reload
   ```

2. **Start the Frontend** (from this folder):
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000)

## Project Structure

```
src/
├── app/
│   ├── globals.css      # Global styles
│   ├── layout.tsx       # Root layout
│   └── page.tsx         # Home page
├── components/
│   ├── ChatMessage.tsx      # Chat bubble component
│   ├── CoachingChat.tsx     # Main chat interface
│   └── SuggestedQuestions.tsx  # Follow-up suggestions
└── lib/
    └── api.ts           # API client functions
```

## Deployment

This app is designed to be deployed on Vercel:

```bash
npm install -g vercel
vercel
```

Remember to set `NEXT_PUBLIC_API_URL` in your Vercel environment variables.
