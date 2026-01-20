#!/usr/bin/env python3
"""
Demo script for the Podcast RAG System

This script demonstrates how to:
1. Load podcast transcripts from the product-strategy index
2. Query the RAG system with questions about product strategy

Usage:
    # Make sure OPENAI_API_KEY is set
    export OPENAI_API_KEY=sk-...
    
    # Activate the virtual environment
    source .venv/bin/activate
    
    # Run the demo
    python demo_rag.py
"""

import os
import sys
from pathlib import Path

# Ensure the rag module can be imported
sys.path.insert(0, str(Path(__file__).parent))

from rag import PodcastRAG


def main():
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    
    print("=" * 60)
    print("Lenny's Podcast RAG System - Product Strategy")
    print("=" * 60)
    
    # Initialize RAG with persistent storage
    persist_dir = Path(__file__).parent / "chroma_db"
    rag = PodcastRAG(persist_directory=str(persist_dir))
    
    # Check if we need to load data
    stats = rag.get_collection_stats()
    
    if stats['total_chunks'] == 0:
        print("\n📚 No data loaded. Loading product strategy episodes...")
        print("-" * 60)
        
        index_file = Path(__file__).parent / "lennys-podcast-transcripts/index/product-strategy.md"
        episodes_path = Path(__file__).parent / "lennys-podcast-transcripts/episodes"
        
        if not index_file.exists():
            print(f"Error: Index file not found at {index_file}")
            sys.exit(1)
        
        load_stats = rag.load_episodes_from_index(
            index_file=str(index_file),
            episodes_base_path=str(episodes_path),
            verbose=True
        )
        
        print("-" * 60)
        print(f"✅ Loaded {load_stats['loaded']} episodes")
        print(f"📄 Total chunks: {load_stats['total_chunks']}")
        if load_stats['failed'] > 0:
            print(f"⚠️  Failed to load: {load_stats['failed']} episodes")
    else:
        print(f"\n✅ Using existing data: {stats['total_chunks']} chunks loaded")
    
    # Interactive query loop
    print("\n" + "=" * 60)
    print("💬 Ask questions about product strategy!")
    print("   Type 'quit' or 'exit' to stop")
    print("=" * 60)
    
    # Sample questions to try
    sample_questions = [
        "What makes a great product strategy?",
        "How do successful PMs prioritize their roadmap?",
        "What's the best way to measure product success?",
        "How do you build a strong product culture?",
        "What are common mistakes in product management?"
    ]
    
    print("\n📝 Sample questions you can ask:")
    for i, q in enumerate(sample_questions, 1):
        print(f"   {i}. {q}")
    
    print()
    
    while True:
        try:
            question = input("\n🔍 Your question: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            # Check if user entered a number for sample question
            if question.isdigit():
                idx = int(question) - 1
                if 0 <= idx < len(sample_questions):
                    question = sample_questions[idx]
                    print(f"   → {question}")
            
            print("\n⏳ Searching and generating answer...")
            
            result = rag.query(question, n_results=5, include_context=False)
            
            print("\n" + "-" * 60)
            print("📖 Answer:")
            print("-" * 60)
            print(result['answer'])
            
            print("\n📚 Sources:")
            for source in result['sources']:
                relevance = source['relevance_score'] * 100
                print(f"   • {source['guest']} ({relevance:.0f}% relevant)")
                print(f"     Episode: {source['episode_title'][:60]}...")
                if source['youtube_url']:
                    print(f"     Watch: {source['youtube_url']}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
