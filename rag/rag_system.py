"""
RAG System for Lenny's Podcast Transcripts

This module implements a Retrieval-Augmented Generation (RAG) system for querying
podcast transcripts using OpenAI embeddings and ChromaDB for vector storage.

Chunking Strategy:
- Speaker-turn based chunking: Each speaker's turn is treated as a natural semantic unit
- Chunk merging: Adjacent speaker turns are merged if total length is under threshold
- Overlap: Maintains context by including the last sentence of the previous chunk
- Metadata preservation: Each chunk retains episode info for attribution
"""

import os
import re
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import chromadb
from chromadb.config import Settings
from openai import OpenAI


@dataclass
class TranscriptChunk:
    """Represents a chunk of transcript text with metadata."""
    text: str
    episode_title: str
    guest: str
    youtube_url: str
    chunk_index: int
    start_timestamp: Optional[str] = None
    

@dataclass
class EpisodeMetadata:
    """Metadata extracted from transcript YAML frontmatter."""
    guest: str
    title: str
    youtube_url: str
    video_id: str
    publish_date: str
    description: str = ""
    duration: str = ""
    keywords: list = field(default_factory=list)


class TranscriptParser:
    """Parses podcast transcript files with YAML frontmatter."""
    
    @staticmethod
    def parse_frontmatter(content: str) -> tuple[dict, str]:
        """
        Extract YAML frontmatter and transcript content from a markdown file.
        
        Returns:
            tuple: (metadata_dict, transcript_text)
        """
        # Split on YAML delimiters
        parts = content.split('---')
        if len(parts) >= 3:
            try:
                frontmatter = yaml.safe_load(parts[1])
                transcript = '---'.join(parts[2:]).strip()
                return frontmatter, transcript
            except yaml.YAMLError:
                return {}, content
        return {}, content
    
    @staticmethod
    def extract_speaker_turns(transcript: str) -> list[dict]:
        """
        Extract individual speaker turns from the transcript.
        
        Speaker turns are identified by the pattern: "Speaker Name (HH:MM:SS):"
        
        Returns:
            list: List of dicts with 'speaker', 'timestamp', and 'text' keys
        """
        # Pattern matches: Name (timestamp):
        pattern = r'^([A-Za-z\s\-\.]+)\s*\((\d{1,2}:\d{2}:\d{2})\):\s*'
        
        turns = []
        current_speaker = None
        current_timestamp = None
        current_text = []
        
        for line in transcript.split('\n'):
            match = re.match(pattern, line)
            if match:
                # Save previous turn if exists
                if current_speaker and current_text:
                    turns.append({
                        'speaker': current_speaker,
                        'timestamp': current_timestamp,
                        'text': ' '.join(current_text).strip()
                    })
                
                # Start new turn
                current_speaker = match.group(1).strip()
                current_timestamp = match.group(2)
                remaining_text = line[match.end():].strip()
                current_text = [remaining_text] if remaining_text else []
            elif line.strip() and current_speaker:
                # Continue current turn (skip empty lines and headers)
                if not line.startswith('#'):
                    current_text.append(line.strip())
        
        # Don't forget the last turn
        if current_speaker and current_text:
            turns.append({
                'speaker': current_speaker,
                'timestamp': current_timestamp,
                'text': ' '.join(current_text).strip()
            })
        
        return turns


class SemanticChunker:
    """
    Implements semantic chunking for podcast transcripts.
    
    Strategy:
    1. Use speaker turns as natural semantic boundaries
    2. Merge small adjacent turns to create meaningful chunks
    3. Split overly long turns while preserving sentence boundaries
    4. Add overlap between chunks for context continuity
    """
    
    def __init__(
        self,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
        overlap_sentences: int = 1
    ):
        """
        Initialize the chunker with size parameters.
        
        Args:
            min_chunk_size: Minimum characters per chunk (merge smaller turns)
            max_chunk_size: Maximum characters per chunk (split larger turns)
            overlap_sentences: Number of sentences to overlap between chunks
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_sentences = overlap_sentences
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations and split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_text(self, previous_chunk: str) -> str:
        """Get the last N sentences from previous chunk for overlap."""
        if not previous_chunk:
            return ""
        sentences = self._split_into_sentences(previous_chunk)
        overlap = sentences[-self.overlap_sentences:] if sentences else []
        return ' '.join(overlap)
    
    def chunk_transcript(
        self,
        speaker_turns: list[dict],
        metadata: EpisodeMetadata
    ) -> list[TranscriptChunk]:
        """
        Convert speaker turns into semantic chunks.
        
        Args:
            speaker_turns: List of speaker turn dictionaries
            metadata: Episode metadata for attribution
            
        Returns:
            List of TranscriptChunk objects
        """
        chunks = []
        current_text = []
        current_timestamp = None
        previous_chunk_text = ""
        
        for turn in speaker_turns:
            turn_text = f"{turn['speaker']}: {turn['text']}"
            
            if current_timestamp is None:
                current_timestamp = turn['timestamp']
            
            # Check if adding this turn would exceed max size
            combined_length = len(' '.join(current_text + [turn_text]))
            
            if combined_length > self.max_chunk_size and current_text:
                # Save current chunk and start new one
                chunk_text = ' '.join(current_text)
                
                # Add overlap from previous chunk
                if previous_chunk_text:
                    overlap = self._get_overlap_text(previous_chunk_text)
                    if overlap:
                        chunk_text = f"[Previous context: {overlap}]\n\n{chunk_text}"
                
                chunks.append(TranscriptChunk(
                    text=chunk_text,
                    episode_title=metadata.title,
                    guest=metadata.guest,
                    youtube_url=metadata.youtube_url,
                    chunk_index=len(chunks),
                    start_timestamp=current_timestamp
                ))
                
                previous_chunk_text = ' '.join(current_text)
                current_text = [turn_text]
                current_timestamp = turn['timestamp']
            else:
                current_text.append(turn_text)
        
        # Handle remaining text
        if current_text:
            chunk_text = ' '.join(current_text)
            
            # Add overlap from previous chunk
            if previous_chunk_text:
                overlap = self._get_overlap_text(previous_chunk_text)
                if overlap:
                    chunk_text = f"[Previous context: {overlap}]\n\n{chunk_text}"
            
            chunks.append(TranscriptChunk(
                text=chunk_text,
                episode_title=metadata.title,
                guest=metadata.guest,
                youtube_url=metadata.youtube_url,
                chunk_index=len(chunks),
                start_timestamp=current_timestamp
            ))
        
        return chunks


class PodcastRAG:
    """
    Main RAG system for querying podcast transcripts.
    
    Uses OpenAI embeddings and ChromaDB for vector storage and retrieval.
    """
    
    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        collection_name: str = "podcast_transcripts",
        persist_directory: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize the RAG system.
        
        Args:
            openai_api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            collection_name: Name for the ChromaDB collection
            persist_directory: Directory to persist ChromaDB data (None for in-memory)
            embedding_model: OpenAI embedding model to use
        """
        self.openai_client = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        
        # Initialize ChromaDB
        if persist_directory:
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.chroma_client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Lenny's Podcast transcripts for product strategy"}
        )
        
        # Initialize helpers
        self.parser = TranscriptParser()
        self.chunker = SemanticChunker()
    
    def _generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text using OpenAI."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def _generate_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in a single API call."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def load_episode(self, transcript_path: str) -> int:
        """
        Load a single episode transcript into the vector store.
        
        Args:
            transcript_path: Path to the transcript.md file
            
        Returns:
            Number of chunks added
        """
        with open(transcript_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse the transcript
        frontmatter, transcript_text = self.parser.parse_frontmatter(content)
        
        if not frontmatter:
            print(f"Warning: Could not parse frontmatter from {transcript_path}")
            return 0
        
        # Create metadata object
        metadata = EpisodeMetadata(
            guest=frontmatter.get('guest', 'Unknown'),
            title=frontmatter.get('title', 'Unknown'),
            youtube_url=frontmatter.get('youtube_url', ''),
            video_id=frontmatter.get('video_id', ''),
            publish_date=frontmatter.get('publish_date', ''),
            description=frontmatter.get('description', ''),
            duration=frontmatter.get('duration', ''),
            keywords=frontmatter.get('keywords', [])
        )
        
        # Extract speaker turns and create chunks
        speaker_turns = self.parser.extract_speaker_turns(transcript_text)
        chunks = self.chunker.chunk_transcript(speaker_turns, metadata)
        
        if not chunks:
            print(f"Warning: No chunks generated from {transcript_path}")
            return 0
        
        # Generate embeddings in batches
        chunk_texts = [chunk.text for chunk in chunks]
        embeddings = self._generate_embeddings_batch(chunk_texts)
        
        # Add to ChromaDB
        ids = [f"{metadata.video_id}_{chunk.chunk_index}" for chunk in chunks]
        metadatas = [
            {
                "episode_title": chunk.episode_title,
                "guest": chunk.guest,
                "youtube_url": chunk.youtube_url,
                "chunk_index": chunk.chunk_index,
                "timestamp": chunk.start_timestamp or ""
            }
            for chunk in chunks
        ]
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    def load_episodes_from_index(
        self,
        index_file: str,
        episodes_base_path: str,
        verbose: bool = True
    ) -> dict:
        """
        Load all episodes listed in an index file.
        
        Args:
            index_file: Path to the index markdown file (e.g., product-strategy.md)
            episodes_base_path: Base path to the episodes folder
            verbose: Print progress information
            
        Returns:
            dict with 'loaded', 'failed', and 'total_chunks' counts
        """
        # Parse the index file to get episode paths
        with open(index_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract episode folder names from markdown links
        # Pattern: [Name](../episodes/folder-name/transcript.md)
        pattern = r'\[.*?\]\(\.\./episodes/([^/]+)/transcript\.md\)'
        episode_folders = re.findall(pattern, content)
        
        stats = {'loaded': 0, 'failed': 0, 'total_chunks': 0, 'episodes': []}
        
        for folder in episode_folders:
            transcript_path = Path(episodes_base_path) / folder / "transcript.md"
            
            if not transcript_path.exists():
                if verbose:
                    print(f"  ✗ Not found: {folder}")
                stats['failed'] += 1
                continue
            
            try:
                chunks_added = self.load_episode(str(transcript_path))
                stats['loaded'] += 1
                stats['total_chunks'] += chunks_added
                stats['episodes'].append(folder)
                if verbose:
                    print(f"  ✓ Loaded: {folder} ({chunks_added} chunks)")
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error loading {folder}: {e}")
                stats['failed'] += 1
        
        return stats
    
    def query(
        self,
        question: str,
        n_results: int = 5,
        include_context: bool = True
    ) -> dict:
        """
        Query the RAG system with a question.
        
        Args:
            question: The question to ask
            n_results: Number of relevant chunks to retrieve
            include_context: Whether to include retrieved context in response
            
        Returns:
            dict with 'answer', 'sources', and optionally 'context'
        """
        # Generate embedding for the question
        question_embedding = self._generate_embedding(question)
        
        # Query ChromaDB for relevant chunks
        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            context_parts.append(f"[Source {i+1}: {metadata['guest']} - {metadata['episode_title']}]\n{doc}")
            sources.append({
                'guest': metadata['guest'],
                'episode_title': metadata['episode_title'],
                'youtube_url': metadata['youtube_url'],
                'timestamp': metadata['timestamp'],
                'relevance_score': 1 - distance  # Convert distance to similarity
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer using GPT
        system_prompt = """You are a helpful assistant that answers questions about product strategy 
based on insights from Lenny's Podcast interviews with product leaders.

Use the provided context from podcast transcripts to answer the question. 
Always cite which guest/episode your answer comes from.
If the context doesn't contain relevant information, say so honestly.
Be concise but thorough in your answers."""

        user_prompt = f"""Context from podcast transcripts:
{context}

Question: {question}

Please provide a comprehensive answer based on the podcast insights above."""

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        result = {
            'answer': answer,
            'sources': sources
        }
        
        if include_context:
            result['context'] = context
        
        return result
    
    def get_collection_stats(self) -> dict:
        """Get statistics about the loaded collection."""
        return {
            'collection_name': self.collection_name,
            'total_chunks': self.collection.count(),
            'embedding_model': self.embedding_model
        }


# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    # Simple test
    print("Initializing RAG system...")
    rag = PodcastRAG(persist_directory="./chroma_db")
    
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        print("\nLoading product strategy episodes...")
        stats = rag.load_episodes_from_index(
            index_file="lennys-podcast-transcripts/index/product-strategy.md",
            episodes_base_path="lennys-podcast-transcripts/episodes"
        )
        print(f"\nLoaded {stats['loaded']} episodes with {stats['total_chunks']} total chunks")
        print(f"Failed to load: {stats['failed']} episodes")
    
    elif len(sys.argv) > 1 and sys.argv[1] == "query":
        question = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "What is the best approach to product strategy?"
        print(f"\nQuerying: {question}")
        result = rag.query(question)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources:")
        for source in result['sources']:
            print(f"  - {source['guest']}: {source['episode_title']}")
    
    else:
        stats = rag.get_collection_stats()
        print(f"\nCollection stats: {stats}")
        print("\nUsage:")
        print("  python -m rag.rag_system load    # Load episodes")
        print("  python -m rag.rag_system query <question>    # Query the system")
