#!/usr/bin/env python3
"""
Review Chunker Script
Splits review_highlights.txt into sentence-level chunks for Small-to-Big retrieval.

SOTA Reference: LlamaIndex Parent-Child Retrieval, RAPTOR (Sarthi et al., 2024)
"""
import json
import re
from pathlib import Path
from typing import List, Dict

# Simple sentence splitter (no external dependency)
def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    # Handle common abbreviations
    text = re.sub(r'(Mr|Mrs|Dr|Ms|Prof|Jr|Sr)\.', r'\1<DOT>', text)
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Restore abbreviations
    sentences = [s.replace('<DOT>', '.') for s in sentences]
    # Filter empty and very short sentences
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def chunk_reviews(input_path: str, output_path: str, min_chunk_len: int = 50, max_chunk_len: int = 300):
    """
    Read review_highlights.txt and output sentence-level chunks with parent ISBN.
    
    Format of input: "ISBN review_text" per line
    Format of output: JSONL with {"text": "...", "parent_isbn": "..."}
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        print(f"Error: {input_path} not found.")
        return
    
    chunks = []
    total_reviews = 0
    
    print(f"Reading reviews from {input_path}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Parse: First token is ISBN, rest is review
            parts = line.split(' ', 1)
            if len(parts) < 2:
                continue
            
            isbn = parts[0].strip()
            review = parts[1].strip()
            total_reviews += 1
            
            # Split into sentences
            sentences = split_sentences(review)
            
            # Create chunks (may combine very short sentences)
            current_chunk = ""
            for sent in sentences:
                if len(current_chunk) + len(sent) < max_chunk_len:
                    current_chunk += " " + sent if current_chunk else sent
                else:
                    # Save current chunk if long enough
                    if len(current_chunk) >= min_chunk_len:
                        chunks.append({
                            "text": current_chunk.strip(),
                            "parent_isbn": isbn
                        })
                    current_chunk = sent
            
            # Don't forget the last chunk
            if len(current_chunk) >= min_chunk_len:
                chunks.append({
                    "text": current_chunk.strip(),
                    "parent_isbn": isbn
                })
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
    
    print(f"Processed {total_reviews} reviews -> {len(chunks)} chunks")
    print(f"Output written to {output_path}")
    
    # Show sample
    print("\n--- Sample Chunks ---")
    for c in chunks[:3]:
        print(f"[{c['parent_isbn']}] {c['text'][:80]}...")


if __name__ == "__main__":
    chunk_reviews(
        input_path="data/review_highlights.txt",
        output_path="data/review_chunks.jsonl"
    )
