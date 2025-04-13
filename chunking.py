import os
import sys
import numpy as np
from scipy.spatial.distance import cosine
from openai import OpenAI
import textwrap
import markdown
import re
from bs4 import BeautifulSoup

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# Sample document content - replace with your own document or load from file
DOCUMENT = """
# Understanding Vector Embeddings

## What are Vector Embeddings?

Vector embeddings are numerical representations of data that capture semantic meaning. 
They allow machines to understand and process human language and other complex data types.

## How Do Embeddings Work?

Embeddings map items (like words, sentences, or documents) to vectors in a high-dimensional space.
Similar items are positioned closer together in this space, while dissimilar items are positioned farther apart.

### Key Properties:

1. **Dimensionality**: Usually ranges from hundreds to thousands of dimensions
2. **Similarity**: Can be measured using metrics like cosine similarity or Euclidean distance
3. **Transferability**: Pre-trained embeddings can be reused across different tasks

## Applications of Embeddings

* **Search**: Finding semantically similar documents
* **Recommendations**: Suggesting related content
* **Classification**: Categorizing text or other data
* **Clustering**: Grouping similar items
* **Question Answering**: Matching questions to relevant document sections

## Common Embedding Models

* Word2Vec
* GloVe
* BERT
* OpenAI Embeddings (text-embedding-3-small, text-embedding-3-large)
* Sentence Transformers

## Challenges in Using Embeddings

* Handling out-of-vocabulary words
* Contextual understanding
* Domain-specific applications
* Computational resources for large datasets
"""

# Chunking methods
def chunk_by_paragraphs(text, min_size=1):
    """Split text by paragraph breaks, ensuring each chunk has at least min_size paragraphs."""
    paragraphs = [p for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        current_chunk.append(paragraph)
        current_size += 1
        
        if current_size >= min_size:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    # Add any remaining paragraphs
    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))
    
    return chunks

def chunk_by_headings(text):
    """Split text by markdown headings."""
    # Convert markdown to HTML for easier processing
    html = markdown.markdown(text)
    soup = BeautifulSoup(html, 'html.parser')
    
    chunks = []
    current_section = ""
    current_heading = None
    
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol']):
        if element.name.startswith('h'):
            # If we have content from a previous section, add it as a chunk
            if current_heading and current_section:
                chunks.append(f"{current_heading}\n\n{current_section.strip()}")
            
            # Start a new section
            current_heading = element.get_text()
            current_section = ""
        else:
            # Add content to the current section
            content = element.get_text()
            if element.name == 'p':
                current_section += content + "\n\n"
            else:  # list items
                current_section += content + "\n"
    
    # Add the last section
    if current_heading and current_section:
        chunks.append(f"{current_heading}\n\n{current_section.strip()}")
    
    return chunks

def chunk_by_fixed_size(text, chunk_size=200, overlap=50):
    """Split text into fixed-size chunks with overlap."""
    words = text.split()
    
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        # Stop if we've reached or exceeded the end of the text
        if i + chunk_size >= len(words):
            break
    
    return chunks

def chunk_by_semantic_units(text):
    """Split text by semantic units (sections with complete thoughts)."""
    # First split by headings
    heading_chunks = chunk_by_headings(text)
    
    # Then refine by looking for semantic boundaries within larger sections
    refined_chunks = []
    for chunk in heading_chunks:
        # If chunk is small enough, keep it as is
        if len(chunk.split()) < 100:  
            refined_chunks.append(chunk)
        else:
            # Split larger chunks at natural semantic boundaries
            # Look for patterns like bullet points, numbered lists, or paragraph breaks
            sub_chunks = re.split(r'\n\s*\n|\n\s*•|\n\s*\d+\.', chunk)
            
            # Recombine the heading with each sub-chunk
            heading = chunk.split('\n\n')[0] if '\n\n' in chunk else ""
            
            for sub in sub_chunks:
                if sub.strip():
                    if heading and not sub.startswith(heading):
                        refined_chunks.append(f"{heading}\n\n{sub.strip()}")
                    else:
                        refined_chunks.append(sub.strip())
    
    return refined_chunks

# Get embeddings for text
def get_embedding(text, model="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# Calculate similarity between query and chunk
def calculate_similarity(query_embedding, chunk_embedding):
    return 1 - cosine(query_embedding, chunk_embedding)

# Simple retrieval function
def retrieve_chunks(query, chunks, top_k=3):
    query_embedding = get_embedding(query)
    if not query_embedding:
        return []
    
    # Get embeddings for each chunk
    chunk_embeddings = [(chunk, get_embedding(chunk)) for chunk in chunks]
    chunk_embeddings = [(chunk, emb) for chunk, emb in chunk_embeddings if emb]
    
    # Calculate similarities
    similarities = [
        (chunk, calculate_similarity(query_embedding, chunk_emb))
        for chunk, chunk_emb in chunk_embeddings
    ]
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top K results
    return similarities[:top_k]

# Simple QA function
def answer_question(query, retrieved_chunks):
    if not retrieved_chunks:
        return "No relevant information found."
    
    # Format context from retrieved chunks
    context = "\n\n".join([chunk for chunk, _ in retrieved_chunks])
    
    try:
        # Call OpenAI to answer based on retrieved chunks
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the question based only on the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return "Error generating answer."

# Print the chunks with some formatting
def print_chunks(chunks, method_name):
    print(f"\n{'-'*80}")
    print(f"Chunking Method: {method_name}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"{'-'*80}")
    for i, chunk in enumerate(chunks):
        # Truncate very long chunks for display
        display_chunk = textwrap.shorten(chunk, width=100, placeholder="...") if len(chunk) > 120 else chunk
        print(f"Chunk {i+1}: {display_chunk}\n")

# Compare different chunking methods
def run_chunking_comparison(query):
    chunking_methods = [
        ("Paragraphs", chunk_by_paragraphs(DOCUMENT, min_size=1)),
        ("Headings", chunk_by_headings(DOCUMENT)),
        ("Fixed Size", chunk_by_fixed_size(DOCUMENT, chunk_size=100, overlap=20)),
        ("Semantic Units", chunk_by_semantic_units(DOCUMENT))
    ]
    
    # Print chunking results
    for method_name, chunks in chunking_methods:
        print_chunks(chunks, method_name)
    
    print("\n" + "="*100)
    print(f"QUERY: {query}")
    print("="*100)
    
    # Compare retrieval and QA for each method
    for method_name, chunks in chunking_methods:
        print(f"\n{'-'*80}")
        print(f"Results with {method_name} chunking:")
        print(f"{'-'*80}")
        
        # Retrieve chunks
        retrieved = retrieve_chunks(query, chunks, top_k=2)
        
        # Print retrieved chunks
        print("Retrieved chunks:")
        for i, (chunk, similarity) in enumerate(retrieved):
            print(f"\nResult {i+1} (similarity: {similarity:.4f}):")
            print(textwrap.fill(chunk, width=80))
        
        # Answer based on retrieved chunks
        answer = answer_question(query, retrieved)
        print("\nAnswer:")
        print(textwrap.fill(answer, width=80))
        print(f"{'-'*80}")

if __name__ == "__main__":
    # If a query is provided as a command-line argument, use it
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Otherwise, use a default query
        query = "What are some applications of vector embeddings?"
    
    run_chunking_comparison(query)