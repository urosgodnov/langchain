import os
import sys
import textwrap
from typing import List

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter
)
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma

# Initialize OpenAI client
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Sample document content - same as in the original code
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

# Set up a temporary file to use with LangChain loaders
def setup_document_file(content):
    with open("temp_document.md", "w") as f:
        f.write(content)
    return "temp_document.md"

# Create a Document object from text
def text_to_docs(text: str) -> List[Document]:
    return [Document(page_content=text)]

# Different text splitters for chunking methods

def chunk_by_paragraphs(docs: List[Document], chunk_size=1000, chunk_overlap=0):
    """Chunk documents by paragraphs using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(docs)

def chunk_by_headings(docs: List[Document]):
    """Chunk document using markdown headers."""
    headers_to_split_on = [
        ("#", "heading1"),
        ("##", "heading2"),
        ("###", "heading3"),
    ]
    
    # First split by headers
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    splits = []
    
    for doc in docs:
        header_splits = markdown_splitter.split_text(doc.page_content)
        splits.extend(header_splits)
    
    return splits

def chunk_by_fixed_size(docs: List[Document], chunk_size=100, chunk_overlap=20):
    """Split text into fixed-size chunks with overlap."""
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(docs)

def chunk_by_semantic_units(docs: List[Document]):
    """Attempt to split by semantic units - first by headers, then refine large chunks."""
    # First split by headers
    header_chunks = chunk_by_headings(docs)
    
    # Then refine large chunks further by paragraphs
    refined_chunks = []
    for chunk in header_chunks:
        # If chunk is small enough, keep it as is
        if len(chunk.page_content) < 500:
            refined_chunks.append(chunk)
        else:
            # Split larger chunks at paragraph level
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n", ". ", " ", ""],
                chunk_size=300,
                chunk_overlap=50,
                length_function=len
            )
            smaller_chunks = text_splitter.split_text(chunk.page_content)
            
            # Convert to Documents and preserve metadata
            for small_chunk in smaller_chunks:
                refined_chunks.append(Document(
                    page_content=small_chunk,
                    metadata=chunk.metadata
                ))
    
    return refined_chunks

# Setup retrieval and QA
def setup_retrieval_qa(chunks, model_name="gpt-4o-mini"):
    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="temp_collection",
        persist_directory=None  # In-memory only
    )
    
    # Create LLM
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    
    # Create template
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context:
    {context}
    
    Question: {question}
    """
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    # Create retrieval QA chain
    qa_chain = (
        {"context": vectorstore.as_retriever(search_kwargs={"k": 2}), "question": RunnablePassthrough()}
        | QA_CHAIN_PROMPT
        | llm
        | StrOutputParser()
    )
    
    return qa_chain, vectorstore

# Print the chunks with some formatting
def print_chunks(chunks, method_name):
    print(f"\n{'-'*80}")
    print(f"Chunking Method: {method_name}")
    print(f"Number of chunks: {len(chunks)}")
    print(f"{'-'*80}")
    for i, chunk in enumerate(chunks):
        # Truncate very long chunks for display
        display_chunk = textwrap.shorten(chunk.page_content, width=100, placeholder="...")
        print(f"Chunk {i+1}: {display_chunk}")
        
        # Print metadata if it exists and isn't empty
        if hasattr(chunk, 'metadata') and chunk.metadata:
            print(f"   Metadata: {chunk.metadata}")
        print()

# Compare different chunking methods
def run_chunking_comparison(query):
    # Convert document to LangChain Documents
    docs = text_to_docs(DOCUMENT)
    
    # Define chunking methods
    chunking_methods = [
        ("Paragraphs", chunk_by_paragraphs(docs)),
        ("Headings", chunk_by_headings(docs)),
        ("Fixed Size", chunk_by_fixed_size(docs)),
        ("Semantic Units", chunk_by_semantic_units(docs))
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
        
        # Setup retrieval QA
        qa_chain, vectorstore = setup_retrieval_qa(chunks)
        
        # Retrieve documents
        retrieved_docs = vectorstore.similarity_search(query, k=2)
        
        # Print retrieved chunks
        print("Retrieved chunks:")
        for i, doc in enumerate(retrieved_docs):
            print(f"\nResult {i+1}:")
            print(textwrap.fill(doc.page_content, width=80))
            if hasattr(doc, 'metadata') and doc.metadata:
                print(f"Metadata: {doc.metadata}")
        
        # Get answer
        answer = qa_chain.invoke(query)
        
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
    
    # Clean up any previous temp files or directories
    import shutil
    if os.path.exists("temp_document.md"):
        os.remove("temp_document.md")
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    
    # Run comparison
    run_chunking_comparison(query)
    
    # Clean up
    if os.path.exists("temp_document.md"):
        os.remove("temp_document.md")