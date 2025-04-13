import os
import numpy as np
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from openai import OpenAI

# Get API key from environment variable
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Initialize the client
client = OpenAI(api_key=api_key)

# The two statements to compare
statement1 = "The children like to play outside."
statement2 = "The child likes to play outside."

# Get embeddings from text-embedding-3-small model
small_response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[statement1, statement2]
)

# Get embeddings from text-embedding-3-large model
large_response = client.embeddings.create(
    model="text-embedding-3-large",
    input=[statement1, statement2]
)

# Extract the embedding vectors
small_embedding1 = small_response.data[0].embedding
small_embedding2 = small_response.data[1].embedding
large_embedding1 = large_response.data[0].embedding
large_embedding2 = large_response.data[1].embedding

# Calculate similarities (1 - cosine distance)
small_similarity = 1 - cosine(small_embedding1, small_embedding2)
large_similarity = 1 - cosine(large_embedding1, large_embedding2)


# Print results
print("COMPARISON BETWEEN MODELS:")
print(f"Statement 1: {statement1}")
print(f"Statement 2: {statement2}")
print("\ntext-embedding-3-small model:")
print(f"Similarity score: {small_similarity:.4f}")
print(f"Embedding dimensions: {len(small_embedding1)}")

print("\ntext-embedding-3-large model:")
print(f"Similarity score: {large_similarity:.4f}")
print(f"Embedding dimensions: {len(large_embedding1)}")

# Compare the difference in similarity assessments
print("\nDIFFERENCE ANALYSIS:")
print(f"Similarity difference (large - small): {large_similarity - small_similarity:.4f}")
print(f"The large model {'finds the statements more similar' if large_similarity > small_similarity else 'finds the statements less similar'} than the small model")

# Optional visualization of embedding comparisons (first 10 dimensions)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(range(10), small_embedding1[:10], alpha=0.5, label='Statement 1')
plt.bar(range(10), small_embedding2[:10], alpha=0.5, label='Statement 2')
plt.title('text-embedding-3-small (First 10 dims)')
plt.legend()

plt.subplot(1, 2, 2)
plt.bar(range(10), large_embedding1[:10], alpha=0.5, label='Statement 1')
plt.bar(range(10), large_embedding2[:10], alpha=0.5, label='Statement 2')
plt.title('text-embedding-3-large (First 10 dims)')
plt.legend()

plt.tight_layout()
plt.savefig('embedding_comparison.png')
print("Visualization saved as 'embedding_comparison.png'")