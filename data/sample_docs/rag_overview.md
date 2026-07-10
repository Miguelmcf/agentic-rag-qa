# Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is a technique that improves the responses
of large language models by grounding them in external knowledge. Instead of
relying only on what a model memorized during training, a RAG system retrieves
relevant documents at query time and passes them to the model as context.

## How it works

A typical RAG pipeline has three stages. First, documents are split into chunks
and converted into vector embeddings. Second, those embeddings are stored in a
vector database that supports fast similarity search. Third, at query time the
user's question is embedded, the most similar chunks are retrieved, and the
language model generates an answer grounded in that retrieved context.

## Why it helps

RAG reduces hallucinations because the model answers from real source material,
and it lets systems stay up to date without retraining the underlying model.
Citations can be attached to each answer so users can verify the source.
