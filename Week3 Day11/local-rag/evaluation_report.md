# Evaluation Report — Local RAG Pipeline

## Executive Summary
The local RAG pipeline successfully answers questions using retrieved document context. Generation quality is strong, but retrieval performance requires improvement. The system shows moderate resistance to prompt injection attacks.

---

# Retrieval Quality

Contextual Relevancy: 0.463  
Contextual Recall: 0.400  
Contextual Precision: 0.333  

Interpretation:

The retriever often returns partially relevant chunks and misses some important information. This indicates the chunking strategy or embedding model could be improved.

Root causes:

• Small embedding model  
• Limited dataset  
• top-k retrieval not optimized

---

# Generation Quality

Faithfulness: 0.955  
Answer Relevancy: 0.889  

Interpretation:

The model generally answers correctly and stays grounded in retrieved context.

Hallucination rate is low.

---

# Local vs Cloud Comparison

Cloud evaluation could not be completed due to Azure authentication issues.

However the local model shows acceptable performance for small internal datasets.

---

# Security Assessment

Prompt Extraction: Partially Resistant  
Document Injection: Resistant  
Jailbreak Attempts: Partially Resistant

The model did not expose system prompts but could still be influenced by adversarial phrasing.

---

# Recommendations

1. Improve retrieval using better embeddings
2. Increase chunk overlap
3. Add prompt injection defenses
4. Implement query validation
5. Consider hybrid search