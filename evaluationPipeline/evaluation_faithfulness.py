from sentence_transformers import CrossEncoder
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

_nli_model = None

def get_nli_model():
    global _nli_model
    if _nli_model is None:
        # We use a lightweight Cross-Encoder trained on NLI data
        # 'cross-encoder/nli-MiniLM2-L6-H768' is fast and good for this
        model_name = 'cross-encoder/nli-MiniLM2-L6-H768'
        print(f"Loading Faithfulness Model: {model_name}")
        _nli_model = CrossEncoder(model_name, max_length=512)
    return _nli_model

def calculate_faithfulness(context, answer):
    """
    Calculates faithfulness score checking if the answer is entailed by the context.
    Uses an NLI model.
    Score = (Sentences Entailed by Context) / (Total Sentences)
    """
    if not context or not answer:
        return 0.0
        
    # 1. Handle "NOT_FOUND" case
    # If the system correctly identified that the answer is missing, it is faithful to the instruction.
    if "NOT_FOUND_IN_CONTEXT" in answer:
        return 1.0

    model = get_nli_model()
    
    # Split answer into sentences
    sentences = nltk.sent_tokenize(answer)
    if not sentences:
        return 0.0
        
    # Split context into chunks (segments) to avoid 512 token limit of NLI model
    # The RAG pipeline joins chunks with "\n\n", so we split by that.
    context_chunks = context.split("\n\n")
    
    entailed_count = 0
    
    # Predict returns scores for classes. 
    # Label mapping: 0: Contradiction, 1: Entailment, 2: Neutral
    
    for sent in sentences:
        # Check if this sentence is supported by ANY of the context chunks
        is_supported = False
        
        # Optimization: First check exact substring match in the full context
        if sent in context:
            entailed_count += 1
            print(f"Faithfulness: Substring match for '{sent[:30]}...'")
            continue

        # NLI Check against each chunk
        for chunk in context_chunks:
            if not chunk.strip():
                continue
                
            pair = (chunk, sent)
            scores = model.predict([pair], show_progress_bar=False)[0]
            label_idx = scores.argmax()
            
            # If ANY chunk entails the sentence (Label 1), it's valid.
            if label_idx == 1:
                is_supported = True
                break
        
        if is_supported:
            entailed_count += 1
        else:
             print(f"Faithfulness: Failed to find entailment for '{sent[:30]}...'")

    return entailed_count / len(sentences)
