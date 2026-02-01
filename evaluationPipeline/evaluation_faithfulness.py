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
        
    model = get_nli_model()
    
    # Split answer into sentences
    sentences = nltk.sent_tokenize(answer)
    if not sentences:
        return 0.0
        
    entailed_count = 0
    
    # Predict returns scores for classes. 
    # Label mapping: 0: Contradiction, 1: Entailment, 2: Neutral
    
    for sent in sentences:
        # 1. Exact Substring Match (Extractive QA is faithful by definition)
        if sent in context:
            entailed_count += 1
            print(f"Faithfulness: Substring match for '{sent[:30]}...'")
            continue
            
        # 2. NLI Model Check
        pair = (context, sent)
        scores = model.predict([pair], show_progress_bar=False)[0]
        label_idx = scores.argmax()
        
        # Consider Entailment (1) as faithful. 
        # Some strict models might classify 'paraphrase' as Neutral (2).
        # We can optionally allow Neutral if Semantic Score is high, but strictly:
        if label_idx == 1: 
            entailed_count += 1
        else:
             print(f"Faithfulness: Failed ({label_idx}) for '{sent[:30]}...'")

    return entailed_count / len(sentences)
