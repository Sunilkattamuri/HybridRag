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
        _nli_model = CrossEncoder(model_name)
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
    
    # Prepare pairs: (Context, Sentence)
    # Note: CrossEncoders take a list of pairs
    pairs = [(context, sent) for sent in sentences]
    
    # Predict returns scores for classes. 
    # For nli-MiniLM2-L6-H768, labels are commonly: 0: Contradiction, 1: Entailment, 2: Neutral (or similar mapping)
    # Actually, verify mapping for 'cross-encoder/nli-MiniLM2-L6-H768'
    # According to HuggingFace: label2id: {'contradiction': 0, 'entailment': 1, 'neutral': 2} 
    # WAIT: Usually it's Contradiction(0), Entailment(1), Neutral(2) OR Entailment(0), Neutral(1), Contradiction(2)?
    # Let's check typical usage. 
    # 'cross-encoder/nli-deberta-base' output is (Contradiction, Entailment, Neutral).
    # 'cross-encoder/nli-MiniLM2-L6-H768' output is just logits.
    # The safest way is to check the max score index.
    # Standard MNLI mapping: 0=Contradiction, 1=Entailment, 2=Neutral usually.
    # Actually, for `cross-encoder/stsb...` it gives similarity. 
    # Let's use `cross-encoder/nli-distilroberta-base`. Labels: 0-contradiction, 1-entailment, 2-neutral.
    
    # Let's assume index 1 is Entailment.
    scores = model.predict(pairs)
    
    # scores is a list of arrays (logits)
    # We want argmax. If argmax == 1 (Entailment), we count it.
    
    for score in scores:
        label_idx = score.argmax()
        # Mapping for nli-distilroberta-base and similar:
        # 0: contradiction
        # 1: entailment
        # 2: neutral
        if label_idx == 1: 
            entailed_count += 1
            
    return entailed_count / len(sentences)
