from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

# Ensure punkt is downloaded (basic tokenizer data)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def calculate_bleu_score(reference, candidate):
    """
    Calculate BLEU score between reference and candidate text.
    Uses SmoothingFunction to handle short texts/zero overlaps better.
    """
    if not reference or not candidate:
        return 0.0
    
    # BLEU expects tokenized lists
    # Reference needs to be a list of lists (multiple references supported, we have 1)
    ref_tokens = [reference.split()] 
    cand_tokens = candidate.split()
    
    # Smoothing function method 1 is standard for sentence-level BLEU
    smoothie = SmoothingFunction().method1
    
    score = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=smoothie)
    return float(score)
