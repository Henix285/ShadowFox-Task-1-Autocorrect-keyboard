from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from spellchecker import SpellChecker
from collections import defaultdict, Counter
import re
import random

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize spell checker with English dictionary
spell = SpellChecker()

# Add custom words to the dictionary to avoid false positives
CUSTOM_WORDS = [
    "hari", "supriya", "artificial", "intelligence",
    "autocorrect", "prediction", "keyboard", "nlp",
    "ngrams", "recurrent", "neural", "networks",
    "textflow", "api", "fastapi", "python"
]
spell.word_frequency.load_words(CUSTOM_WORDS)

# ---------- Enhanced Next Word Corpus ----------
CORPUS = """
the quick brown fox jumps over the lazy dog
i am going to the store today
can you help me with this task
what time is the meeting scheduled for
how are you doing today my friend
i think we should start the project now
the weather is really nice today
please send me the document by tomorrow
let me know if you need any help
thank you very much for your assistance
i would like to schedule a meeting
the system is working as expected now
we need to improve the user experience
artificial intelligence is transforming the world
machine learning helps us make better decisions
deep learning models are very powerful
natural language processing is fascinating
the project deadline is next week
i will send you the details soon
can we discuss this in the meeting
the team is working on the solution
this is a great opportunity for us
we should focus on the main goal
please review the document and share feedback
i hope you have a wonderful day
the application is now ready for testing
let me check the status of the request
we can start working on this tomorrow
the results look very promising today
i really appreciate your hard work
develop an autocorrect keyboard system that anticipates the next word
in a sentence by leveraging the contextual information provided by
preceding words this task involves the implementation of ngrams or
recurrent neural networks to enhance predictive capabilities the goal
is to create an intuitive keyboard that improves user experience by
accurately anticipating and suggesting the next word thereby facilitating
efficient and errorfree text input real time text prediction system
the system should provide accurate suggestions based on context
machine learning algorithms can improve prediction accuracy over time
natural language processing techniques help understand user intent
the keyboard learns from user typing patterns and adapts accordingly
"""

def tokenize(text):
    """Extract words from text"""
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())

# Build n-gram models (bigram, trigram, 4-gram)
bigram = defaultdict(Counter)
trigram = defaultdict(Counter)
fourgram = defaultdict(Counter)

tokens = tokenize(CORPUS)

# Build bigrams
for i in range(len(tokens) - 1):
    bigram[tokens[i]][tokens[i + 1]] += 1

# Build trigrams
for i in range(len(tokens) - 2):
    trigram[(tokens[i], tokens[i + 1])][tokens[i + 2]] += 1

# Build 4-grams
for i in range(len(tokens) - 3):
    fourgram[(tokens[i], tokens[i + 1], tokens[i + 2])][tokens[i + 3]] += 1

COMMON_FALLBACK = [
    "the", "and", "to", "of", "in", "that", "is", "for", 
    "with", "a", "on", "at", "from", "by", "about", 
    "as", "into", "like", "through", "after", "over",
    "can", "will", "should", "would", "could", "may",
    "i", "you", "we", "they", "it", "this", "these"
]

# ---------- API Models ----------
class TextIn(BaseModel):
    text: str

class WordIn(BaseModel):
    word: str

# ---------- Next Word Prediction ----------
def predict_next(words):
    """Predict next word using 4-gram -> trigram -> bigram -> fallback"""
    if not words:
        return random.choice(COMMON_FALLBACK)
    
    # Try 4-gram first (most context)
    if len(words) >= 3:
        key = (words[-3], words[-2], words[-1])
        if key in fourgram and fourgram[key]:
            return fourgram[key].most_common(1)[0][0]
    
    # Try trigram
    if len(words) >= 2:
        key = (words[-2], words[-1])
        if key in trigram and trigram[key]:
            return trigram[key].most_common(1)[0][0]
    
    # Fall back to bigram
    if len(words) >= 1 and words[-1] in bigram and bigram[words[-1]]:
        return bigram[words[-1]].most_common(1)[0][0]
    
    # Ultimate fallback - return most common word
    return random.choice(COMMON_FALLBACK)

def predict_next_multiple(words, n=3):
    """Predict next N words using n-gram models"""
    predictions = []
    current_words = words.copy()
    
    for _ in range(n):
        next_word = predict_next(current_words)
        predictions.append(next_word)
        current_words.append(next_word)
    
    return predictions

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {
        "message": "TextFlow Assist API",
        "version": "2.0",
        "endpoints": ["/live", "/correct-word", "/predict-multiple", "/health"]
    }

@app.post("/live")
def live_check(data: TextIn):
    """
    Real-time spell checking and next word prediction
    """
    text = data.text.strip()
    
    if not text:
        return {
            "mistakes": [],
            "next_word": "",
            "next_words": []
        }
    
    # Extract words
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    
    # Find spelling mistakes (words longer than 2 characters)
    mistakes = []
    for word in words:
        if len(word) > 2:
            word_lower = word.lower()
            # Check if word is misspelled
            if word_lower not in spell:
                mistakes.append(word_lower)
    
    # Get unique mistakes
    unique_mistakes = list(set(mistakes))
    
    # Predict next word(s)
    words_lower = [w.lower() for w in words]
    next_word = predict_next(words_lower)
    next_words = predict_next_multiple(words_lower, n=3)
    
    return {
        "mistakes": unique_mistakes,
        "next_word": next_word,
        "next_words": next_words  # Returns 3 predicted words
    }

@app.post("/predict-multiple")
def predict_multiple_words(data: TextIn):
    """
    Predict next 2-3 words based on input text
    """
    text = data.text.strip()
    
    if not text:
        return {"predictions": []}
    
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z]+\b", text)]
    predictions = predict_next_multiple(words, n=3)
    
    return {
        "input": text,
        "predictions": predictions,
        "prediction_sentence": " ".join(predictions)
    }

@app.post("/correct-word")
def correct_word(data: WordIn):
    """
    Get correction suggestion for a misspelled word using pyspellchecker
    """
    word = data.word.lower().strip()
    
    # If word is empty, return it as is
    if not word:
        return {"corrected": word}
    
    # Check if word is already correct
    if word in spell:
        return {"corrected": word}
    
    # Get the correction from pyspellchecker
    corrected = spell.correction(word)
    
    # If no correction found, return original word
    if corrected is None:
        corrected = word
    
    # Get additional candidate suggestions
    candidates = spell.candidates(word)
    suggestions = list(candidates)[:5] if candidates else [corrected]
    
    print(f"Correcting: '{word}' -> '{corrected}' (candidates: {suggestions})")
    
    return {
        "original": word,
        "corrected": corrected,
        "suggestions": suggestions
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "TextFlow Assist API",
        "spell_checker": "pyspellchecker",
        "dictionary_size": len(spell),
        "ngram_models": {
            "bigram": len(bigram),
            "trigram": len(trigram),
            "fourgram": len(fourgram)
        }
    }

# ---------- Run Server ----------
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Starting TextFlow Assist API...")
    print("📍 Server running at: http://127.0.0.1:8000")
    print("📖 API docs available at: http://127.0.0.1:8000/docs")
    print("🔤 Using pyspellchecker for spell correction")
    print("🔮 N-gram models: Bigram, Trigram, 4-gram")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)