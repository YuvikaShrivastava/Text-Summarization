import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from heapq import nlargest

def summarize(text, percent=0.3):
    # Split text into sentences
    sentences = sent_tokenize(text)

    # Tokenize and clean words
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    word_frequencies = defaultdict(int)

    for sentence in sentences:
        words = word_tokenize(sentence)
        for word in words:
            word = word.lower()
            if word.isalpha() and word not in stop_words:
                lemma = lemmatizer.lemmatize(word)
                word_frequencies[lemma] += 1

    # Normalize frequencies
    max_freq = max(word_frequencies.values(), default=1)
    for word in word_frequencies:
        word_frequencies[word] /= max_freq

    # Score sentences
    sentence_scores = defaultdict(int)
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_frequencies:
                sentence_scores[sentence] += word_frequencies[word]

    # Select top N sentences
    num_sentences = max(1, int(len(sentences) * percent))
    summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)

    #Return final summary
    return "\n".join(summary_sentences)

# Get user input
print("Enter your paragraph below:")
user_text = input()

# Generate and print summary
summary = summarize(user_text)
print("\n📝 Summary:\n")
print(summary)
