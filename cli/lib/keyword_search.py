import string

from nltk.stem import PorterStemmer

from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    movies = load_movies()
    query_tokens = tokenize_text(query)
    results = []
    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            results.append(movie)
        if len(results) >= limit:
            break
    return results

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    stopwords = load_stopwords()
    filtered_tokens = filter(lambda token: token not in stopwords, tokens)
    stemmer = PorterStemmer()
    stemmed_tokens = map(lambda token: stemmer.stem(token), filtered_tokens)
    return list(stemmed_tokens)

def preprocess_text(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation)).lower()
