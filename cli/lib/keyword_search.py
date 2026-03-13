import string

from nltk.stem import PorterStemmer

from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    INDEX_PATH,
    DOCMAP_PATH,
    load_movies,
    load_stopwords,
    ensure_cache_dir,
    save_to_cache,
)

class InvertedIndex:
    def __init__(self) -> None:
        self.__index: dict[str,set[int]] = {}
        self.__docmap: dict[int, dict] = {}

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            id = movie["id"]
            text = f"{movie["title"]} {movie["description"]}"
            self.__add_document(id, text)
            self.__docmap[id] = movie

    def save(self) -> None:
        ensure_cache_dir()
        save_to_cache(INDEX_PATH, self.__index)
        save_to_cache(DOCMAP_PATH, self.__docmap)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.__index.get(term.lower(), set())
        return sorted(list(doc_ids))

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            if token in self.__index:
                self.__index[token].add(doc_id)
            else:
                self.__index[token] = set([doc_id])

def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()
    docs = index.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")

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
