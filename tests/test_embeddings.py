"""Tests for embeddings and text utilities."""
from zeroh.embeddings import TfidfVectorizer, cosine_similarity
from zeroh.text import split_sentences, tokenize


def test_tokenize_removes_stopwords():
    assert tokenize("The cat sat on the mat") == ["cat", "sat", "mat"]


def test_tokenize_keep_stopwords():
    assert "the" in tokenize("the cat", remove_stopwords=False)


def test_split_sentences():
    sents = split_sentences("Hello world. How are you? Fine!")
    assert sents == ["Hello world.", "How are you?", "Fine!"]


def test_cosine_identical_is_one():
    v = TfidfVectorizer().fit(["paris is the capital of france"])
    vec = v.transform("paris is the capital of france")
    assert abs(cosine_similarity(vec, vec) - 1.0) < 1e-9


def test_cosine_disjoint_is_zero():
    v = TfidfVectorizer().fit(["alpha beta", "gamma delta"])
    a = v.transform("alpha beta")
    b = v.transform("gamma delta")
    assert cosine_similarity(a, b) == 0.0


def test_related_more_similar_than_unrelated():
    corpus = [
        "the capital of france is paris",
        "the eiffel tower is in paris",
        "bananas are a yellow fruit",
    ]
    v = TfidfVectorizer().fit(corpus)
    q = v.transform("what is the capital of france")
    related = cosine_similarity(q, v.transform(corpus[0]))
    unrelated = cosine_similarity(q, v.transform(corpus[2]))
    assert related > unrelated
