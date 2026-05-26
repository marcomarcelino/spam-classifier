import pickle
import pytest

def test_model_loads():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    assert model is not None

def test_vectorizer_loads():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    assert vectorizer is not None

def test_prediction():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    spam_msg = ["Congratulations! You won a FREE iPhone! Click now!"]
    ham_msg = ["Hey, are we still meeting for lunch tomorrow?"]
    
    spam_vec = vectorizer.transform(spam_msg)
    ham_vec = vectorizer.transform(ham_msg)
    
    assert model.predict(spam_vec)[0] == 1  # spam
    assert model.predict(ham_vec)[0] == 0   # ham