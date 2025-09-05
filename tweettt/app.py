import os, re, string, zipfile, time, warnings, joblib
import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
from wordcloud import WordCloud

import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

# Sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

# Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D

# Transformers
from transformers import DistilBertTokenizerFast, TFDistilBertForSequenceClassification

# FastText
from gensim.models import FastText

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

warnings.filterwarnings("ignore")

# -----------------------------
# NLTK resources
# -----------------------------
resources = ['punkt','stopwords','wordnet','averaged_perceptron_tagger']
for res in resources:
    try:
        nltk.data.find(f'tokenizers/{res}')
    except LookupError:
        nltk.download(res, quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#[A-Za-z0-9_]+")
EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]", flags=re.UNICODE)

# -----------------------------
# Text Cleaner
# -----------------------------
class TextCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, lowercase=True, rm_urls=True, rm_mentions=True, rm_hashtags=True,
                 rm_numbers=False, rm_punct=True, use_stopwords=True, do_lemma=True, do_stem=False):
        self.lowercase = lowercase
        self.rm_urls = rm_urls
        self.rm_mentions = rm_mentions
        self.rm_hashtags = rm_hashtags
        self.rm_numbers = rm_numbers
        self.rm_punct = rm_punct
        self.use_stopwords = use_stopwords
        self.do_lemma = do_lemma
        self.do_stem = do_stem

    def _clean(self, text: str) -> str:
        if not isinstance(text, str): text = str(text)
        t = text.lower() if self.lowercase else text
        t = URL_RE.sub(' ', t) if self.rm_urls else t
        t = MENTION_RE.sub(' ', t) if self.rm_mentions else t
        t = HASHTAG_RE.sub(lambda m: m.group(0).replace('#', ' '), t) if self.rm_hashtags else t
        t = EMOJI_RE.sub(' ', t)
        if self.rm_numbers: t = re.sub(r"\d+", " ", t)
        if self.rm_punct: t = t.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(t)
        tokens = [tok for tok in tokens if len(tok) > 2]
        if self.use_stopwords: tokens = [tok for tok in tokens if tok not in STOPWORDS]
        if self.do_lemma: tokens = [LEMMATIZER.lemmatize(tok) for tok in tokens]
        if self.do_stem: tokens = [STEMMER.stem(tok) for tok in tokens]
        return ' '.join(tokens)

    def fit(self, X, y=None): return self
    def transform(self, X): return [self._clean(x) for x in X]

# -----------------------------
# Load Dataset
# -----------------------------
def load_sentiment140(csv_path_or_zip: str, sample_n: int=None):
    if not os.path.exists(csv_path_or_zip): raise FileNotFoundError(f"Path not found: {csv_path_or_zip}")
    if csv_path_or_zip.endswith('.zip'):
        with zipfile.ZipFile(csv_path_or_zip) as z:
            inner = [f for f in z.namelist() if f.endswith('.csv')][0]
            with z.open(inner) as f:
                df = pd.read_csv(f, encoding='latin-1', header=None)
    else:
        df = pd.read_csv(csv_path_or_zip, encoding='latin-1', header=None)
    df = df[[0,5]]
    df.columns = ['label_raw','text']
    df['label'] = df['label_raw'].map({0:0,4:1})
    df = df.dropna(subset=['text'])
    if sample_n and sample_n < len(df): df = df.sample(sample_n, random_state=42)
    return df[['text','label']].reset_index(drop=True)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Tweet Sentiment Analysis", layout="wide")
st.title("📊 Tweet Sentiment Analysis")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    data_source = st.radio("Dataset source", ["Upload CSV/ZIP","Local path"])
    sample_n = st.number_input("Sample N rows", min_value=0, value=50000, step=10000)
    # Preprocessing
    lowercase = st.checkbox("Lowercase", True)
    rm_urls = st.checkbox("Remove URLs", True)
    rm_mentions = st.checkbox("Remove mentions", True)
    rm_hashtags = st.checkbox("Remove hashtags", True)
    rm_numbers = st.checkbox("Remove numbers", False)
    rm_punct = st.checkbox("Remove punctuation", True)
    use_stop = st.checkbox("Remove stopwords", True)
    do_lemma = st.checkbox("Lemmatize", True)
    do_stem = st.checkbox("Stem", False)
    cleaner_kwargs = dict(lowercase=lowercase, rm_urls=rm_urls, rm_mentions=rm_mentions,
                          rm_hashtags=rm_hashtags, rm_numbers=rm_numbers, rm_punct=rm_punct,
                          use_stopwords=use_stop, do_lemma=do_lemma, do_stem=do_stem)

    st.subheader("Select Model")
    model_options = ["LogisticRegression","LinearSVC","XGBoost","MLP","LSTM","CNN","DistilBERT","FastText"]
    model_key = st.selectbox("Model", model_options)

# File input
uploaded = None
local_path = None
if data_source == "Upload CSV/ZIP": uploaded = st.file_uploader("Upload CSV/ZIP", type=["csv","zip"])
else: local_path = st.text_input("Local path")

# Create saved_models folder if not exists
if not os.path.exists("saved_models"): os.makedirs("saved_models")
model_path = os.path.join("saved_models", f"{model_key}.pkl")

# Load dataset
if st.button("Load dataset"):
    try:
        if uploaded:
            tmp_path = os.path.join(".", uploaded.name)
            open(tmp_path,'wb').write(uploaded.getvalue())
            df = load_sentiment140(tmp_path, sample_n if sample_n>0 else None)
        elif local_path:
            df = load_sentiment140(local_path, sample_n if sample_n>0 else None)
        else:
            st.error("Upload file or enter path")
            st.stop()
        st.session_state['df'] = df
        st.success(f"Loaded {len(df):,} rows.")
    except Exception as e:
        st.exception(e)

# -----------------------------
# Training / Evaluation
# -----------------------------


if 'df' in st.session_state:

    df = st.session_state['df']


    st.subheader("Graphical Analysis")

    # Compter les tweets positifs et négatifs
    tweet_counts = df['label'].value_counts().rename({0: 'Negative', 1: 'Positive'})

    # Afficher l’histogramme avec Streamlit
    st.bar_chart(tweet_counts)


    if st.button("Train / Evaluate"):
        X = df['text'].tolist()
        y = df['label'].values

        cleaner = TextCleaner(**cleaner_kwargs)
        X_clean = cleaner.transform(X)

        # Load saved model if exists
        if os.path.exists(model_path):
            st.info("Loading saved model...")
            model_data = joblib.load(model_path)
        else:
            st.info("Training new model...")

        # -----------------------------
        # Model selection
        # -----------------------------
        if model_key in ["LogisticRegression","LinearSVC","XGBoost","MLP"]:
            vec = TfidfVectorizer()
            X_vec = vec.fit_transform(X_clean)
            if model_key=="LogisticRegression": model = LogisticRegression(max_iter=200)
            elif model_key=="LinearSVC": model = LinearSVC()
            elif model_key=="XGBoost" and XGB_AVAILABLE: model = XGBClassifier()
            elif model_key=="MLP": model = MLPClassifier(max_iter=200)
            model.fit(X_vec, y)
            joblib.dump((model, vec), model_path)
            y_pred = model.predict(X_vec)

        elif model_key in ["LSTM","CNN"]:
            tokenizer = Tokenizer(num_words=50000)
            tokenizer.fit_on_texts(X_clean)
            X_seq = pad_sequences(tokenizer.texts_to_sequences(X_clean), maxlen=50)
            if model_key=="LSTM":
                model = Sequential([Embedding(50000,64,input_length=50), LSTM(64), Dense(1,activation='sigmoid')])
            else:
                model = Sequential([Embedding(50000,64,input_length=50), Conv1D(64,5,activation='relu'), GlobalMaxPooling1D(), Dense(1,activation='sigmoid')])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            model.fit(X_seq, y, epochs=3, batch_size=64, validation_split=0.1)
            joblib.dump((model, tokenizer), model_path)
            y_pred = (model.predict(X_seq) > 0.5).astype(int)

        elif model_key=="DistilBERT":
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
            X_enc = tokenizer(X_clean, truncation=True, padding=True, return_tensors='tf')
            model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
            optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
            model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
            y_tf = tf.convert_to_tensor(y)
            model.fit(X_enc.data, y_tf, epochs=1, batch_size=8)
            joblib.dump((model, tokenizer), model_path)
            y_pred = np.argmax(model.predict(X_enc.data).logits, axis=1)

        elif model_key=="FastText":
            tokenized = [x.split() for x in X_clean]
            ft_model = FastText(sentences=tokenized, vector_size=100, window=5, min_count=2, epochs=5)
            X_emb = np.array([np.mean([ft_model.wv[w] for w in doc if w in ft_model.wv] or [np.zeros(100)], axis=0) for doc in tokenized])
            model = LogisticRegression(max_iter=200)
            model.fit(X_emb, y)
            joblib.dump((model, ft_model), model_path)
            y_pred = model.predict(X_emb)

        # -----------------------------
        # Evaluation
        # -----------------------------
        st.subheader("Evaluation")
        st.text(classification_report(y, y_pred))
        st.write("Confusion Matrix", confusion_matrix(y, y_pred))

# -----------------------------
# Single Tweet Prediction
# -----------------------------
st.subheader("Predict Sentiment for a Tweet")
tweet_input = st.text_area("Enter a tweet")
if st.button("Predict Tweet Sentiment") and tweet_input:
    if os.path.exists(model_path):
        model_data = joblib.load(model_path)
        cleaner = TextCleaner(**cleaner_kwargs)
        tweet_clean = cleaner.transform([tweet_input])
        if model_key in ["LogisticRegression","LinearSVC","XGBoost","MLP"]:
            model, vec = model_data
            X_vec = vec.transform(tweet_clean)
            pred = model.predict(X_vec)[0]
        elif model_key in ["LSTM","CNN"]:
            model, tokenizer = model_data
            X_seq = pad_sequences(tokenizer.texts_to_sequences(tweet_clean), maxlen=50)
            pred = int((model.predict(X_seq) > 0.5)[0][0])
        elif model_key=="DistilBERT":
            model, tokenizer = model_data
            X_enc = tokenizer(tweet_clean, truncation=True, padding=True, return_tensors='tf')
            pred = int(np.argmax(model.predict(X_enc.data).logits, axis=1)[0])
        elif model_key=="FastText":
            model, ft_model = model_data
            tokenized = [x.split() for x in tweet_clean]
            X_emb = np.array([np.mean([ft_model.wv[w] for w in doc if w in ft_model.wv] or [np.zeros(100)], axis=0) for doc in tokenized])
            pred = model.predict(X_emb)[0]
        st.success(f"Predicted Sentiment: {'Positive' if pred==1 else 'Negative'}")
    else:
        st.error("No trained model found. Train a model first.")
