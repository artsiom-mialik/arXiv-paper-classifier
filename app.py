import streamlit as st
import torch
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./final_model"
MAX_LENGTH = 256

CLASSES = ['cs', 'econ', 'eess', 'math', 'physics', 'q-bio', 'q-fin', 'stat']
idx_to_classes = {i: c for i, c in enumerate(CLASSES)}
DISPLAY_NAMES = {
    "cs": "Computer Science",
    "econ": "Economics",
    "eess": "Electrical Engineering and Systems Science",
    "math": "Mathematics",
    "physics": "Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
}


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

def data_to_text(title, abstract):
    if abstract and title:
        return "Title: " + title.strip() + " Abstract: " + abstract.strip()
    elif title:
        return "Title: " + title.strip()
    elif abstract:
        return "Abstract: " + abstract.strip()
    else:
        return ""

def prediction(text, tokenizer, model):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = torch.softmax(logits, dim=-1).squeeze(0)

    idx_sorted = torch.argsort(probs, descending=True)
    results = []
    cum_sum = 0
    for idx in idx_sorted:
        idx = idx.item()
        prob = probs[idx].item()
        cum_sum += prob
        results.append((idx_to_classes[idx], prob))
        if cum_sum >= 0.95:
            break

    return results


tokenizer, model = load_model()

sample_df = pd.read_csv("test_examples.csv")

def sample_data():
    sample = sample_df.sample(1).iloc[0]
    st.session_state.title = sample["title"]
    st.session_state.abstract = sample["abstract"]

def clear_fields():
    st.session_state.title = ""
    st.session_state.abstract = ""


st.title("ArXiv paper classification")

st.markdown("""This app classifies arXiv papers based on their title and abstract into 8 different topic categories.
                The model is based on SciBert embedding with a head trained on Kaggle arXiv dataset.
                You may input your own article title and/or abstract or sample a random one and get top categories which cover 95% of the predicted probabilities
            """
            )

if "title" not in st.session_state:
    st.session_state.title = ""
if "abstract" not in st.session_state:
    st.session_state.abstract = ""

title = st.text_input(label="Title", placeholder="Enter title of your paper", key="title")
abstract = st.text_area(label="Abstract", placeholder="Enter abstract of your paper", key="abstract", height=250)

col1, col2, col3 = st.columns(3)
with col1:
    classify_button = st.button("Classify")
with col2:
    st.button("Sample Title and Abstract", on_click=sample_data)
with col3:
    st.button("Clear", on_click=clear_fields)

if classify_button:
    text = data_to_text(title, abstract)
    if not text:
        st.error("Please enter a title or an abstract")
    else:
        try:
            preds = prediction(text, tokenizer, model)
            best_label, best_prob = preds[0]
            st.markdown(f"**Top match topic is:** {DISPLAY_NAMES[best_label]} ({best_prob*100:.2f}%)")

            st.markdown("**Top 95% categories:** ")
            for label, prob in preds:
                st.write(f"{DISPLAY_NAMES[label]}: {prob * 100:.2f}%")
        except Exception as e:
            st.error(e)

