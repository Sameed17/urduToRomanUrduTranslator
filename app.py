import streamlit as st
from unigram_urdu_roman_seq2seq import (
    load_tokenizers_and_model,
    load_xlstm_tokenizers_and_model,
    streamlit_translate_urdu_to_roman
)

st.set_page_config(page_title="Urdu to Roman Urdu Translator", layout="centered")

@st.cache_resource(show_spinner=True)
def load_resources():
    return load_tokenizers_and_model()

st.title("Urdu to Roman Urdu Translator")
st.write("Enter Urdu text (max 400 characters) and get its Roman Urdu translation.")

urdu_input = st.text_area("Urdu Text", max_chars=400, height=120, placeholder="یہ ایک مثال ہے")

if st.button("Translate"):
    if urdu_input.strip():
        with st.spinner("Translating..."):
            model, urdu_tokenizer, roman_tokenizer, device = load_resources()
            roman_output = streamlit_translate_urdu_to_roman(
                urdu_input, model, urdu_tokenizer, roman_tokenizer, device
            )
        st.markdown("**Roman Urdu Translation:**")
        st.success(roman_output)
    else:
        st.warning("Please enter some Urdu text to translate.")