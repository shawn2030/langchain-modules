# app.py

import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from langchain_huggingface import HuggingFacePipeline  # LangChain wrapper for HF pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# ------------------ STREAMLIT UI SETUP ------------------ #
st.set_page_config(page_title="Multilingual Paraphraser", layout="wide")
st.title("🌀 Multilingual Paraphraser")

# Text input area for user to enter a sentence
text_input = st.text_area("✍️ Enter a Sentence to Paraphrase:", height=200)

# Run the paraphrasing when user submits input
if text_input:
    # ------------------ LOAD MODEL & TOKENIZER ------------------ #
    tokenizer = T5Tokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws", use_fast=False, legacy=False)
    model = T5ForConditionalGeneration.from_pretrained("Vamsi/T5_Paraphrase_Paws")

    # ------------------ CREATE PARAPHRASING PIPELINE ------------------ #
    paraphrase_pipeline = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=64,
        do_sample=True,         # Enable sampling for diversity
        top_k=100,               # Top-k sampling
        top_p=0.92,             # Nucleus sampling
        temperature=1.5,        # Add randomness for better paraphrasing
        repetition_penalty=10.2,
        num_return_sequences=3,
 # Penalize repeating phrases
    )

    # ------------------ LANGCHAIN WRAPPER ------------------ #
    llm = HuggingFacePipeline(pipeline=paraphrase_pipeline)

    # Prompt format that model expects: "paraphrase: <text> </s>"
    prompt = PromptTemplate(
        input_variables=["text"],
        template="paraphrase: {text} </s>"
    )

    # Chain = Prompt Template + LLM
    paraphrase_chain = prompt | llm

    # ------------------ PARAPHRASE INVOCATION ------------------ #
    with st.spinner("Paraphrasing..."):
        result = paraphrase_chain.invoke({"text": text_input})

    # ------------------ DISPLAY RESULT ------------------ #
    st.markdown("### ✅ Paraphrased Sentence:")
    st.markdown(
    f"<div style='background-color: #ffffff; color: #000000; padding: 15px; border-radius: 5px;'>{result}</div>",
    unsafe_allow_html=True
    )



