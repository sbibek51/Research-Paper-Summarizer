# Bibek Shiwakoti

'streamlit run summarizer_finetune_1_withFrontend.py'
import streamlit as st
import os
import tiktoken
import gradio as gr
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
# from langchain_community.llms import LLMChain, HuggingFaceHub
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFaceHub
import torch
from PIL import Image
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(
    page_title='Paper Summarizer',
    layout="wide",
    page_icon="🧊",
    initial_sidebar_state="expanded"
)

# print('Welcome to the ArXiv summarization project')

os.environ['HUGGINGFACEHUB_API_TOKEN'] = '?'


def file_preprocess(filepath):
    # loader = PyPDFLoader('parts (1).pdf')
    #     loader = PyPDFLoader('2403.08886v1.pdf')
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    # print(docs)
    # print(type(docs))
    # print(len(docs))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_i = text_splitter.split_documents(docs)
    # print(type(text_i))
    # print(len(text_i))
    text_i_str = " ".join([str(doc) for doc in docs])
    # type(text_i_str)
    # print('Length of the pdf content string is :',len(text_i_str))
    return text_i_str


def led_preprocess(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    return docs


ARTICLE = """ New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York.
    A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.
    Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other.
    In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage.
    Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the
    2010 marriage license application, according to court documents.
    Prosecutors said the marriages were part of an immigration scam.
    On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further.
    After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective
    Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.
    All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say.
    Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.
    Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted.
    The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s
    Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali.
    Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force.
    If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.
    """


def bart_summary(content_text):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    # Tokenize the input text
    inputs = tokenizer(content_text, return_tensors="pt", max_length=1024, truncation=True)
    # Measure the start time
    start_time = time.time()
    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=2500,
        min_length=250,
        do_sample=False,
        length_penalty=1.0,  # Default value
        repetition_penalty=1.0,  # Default value
        early_stopping=False  # Try setting this to True or False
    )

    # Measure the end time
    end_time = time.time()

    # Decode the summary
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Calculate time taken
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60

    # Calculate word counts
    total_words_in_content = len(content_text.split())
    total_words_in_summary = len(summary_text.split())

    # Print results
    # ''' print('Length of summary is:', len(summary_text))
    # print('Summary of the given paper is:')
    # print(summary_text)
    # print(f'Total time taken: {minutes} minutes and {seconds:.2f} seconds')
    # print(f'Total words in content: {total_words_in_content}')
    # print(f'Total words in summary: {total_words_in_summary}') '''

    return summary_text


def led_summary(content_docs):
    llm_led = HuggingFaceHub(repo_id='NielsV/led-arxiv-10240')  # using finetuned model on arxiv
    chain = load_summarize_chain(llm_led, chain_type='map_reduce', verbose=True)
    summary_led = chain.run(content_docs)
    return summary_led


# '''
# summary_text = bart_summary(text_i_str)
# print('Length of summary is:',len(summary_text))
# print('Summary of the given paper is:')
# print(summary_text)
# 
# summary_sentences = summary_text.split('. ')
# formatted_summary = '.\n'.join(summary_sentences)
# print(formatted_summary)
# '''


# Frontend usint the streamlit

# Streamlit code


@st.cache_resource
def displayPDF(file):
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


from transformers import BartTokenizer, BartForConditionalGeneration

# Path to the directory where the model is saved
model_path = 'bart_finetuned (2)'  # Replace with the path where you extracted the model

# Load the model and tokenizer
tokenizer = BartTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)


# Function to summarize text
def summarize_text_bart_finetuned(text):
    # Path to the directory where the model is saved
    model_path = 'bart_finetuned (2)'  # Replace with the path where you extracted the model
    # Load the model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)

    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def main():
    st.title("Paper Summarizer")
    image = Image.open('summarizer dummy.jpg')
    st.image(image, width=200)

    choice = st.sidebar.selectbox("Select your choice",
                                  ["Summarize Text", "Summarize Document-bart", "Summarize Document-Led",'Summarize Document_Bart_Finetuned_local'])

    if choice == "Summarize Text":
        st.subheader("Summarize Text")
        input_text = st.text_area("Enter your text here")
        if input_text is not None:
            if st.button("Summarize Text"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.markdown("**Your Input Text**")
                    st.info(input_text)
                with col2:
                    st.markdown("**Summary Result**")
                    result = bart_summary(input_text)
                    st.success(result)

    elif choice == "Summarize Document":
        st.subheader("Summarize Document")
        uploaded_file = st.file_uploader("Upload your document here", type=['pdf'])
        if uploaded_file is not None:
            if st.button("Summarize Document"):
                col1, col2 = st.columns([1, 1])
                filepath = "paper pdf/" + uploaded_file.name

                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                with col1:
                    st.info("File uploaded successfully")
                    pdf_view = displayPDF(filepath)

                with col2:
                    pdf_text = file_preprocess(filepath)
                    summary = bart_summary(pdf_text)
                    st.info("Summarization")
                    st.success(summary)

    elif choice == "Summarize Document-Led":
        st.subheader("Summarize Document")
        uploaded_file = st.file_uploader("Upload your document here", type=['pdf'])
        if uploaded_file is not None:
            if st.button("Summarize Document"):
                col1, col2 = st.columns([1, 1])
                filepath = "paper pdf/" + uploaded_file.name

                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                with col1:
                    st.info("File uploaded successfully")
                    pdf_view = displayPDF(filepath)

                with col2:
                    pdf_text = file_preprocess(filepath)
                    summary = led_summary(pdf_text)
                    st.info("Summarization")
                    st.success(summary)

    elif choice == "Summarize Document_Bart_Finetuned_local":
        st.subheader("Summarize Document")
        uploaded_file = st.file_uploader("Upload your document here", type=['pdf'])
        if uploaded_file is not None:
            if st.button("Summarize Document"):
                col1, col2 = st.columns([1, 1])
                filepath = "paper pdf/" + uploaded_file.name

                with open(filepath, "wb") as temp_file:
                    temp_file.write(uploaded_file.read())

                with col1:
                    st.info("File uploaded successfully")
                    pdf_view = displayPDF(filepath)

                with col2:
                    pdf_text = file_preprocess(filepath)
                    summary = summarize_text_bart_finetuned(pdf_text)
                    st.info("Summarization")
                    st.success(summary)


# Initializing the app
if __name__ == "__main__":
    main()
