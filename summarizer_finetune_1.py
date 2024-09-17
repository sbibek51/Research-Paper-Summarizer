#Bibek Shiwakoti
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


print('Welcome to the ArXiv summarization project')

os.environ['HUGGINGFACEHUB_API_TOKEN'] ='?'


# loader = PyPDFLoader('parts (1).pdf')
loader = PyPDFLoader('2403.08886v1.pdf')

docs = loader.load()


# print(docs)

print(type(docs))
print(len(docs))

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
text_i = text_splitter.split_documents(docs)

print(type(text_i))
print(len(text_i))

text_i_str = " ".join([str(doc) for doc in docs])
type(text_i_str)
print('Length of the pdf content string is :',len(text_i_str))


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

import time
from transformers import pipeline

import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


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
    print('Length of summary is:', len(summary_text))
    print('Summary of the given paper is:')
    # print(summary_text)
    print(f'Total time taken: {minutes} minutes and {seconds:.2f} seconds')
    print(f'Total words in content: {total_words_in_content}')
    print(f'Total words in summary: {total_words_in_summary}')

    return summary_text


summary_text = bart_summary(text_i_str)
print('Length of summary is:',len(summary_text))
print('Summary of the given paper is:')
# print(summary_text)

summary_sentences = summary_text.split('. ')
formatted_summary = '.\n'.join(summary_sentences)
print(formatted_summary)


#Frontend usint the streamlit

