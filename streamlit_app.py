import streamlit as st
import pandas as pd
import numpy as np
from st_files_connection import FilesConnection

import openai
from openai import OpenAI

from gensim.models.keyedvectors import KeyedVectors

openai.organization = "org-i7aicv7Qc0PO4hkTCT4N2BqR"
openai.api_key = st.secrets["OPENAI_API_KEY"]


st.title('Startup finder')

st.write('Hi')

conn = st.connection('gcs', type=FilesConnection)

embeddings = conn.read("success_new/embeddings.csv", input_format="csv", ttl=600)


embeddings.head()



investment_thesis = "AI agent"
st.write(investment_thesis)

client = OpenAI(api_key=openai.api_key)
model = "text-embedding-3-small"  
response = client.embeddings.create(input=[investment_thesis], model=model).data[0].embedding

