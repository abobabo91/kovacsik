import streamlit as st
import pandas as pd
import numpy as np
from st_files_connection import FilesConnection

from google.oauth2 import service_account
from google.cloud import bigquery

import openai
from openai import OpenAI

def run_query(query):
    query_job = bq_client.query(query)
    rows_raw = query_job.result()
    # Convert to list of dicts. Required for st.cache_data to hash the return value.
    rows = [dict(row) for row in rows_raw]
    return rows


openai.organization = "org-i7aicv7Qc0PO4hkTCT4N2BqR"
openai.api_key = st.secrets['openai']["OPENAI_API_KEY"]



credentials = service_account.Credentials.from_service_account_info(st.secrets["gcp_service_account"])
bq_client = bigquery.Client(credentials=credentials)

st.title('Startup finder')

st.write('Hi guys!')

investment_thesis = 'asd asd'

user_input = st.text_input('Enter your query here:')



investment_thesis = user_input

openai_client = OpenAI(api_key=openai.api_key)
model = "text-embedding-3-small"  
openai_vector = openai_client.embeddings.create(input=[investment_thesis], model=model).data[0].embedding


vector_query = """WITH 
first_row AS ( SELECT """ + str(openai_vector) + """ AS vector,
              -1 AS dealroom_index
),

distances AS (
  SELECT 
    t.dealroom_index, 
    1 - ML.DISTANCE(t.vector, f.vector, 'COSINE') AS cosine_distance
  FROM 
    `ccnr-success.success_new.merged` t,
    first_row f
  WHERE 
    t.vector[0] != 0
    and f.dealroom_index != t.dealroom_index
)

SELECT 
  full_table.dealroom_index, 
  full_table.NAME,
  full_table.WEBSITE,
  full_table.gpt_description,
  distances.cosine_distance
FROM 
    `ccnr-success.success_new.merged` as full_table 
    join
    distances as distances
    on full_table.dealroom_index = distances.dealroom_index
ORDER BY 
  distances.cosine_distance DESC
LIMIT 25;"""


rows = run_query(vector_query)

indexes = [[x['dealroom_index'],x['NAME'],x['cosine_distance'],x['WEBSITE'],x['gpt_description'] ] for x in rows]

indexes


