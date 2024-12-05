import streamlit as st
import pandas as pd
import numpy as npimport
from st_files_connection import FilesConnection

st.title('Startup finder')

st.write('Hi')


conn = st.connection('gcs', type=FilesConnection)
df = conn.read("success_new/embeddings.csv", input_format="csv", ttl=600)


df

