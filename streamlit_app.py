import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import openai
from openai import OpenAI


#IDEAS:
# - A VEGEN A TOP 50 RESULTOT UJRA VISSZAADNI ES MEGKERNI HOGY RANGSOROLJA UJRA PRECIZEN A QUERYRE
# - AZ ELEJEN ATIRATNI A QUERYT ES AZT KERESNI
# - VALAHOGY AZ ERTELMET ATADNI A QUERYNEK ÉS NEM CSAK A SZAVAK ÖSSZESSEGET, 
#ENNEK UTANANEZNI, HOGY EGYALTALN KEPES E A VEKTORIZALAS ILYENRE, HOGY MUKODIK PL AZ OPENAI VEKTORIZALASA, STB


#* understand more in depth how 'smart' this is on the spectrum of simple keyword search to full context understanding
#* could be worth adding a chatgpt boost layer right after query submission, so instead of searching raw words the system enhances the query to maximize output quality. this would help when the search is vague, too broad, or too narrow
#* work on reducing irrelevant results that contain the right words but in the wrong context. a chatgpt check or ranking round at the end could help
#* add filtering
#* the current version feels a bit mixed between startup and thesis. if we wanna give this to target customers soon we should only include value-add startup-focused stuff. i.e. pick which columns are useful, rename them, etc.
#* I'll think about a more fitting line than "enter your investment thesis and discover matching startups" - this is more VC lingo
#* make sure every company has a description, there are some gaps now where it's missing on tracxn
#* change 'cosine_distance' to something catchy like match score (%). could even distort it a bit so scores are higher and matches feel stronger



# --------------------
# SETUP
# --------------------
st.set_page_config(page_title="Startup Finder", layout="wide")

openai.organization = "org-i7aicv7Qc0PO4hkTCT4N2BqR"
openai.api_key = st.secrets['openai']["OPENAI_API_KEY"]

credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"]
)
bq_client = bigquery.Client(credentials=credentials)

openai_client = OpenAI(api_key=openai.api_key)
model = "text-embedding-3-small"


def run_query(query: str):
    query_job = bq_client.query(query)
    rows_raw = query_job.result()
    return [dict(row) for row in rows_raw]


# --------------------
# COLUMN ORDER LIST
# --------------------
all_columns = [
    "Company_Name","Domain_Name","Founded_Year","Country","Description",
    "Sector_Pratice_Area_Feed_","Business_Models","Special_Flags_TRUE","Company_Stage",
    "All_Associated_Legal_Entities_Name_CIN_","Total_Funding_USD_","Latest_Funded_Amount_USD_",
    "Latest_Funded_Date","Latest_Valuation_USD_","Institutional_Investors","Angel_Investors",
    "Key_People_Info","Key_People_Email_Ids","Links_to_Key_People_Profiles","Acquisition_List",
    "Is_Acquired","Acquired_By","Acquired_Date","Acquired_Amount_USD_","Is_IPO","IPO_Date",
    "Market_Capital_at_IPO_USD_","Editor_s_Rating_x","Tracxn_Score","Company_Emails",
    "Company_Phone_Numbers","LinkedIn","Twitter","Facebook","Blog_Url","Date_Added","Is_Deadpooled",
    "Deadpooled_Date","name_nr_of_chars","domain_extension","Reference_Year","Company_Age","Continent",
    "Region","Legal_Entities_binary","Stage_Level","Funding_Speed","Funding_Amount_per_Year",
    "Key_People_Email_binary","Company_Emails_binary","Company_Phone_Numbers_binary","LinkedIn_binary",
    "Twitter_binary","Facebook_binary","Blog_Url_binary","Revenue_Quantile","Profit_Quantile",
    "EBITDA_Quantile","Employees_Quantile","nr_of_acquisitions","nr_of_acquihires",
    "Acquired_Companies_Avg_Age","Nr_of_Acquisitions_per_Year","Nr_of_Angel_Investors",
    "Angel_Investor_Descriptions","Angel_Investor_LinkedIn_Links","Nr_of_Board_Members",
    "Board_Member_Descriptions","Board_Member_LinkedIn_Links","Avg_Board_Member_Tenure",
    "Avg_Board_Member_Tenure_Age","Exchange","Founder_Descriptions","Founder_LinkedIn_Links",
    "Nr_of_Founders","CEO_is_Founder_","Other_Team_Member_Descriptions",
    "Other_Team_Member_LinkedIn_Links","Editor_s_Rating_y","Mobile_Downloads",
    "Number_of_Funding_Rounds","Number_of_Institutional_Investors","News_Articles_All_Time_",
    "News_Articles_Growth_12_months_","News_Articles_Growth_YoY_","Stage_Number_of_Investments",
    "Latest_Stage_Speed_Quantile","Seed_Stage_Speed_Quantile","Angel_Stage_Speed_Quantile",
    "Series_A_Stage_Speed_Quantile","News_Articles_All_Time_Quantile",
    "News_Articles_Growth_12_months_Quantile","News_Articles_Growth_YoY_Quantile",
    "Mobile_Downloads_Quantile","pitchbook_Company_Website","pitchbook_Description",
    "pitchbook_Keywords","pitchbook_All_Industries","pitchbook_Verticals",
    "pitchbook_CEO_at_time_of_deal_","pitchbook_CEO_Biography","pitchbook_CEO_Education",
    "pitchbook_Revenue","pitchbook_Gross_Profit","pitchbook_Net_Income","pitchbook_EBITDA",
    "pitchbook_EBIT","pitchbook_Valuation_Revenue","pitchbook_Valuation_EBITDA","pitchbook_Deal_Size",
    "pitchbook_Deal_Size_Revenue","pitchbook_Deal_Size_EBITDA","pitchbook_Fiscal_Year",
    "pitchbook_Current_Employees","pitchbook_Total_Patent_Documents","pitchbook_Active_Patent_Documents",
    "pitchbook_Pending_Patent_Documents","pitchbook_Top_CPC_Codes","Years_Since_Founded_Pitchbook_",
    "dealroom_website_domain_full","dealroom_employees_latest","dealroom_client_focus",
    "dealroom_income_streams","dealroom_revenues","dealroom_fundings",
    "dealroom_app_12_months_growth_unique","dealroom_dealroom_signal","dealroom_patents_count",
    "dealroom_twitter_followers_chart","dealroom_twitter_tweets_chart","index_old","index"
]

default_cols = [
    "Company_Name","Domain_Name","Founded_Year","Country","Description","Company_Stage"
]


# --------------------
# UI
# --------------------
st.title("🚀 Startup Finder")
st.markdown("Enter your **investment thesis** and discover matching startups.")

user_input = st.text_input("💡 Investment Thesis", placeholder="e.g. AI in healthcare")
if not user_input:
    st.stop()

# --------------------
# SIDEBAR checkboxes
# --------------------
st.sidebar.header("⚙️ Display Options")

if "selected_cols" not in st.session_state:
    st.session_state.selected_cols = set(default_cols)

selected_cols = []
if st.sidebar.button("✅ Select All"):
    st.session_state.selected_cols = set(all_columns)
if st.sidebar.button("❌ Clear All"):
    st.session_state.selected_cols = set()

for col in all_columns:
    checked = st.sidebar.checkbox(col, value=(col in st.session_state.selected_cols))
    if checked:
        selected_cols.append(col)

st.session_state.selected_cols = set(selected_cols)


# --------------------
# EMBEDDINGS
# --------------------
openai_vector = openai_client.embeddings.create(
    input=[user_input], model=model
).data[0].embedding

# --------------------
# QUERY (all cols, order not guaranteed yet)
# --------------------
vector_query = f"""
WITH first_row AS (
  SELECT {openai_vector} AS vector, -1 AS embed_index
),
distances AS (
  SELECT 
    t.index, 
    1 - ML.DISTANCE(t.vector, f.vector, 'COSINE') AS cosine_distance
  FROM `ccnr-success.success_new.full_merged` t,
       first_row f
  WHERE t.vector[0] != 0
    AND f.embed_index != t.index
)
SELECT full_table.*, distances.cosine_distance
FROM `ccnr-success.success_new.full_merged` AS full_table
JOIN distances
  ON full_table.index = distances.index
ORDER BY cosine_distance DESC
LIMIT 25;
"""

rows = run_query(vector_query)
df_results = pd.DataFrame(rows)

# reorder columns according to our list, add cosine_distance at the end
ordered_cols = [c for c in all_columns if c in df_results.columns] + ["cosine_distance"]
df_results = df_results[ordered_cols]

# --------------------
# OUTPUT
# --------------------
st.subheader("🔎 Matching Startups")
if selected_cols:
    display_cols = [c for c in ordered_cols if c in selected_cols or c == "cosine_distance"]
    st.dataframe(df_results[display_cols], use_container_width=True)
else:
    st.warning("No columns selected. Please pick at least one in the sidebar.")
