import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import openai
from openai import OpenAI

# NEW: for JSON parsing
import json
import re

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
# QUERY ENHANCER
# --------------------
st.sidebar.header("🤖 Query Enhancer")

model_choice = st.sidebar.radio(
    "Choose model for query enhancement:",
    ["gpt-5-nano", "gpt-5-mini", "gpt-5"],
    index=1
)

default_prompt_template = f"""You are an assistant helping to improve search queries for investment thesis matching.
The user provided this query: "{user_input}"

Rephrase the query in a way that makes it highly precise for vector search:
- Preserve all niche / domain-specific details
- Avoid generalization
- Optimize for embeddings and retrieval of relevant companies
- Keep it concise but information-rich
"""

prompt_template = st.text_area(
    "📝 Prompt Template (edit as you like)",
    value=default_prompt_template,
    height=200
)

def enhance_query(user_query: str, model: str, template: str) -> tuple[str, str]:
    prompt = template.replace("{user_input}", user_query)
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that sharpens queries for vector search."},
            {"role": "user", "content": prompt}
        ],
    )
    return prompt, response.choices[0].message.content.strip()

# --------------------
# VECTOR SEARCH
# --------------------
def run_vector_search(vector, limit=50):
    query = f"""
    WITH first_row AS (
      SELECT {vector} AS vector, -1 AS embed_index
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
    LIMIT {limit};
    """
    return pd.DataFrame(run_query(query))

# --------------------
# RUN SEARCH BUTTON
# --------------------

# 🔧 Add sidebar box for result limit
search_limit = st.sidebar.number_input(
    "Max results to fetch",
    min_value=5,
    max_value=200,
    value=20,
    step=5
)

# --------------------
# RUN SEARCH BUTTON
# --------------------

if st.button("🔍 Run Search"):
    # Enhance query
    gpt_prompt, enhanced_query = enhance_query(user_input, model_choice, prompt_template)

    st.subheader("✨ Enhanced Query")
    st.markdown(f"**{enhanced_query}**")

    # Embeddings
    original_vector = openai_client.embeddings.create(
        input=[user_input], model=model
    ).data[0].embedding

    enhanced_vector = openai_client.embeddings.create(
        input=[enhanced_query], model=model
    ).data[0].embedding

    # Run vector searches with chosen limit
    df_original = run_vector_search(original_vector, limit=search_limit)
    df_original["non_enhanced_rank"] = df_original["cosine_distance"].rank(
        method="first", ascending=False
    ).astype(int)

    df_enhanced = run_vector_search(enhanced_vector, limit=search_limit)

    # Merge results
    df_results = df_enhanced.merge(
        df_original[["index", "non_enhanced_rank"]],
        on="index",
        how="left"
    )

    # Order once, then store
    col_order = (
        ["non_enhanced_rank"]
        + [c for c in all_columns if c in df_results.columns]
        + (["cosine_distance"] if "cosine_distance" in df_results.columns else [])
    )
    df_results = df_results[col_order]

    # Persist just the dataframe AND the enhanced_query for later reranking
    st.session_state.df_results = df_results
    st.session_state.enhanced_query = enhanced_query

# --------------------
# COLUMN SELECTOR
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
# OUTPUT (original enhanced ranking)
# --------------------
if "df_results" in st.session_state:
    st.subheader("🔎 Matching Startups (Enhanced Ranking)")

    cols_in_df = st.session_state.df_results.columns

    display_cols = []
    if "non_enhanced_rank" in cols_in_df:
        display_cols.append("non_enhanced_rank")

    display_cols += [
        c for c in all_columns
        if c in cols_in_df and c in st.session_state.selected_cols
    ]

    if "cosine_distance" in cols_in_df:
        display_cols.append("cosine_distance")

    st.dataframe(st.session_state.df_results[display_cols], use_container_width=True)
else:
    st.info("Click **Run Search** to see results.")

# ====================================================================
# 📑 RERANK BY DESCRIPTION (NEW)
# ====================================================================
if "df_results" in st.session_state:
    st.markdown("---")
    st.subheader("📑 Rerank by Description")

    # Strict, non-generalizing default prompt
    default_rerank_prompt = f"""Rerank the following startups STRICTLY by how well their Description matches this investment thesis:

THESIS:
{st.session_state.get('enhanced_query', user_input)}

IMPORTANT RULES:
- Do NOT generalize or broaden the scope beyond the thesis wording.
- Consider ONLY the provided Description (and Company_Name to disambiguate), ignore all other fields.
- If two items are equally relevant, keep the one with LOWER current_rank first (stable sort).
- Return only valid JSON with this exact shape (no commentary, no code block):
{{
  "order": [list of the provided numeric index ids best-to-worst, same length as input]
}}

INPUT ITEMS FORMAT:
- Each item has: index (int), current_rank (int), Company_Name (string), Description (string).
- Use index to produce the order.
"""

    rerank_prompt = st.text_area(
        "📝 Rerank Prompt (strict, no generalization)",
        value=default_rerank_prompt,
        height=260
    )

    # Helper: extract JSON robustly from model output
    def _extract_json_object(text: str):
        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try fenced code blocks
        code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except Exception:
                pass
        # Try first {...} blob
        first_obj = re.search(r"(\{[\s\S]*\})", text)
        if first_obj:
            try:
                return json.loads(first_obj.group(1))
            except Exception:
                pass
        raise ValueError("Could not parse JSON from model response.")

    # Build items we send to the model
    def _build_items_for_model(df: pd.DataFrame):
        # Use current display order (enhanced ranking) and include a stable tie-breaker current_rank
        # current_rank is the row position (1-based) in the displayed df
        items = []
        for pos, row in enumerate(df.itertuples(index=False), start=1):
            idx = getattr(row, "index") if "index" in df.columns else None
            name = getattr(row, "Company_Name") if "Company_Name" in df.columns else ""
            desc = getattr(row, "Description") if "Description" in df.columns else ""
            items.append({
                "index": int(idx) if pd.notna(idx) else pos,
                "current_rank": pos,
                "Company_Name": str(name) if pd.notna(name) else "",
                "Description": str(desc) if pd.notna(desc) else ""
            })
        return items

    # Do the rerank
    def rerank_by_description(df: pd.DataFrame, thesis: str, prompt_text: str, model_name: str) -> pd.DataFrame:
        items = _build_items_for_model(df)

        user_msg = prompt_text + "\n\nITEMS:\n" + json.dumps(items, ensure_ascii=False)
        response = openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a precise reranker. Follow the instructions exactly."},
                {"role": "user", "content": user_msg}
            ],
        )
        content = response.choices[0].message.content.strip()
        parsed = _extract_json_object(content)

        if not isinstance(parsed, dict) or "order" not in parsed or not isinstance(parsed["order"], list):
            raise ValueError("Model response missing a valid 'order' list.")

        proposed_order = [int(x) for x in parsed["order"]]

        # Validate: same set and length
        original_ids = [int(i["index"]) for i in items]
        if len(proposed_order) != len(original_ids):
            raise ValueError("Order length mismatch.")
        if set(proposed_order) != set(original_ids):
            # Be forgiving: filter to intersection, then append any missing by current order
            intersection = [i for i in proposed_order if i in set(original_ids)]
            missing = [i for i in original_ids if i not in set(intersection)]
            proposed_order = intersection + missing

        # Reindex dataframe by 'index' and reorder
        if "index" not in df.columns:
            raise ValueError("Dataframe missing 'index' column required for reranking.")

        df_reordered = (
            df.set_index("index")
              .loc[proposed_order]
              .reset_index()
        )
        return df_reordered

    # Button to run the rerank
    if st.button("🧠 Rerank by Description"):
        try:
            base_df = st.session_state.df_results.copy()
            thesis_str = st.session_state.get("enhanced_query", user_input)

            reranked = rerank_by_description(
                df=base_df,
                thesis=thesis_str,
                prompt_text=rerank_prompt,
                model_name=model_choice  # reuse the same radio
            )

            # Persist and show
            st.session_state.df_reranked = reranked

            st.success("Reordered by description relevance.")
            cols_in_df = reranked.columns
            display_cols = []
            if "non_enhanced_rank" in cols_in_df:
                display_cols.append("non_enhanced_rank")
            display_cols += [
                c for c in all_columns
                if c in cols_in_df and c in st.session_state.selected_cols
            ]
            if "cosine_distance" in cols_in_df:
                display_cols.append("cosine_distance")

            st.subheader("🏁 Final Reordered Results (Description-based Rerank)")
            st.dataframe(reranked[display_cols], use_container_width=True)

        except Exception as e:
            st.error(f"Reranking failed: {e}")
