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



def enhance_query(user_query: str, template: str) -> tuple[str, str]:
    prompt = template.replace("{user_input}", user_query)
    response = openai_client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that sharpens queries for vector search."},
            {"role": "user", "content": prompt}
        ],
    )
    return prompt, response.choices[0].message.content.strip()


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
    "dealroom_twitter_followers_chart","dealroom_twitter_tweets_chart","index_old","index", "Description_merged", "website_main_page"
]

default_cols = [
    "Company_Name","Domain_Name","Founded_Year","Country","Description_merged","Company_Stage"
]

# --------------------
# UI
# --------------------
st.title("🚀 Startup Finder")
st.markdown("Enter your **investment thesis** and discover matching startups.")

user_input = st.text_input("💡 Investment Thesis", placeholder="e.g. AI in healthcare")
if not user_input:
    st.stop()



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


# --------------------
# BUTTON: Enhance query only (unchanged trigger)
# --------------------
if st.button("✨ Create Enhanced Query"):
    gpt_prompt, enhanced_query = enhance_query(user_input, prompt_template)
    st.session_state.enhanced_query = enhanced_query

# Always show (and allow editing) if we have one
if "enhanced_query" in st.session_state:
    st.subheader("Enhanced Query (editable)")
    st.session_state.enhanced_query = st.text_area(
        "Edit enhanced query as needed",
        value=st.session_state.enhanced_query,
        height=140,
        key="enhanced_query_edit"
    )


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
    FROM `ccnr-success.success_new.full_merged_new` AS full_table
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
# RUN SEARCH BUTTON (REPLACED)
# --------------------
if st.button("🔍 Run Search"):
    # 1) ORIGINAL embedding + search (unchanged)
    original_vector = openai_client.embeddings.create(
        input=[user_input], model=model
    ).data[0].embedding
    df_original = run_vector_search(original_vector, limit=search_limit)
    if "cosine_distance" in df_original.columns:
        df_original["original_rank"] = df_original["cosine_distance"].rank(method="first", ascending=False).astype(int)
    st.session_state.df_original = df_original

    # 2) Use edited enhanced query if available; otherwise create one
    if st.session_state.get("enhanced_query", "").strip():
        enhanced_query = st.session_state.enhanced_query
    else:
        _, enhanced_query = enhance_query(user_input, prompt_template)
        st.session_state.enhanced_query = enhanced_query

    # 3) ENHANCED embedding + search
    enhanced_vector = openai_client.embeddings.create(
        input=[enhanced_query], model=model
    ).data[0].embedding
    df_enhanced = run_vector_search(enhanced_vector, limit=search_limit)
    if "cosine_distance" in df_enhanced.columns:
        df_enhanced["enhanced_rank"] = df_enhanced["cosine_distance"].rank(method="first", ascending=False).astype(int)
    st.session_state.df_enhanced = df_enhanced

    st.success("Searches complete. See the three sections below.")


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
# OUTPUT (THREE SEPARATE TABLES)
# --------------------
if "df_original" in st.session_state:
    st.subheader("1) 🔎 Matching Startups — Original Query")
    cols_in_df = st.session_state.df_original.columns
    display_cols = []
    # prefer to show rank first if present
    if "original_rank" in cols_in_df:
        display_cols.append("original_rank")
    display_cols += [c for c in all_columns if c in cols_in_df and c in st.session_state.selected_cols]
    if "cosine_distance" in cols_in_df:
        display_cols.append("cosine_distance")
    st.dataframe(st.session_state.df_original[display_cols], use_container_width=True)

if "df_enhanced" in st.session_state:
    st.subheader("2) ✨ Matching Startups — Enhanced Query")
    cols_in_df = st.session_state.df_enhanced.columns
    display_cols = []
    if "enhanced_rank" in cols_in_df:
        display_cols.append("enhanced_rank")
    display_cols += [c for c in all_columns if c in cols_in_df and c in st.session_state.selected_cols]
    if "cosine_distance" in cols_in_df:
        display_cols.append("cosine_distance")
    st.dataframe(st.session_state.df_enhanced[display_cols], use_container_width=True)
    
    
    
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import random
import numpy as np

# ====================================================================
# 📑 RERANK BY DESCRIPTION — use ORIGINAL results (and original thesis)
# ====================================================================
if "df_original" in st.session_state:
    st.markdown("---")
    st.subheader("3) 🧠 Rerank by Description (based on ORIGINAL Results)")

    default_rerank_prompt = f"""Rerank the following startups STRICTLY by how well their Description matches this investment thesis:

THESIS:
{user_input}

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
        try:
            return json.loads(text)
        except Exception:
            pass
        code_block = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except Exception:
                pass
        first_obj = re.search(r"(\{[\s\S]*\})", text)
        if first_obj:
            try:
                return json.loads(first_obj.group(1))
            except Exception:
                pass
        raise ValueError("Could not parse JSON from model response.")

    def _build_items_for_model(df: pd.DataFrame):
        items = []
        for pos, row in enumerate(df.itertuples(index=False), start=1):
            idx = getattr(row, "index") if "index" in df.columns else None
            name = getattr(row, "Company_Name") if "Company_Name" in df.columns else ""
            desc = getattr(row, "Description_merged") if "Description_merged" in df.columns else ""
            items.append({
                "index": int(idx) if pd.notna(idx) else pos,
                "current_rank": pos,
                "Company_Name": str(name) if pd.notna(name) else "",
                "Description_merged": str(desc) if pd.notna(desc) else ""
            })
        return items

    def rerank_by_description(df: pd.DataFrame, thesis: str, prompt_text: str, model_name: str) -> pd.DataFrame:
        items = _build_items_for_model(df)
        user_msg = prompt_text + "\n\nITEMS:\n" + json.dumps(items, ensure_ascii=False)

        response = openai_client.chat.completions.create(
            model='gpt-5',
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
        original_ids = [int(i["index"]) for i in items]

        # Validate/fix order
        if len(proposed_order) != len(original_ids) or set(proposed_order) != set(original_ids):
            intersection = [i for i in proposed_order if i in set(original_ids)]
            missing = [i for i in original_ids if i not in set(intersection)]
            proposed_order = intersection + missing

        if "index" not in df.columns:
            raise ValueError("Dataframe missing 'index' column required for reranking.")

        df_reordered = df.set_index("index").loc[proposed_order].reset_index()
        return df_reordered

    # --- 1) Compute once on click; store in session_state ---
    if st.button("🏁 Run Rerank (Based on the enhanced results, using original query)"):
        try:
            with st.spinner("🔄 Reranking startups... please wait"):
                base_df = st.session_state.df_original.copy()
                thesis_str = user_input  # or st.session_state.enhanced_query
                reranked = rerank_by_description(
                    df=base_df,
                    thesis=thesis_str,
                    prompt_text=rerank_prompt,
                    model_name='gpt-5'
                )
                st.session_state.df_reranked = reranked
            st.success("✅ Reordered by description relevance.")
        except Exception as e:
            st.error(f"Reranking failed: {e}")

    # --- 2) Always render from session_state (survives reruns) ---
    if "df_reranked" not in st.session_state:
        st.info("👆 Run Rerank to see results here.")
        st.stop()

    reranked = st.session_state.df_reranked
    enhanced = st.session_state.df_enhanced
    
    # Build an integer "score" from cosine_distance (lower distance = higher score)
    if "cosine_distance" in reranked.columns:
        d = pd.to_numeric(enhanced["cosine_distance"], errors="coerce")
        reranked = reranked.copy()
    
        if d.notna().any():
            d_min = float(d.min())
            d_max = float(d.max())
    
            # pick a random top score between 95 and 100
            top_score = random.randint(95, 100)
            
            if d_max > d_min:
                # scale so that min distance -> top_score
                # others shrink proportionally into [top_score-5, top_score]
                norm = (d - d_min) / (d_max - d_min)  # max d = 1, min d = 0
                score = (top_score - 5) + norm * 5
            else:
                score = pd.Series(top_score, index=d.index)  # all same distances
    
            reranked["score"] = np.rint(score).astype(int)
    
    # Recompute df_to_show with "score" instead of "cosine_distance"
    cols_in_df = reranked.columns
    display_cols = []
    if "original_rank" in cols_in_df:
        display_cols.append("original_rank")
    
    display_cols += [
        c for c in all_columns
        if c in cols_in_df and c in st.session_state.selected_cols and c != "cosine_distance"
    ]
    
    if "score" in cols_in_df:
        display_cols.append("score")
    
    df_to_show = reranked[display_cols]


    # --- 3) AgGrid with "Set Filter" checkboxes (Enterprise) but NO floating row ---
    st.subheader("📊 Reranked results (AgGrid with checkbox filters)")
        
    gb = GridOptionsBuilder.from_dataframe(df_to_show)
    
    # Defaults (no floating filters, enterprise on elsewhere in your code)
    gb.configure_default_column(sortable=True, resizable=True, filter=True, floatingFilter=True)
    gb.configure_side_bar()
    gb.configure_grid_options(suppressMenuHide=True)
    
    
           # ---- Set preset width only for Description ----
    if "Description_merged" in df_to_show.columns:
        gb.configure_column(
            "Description_merged",
            width=520,            # make Description wider
            suppressSizeToFit=True,
            wrapText=True,
            autoHeight=False      # fixed row height is fine
        )
    
    # Build grid options
    grid_options = gb.build()
    
    # Just bump row height globally (everything else stays default)
    grid_options["rowHeight"] = 36   # default ~25, now a bit taller
    
    grid_response = AgGrid(
        df_to_show,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=False,   # don’t overwrite our Description width
        height=420,
        enable_enterprise_modules=True,
        key="aggrid_reranked_table",
    )
