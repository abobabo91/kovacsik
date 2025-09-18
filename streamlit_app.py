import streamlit as st
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery
import openai
from openai import OpenAI

# NEW: for JSON parsing
import json
import re
import random
import numpy as np
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

import time
from typing import List, Dict

# Claude (Anthropic)
import anthropic

# --------------------
# SETUP
# --------------------
st.set_page_config(page_title="GoldenClue", layout="wide")

openai.organization = "org-i7aicv7Qc0PO4hkTCT4N2BqR"
openai.api_key = st.secrets['openai']["OPENAI_API_KEY"]


anthropic_client = anthropic.Client(api_key=st.secrets["anthropic"]["ANTHROPIC_API_KEY"])


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
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that sharpens queries for vector search."},
            {"role": "user", "content": prompt}
        ],
    )
    return prompt, response.choices[0].message.content.strip()


FOUNDED_YEARS = [
    2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017,
    2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025
]

COMPANY_STAGES = [
    "Early Stage",
    "Seed",
    "Series A",
    "Series B",
    "Series C",
    "Series D",
    "Series E",
    "Series F",
    "Series G",
    "Series H",
    "Late Stage",
    "Funding Raised",   # generic / ambiguous
    "Acqui-Hired",      # exit path
    "Acquired",         # exit path
    "Public",           # IPO stage
    "Deadpooled"        # shut down
]


COUNTRIES = sorted(list({
    "Italy","France","China","United States","United Kingdom","Canada","Germany","Sweden",
    "Colombia","Switzerland","Kazakhstan","India","Saudi Arabia","Australia","Netherlands",
    "Israel","Spain","Cyprus","South Korea","Belgium","Ireland","Portugal","Brazil","Egypt",
    "Japan","Cayman Islands","South Africa","Indonesia","Estonia","Nigeria","Puerto Rico",
    "Denmark","Romania","Finland","Kenya","Czech Republic","Singapore","Taiwan","Chile",
    "Poland","Austria","Malaysia","Rwanda","Ivory Coast","Republic Of Congo","Norway",
    "Thailand","Kuwait","Philippines","Tunisia","Greece","Pakistan","Morocco","Turkey",
    "Iceland","New Zealand","Costa Rica","Venezuela","Slovenia","Croatia","Ethiopia",
    "Jordan","Argentina","Russia","Bangladesh","Luxembourg","Bulgaria","Mexico","Cameroon",
    "Qatar","Lithuania","Uruguay","Vietnam","Hungary","Lebanon","Slovakia","Peru","Moldova",
    "Latvia","Ghana","Uganda","Ukraine","Oman","Zambia","Anguilla","Honduras","Bahrain",
    "Zimbabwe","Senegal","Serbia","Tanzania","Guernsey","Mauritius","Ecuador","Malta",
    "Panama","Myanmar","Iran","Belarus","American Samoa","Palestine","Sri Lanka",
    "Channel Islands","Montenegro","Bermuda","Angola","Gibraltar","Jamaica","Azerbaijan",
    "Georgia","Bahamas","Guatemala","North Macedonia","Scotland","Bosnia And Herzegovina",
    "Iraq","Congo"
}))


continent_map = {
    "Europe": [
        "Italy","France","United Kingdom","Germany","Sweden","Switzerland","Netherlands",
        "Spain","Cyprus","Belgium","Ireland","Portugal","Estonia","Denmark","Romania",
        "Finland","Czech Republic","Poland","Norway","Lithuania","Slovakia","Hungary",
        "Moldova","Latvia","Slovenia","Croatia","Serbia","Bulgaria","Luxembourg","Ukraine",
        "Belarus","Russia","Iceland","Greece","Turkey","Montenegro","North Macedonia",
        "Scotland","Bosnia And Herzegovina"
    ],
    "Asia": [
        "China","India","Israel","South Korea","Japan","Singapore","Taiwan","Malaysia",
        "Indonesia","Thailand","Philippines","Vietnam","Kazakhstan","Saudi Arabia",
        "United Arab Emirates","Qatar","Oman","Kuwait","Pakistan","Bangladesh","Lebanon",
        "Jordan","Iran","Palestine","Sri Lanka","Georgia","Azerbaijan"
    ],
    "Africa": [
        "South Africa","Nigeria","Kenya","Rwanda","Egypt","Morocco","Tunisia","Ivory Coast",
        "Ghana","Uganda","Ethiopia","Republic Of Congo","Zimbabwe","Senegal","Tanzania",
        "Angola","Cameroon","Congo"
    ],
    "North America": [
        "United States","Canada","Mexico","Puerto Rico","Costa Rica","Panama","Jamaica",
        "Honduras","Guatemala","Bermuda","Bahamas","Anguilla"
    ],
    "South America": [
        "Brazil","Argentina","Chile","Colombia","Uruguay","Peru","Venezuela","Ecuador"
    ],
    "Oceania": ["Australia","New Zealand"],
}



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

column_rules = {
    "Company_Name": (1, "Name"),
    "Domain_Name": (1, "Domain"),
    "Founded_Year": (1, "Year Founded"),
    "Country": (1, "Country"),
    "Company_Stage": (1, "Stage"),
    "Description_merged": (1, "Description"),
    "score": (1, "Match Score"),
    "Latest_Funded_Amount_USD_": (2, "Latest Funding Amount"),
    "Latest_Funded_Date": (2, "Latest Funding Date"),
    "Company_Emails": (2, "Company Emails"),
    "Company_Phone_Numbers": (2, "Company Phone Numbers"),
    "LinkedIn": (2, "LinkedIn"),
    "Twitter": (2, "X (Twitter)"),
    "Facebook": (2, "Facebook"),
    "Blog_Url": (2, "Blog"),
    "Continent": (2, "Continent"),
    "Region": (2, "Region"),
    "dealroom_client_focus": (2, "Customer Focus"),
    "dealroom_patents_count": (2, "Number of Patents"),
}

default_cols = [col for col, (score, _) in column_rules.items() if score == 1]


# --------------------
# UI (only intro + input + single button; everything else background)
# --------------------
st.markdown(
    """
    <div style="text-align: center;">
        <h1>🪙 GoldenClue</h1>
        <p>Enter your investment thesis and discover matching startups.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Centered input
center_cols = st.columns([1,2,1])
with center_cols[1]:
    user_input = st.text_input(
        "Investment thesis",   # vagy valami rövid, de nem üres címke
        placeholder="e.g. AI in healthcare",
        label_visibility="collapsed"  # így nem látszik a label, de nem lesz üres
    )

if not user_input:
    st.stop()



# --------------------
# CORE FUNCTIONS
# --------------------
def run_vector_search(vector, limit=50, filters=None):
    where_clauses = []
    if filters:
        if filters.get("year"):
            years = ",".join(str(y) for y in filters["year"])
            where_clauses.append(f"full_table.Founded_Year IN ({years})")

        if filters.get("stage"):
            stages = ",".join(f"'{s}'" for s in filters["stage"])
            where_clauses.append(f"full_table.Company_Stage IN ({stages})")

        if filters.get("country"):
            parts = []
            for c in filters["country"]:
                safe_c = c.replace("'", "''")  # escape quotes
                parts.append(f"full_table.Country LIKE '%{safe_c}%'")
            countries = " OR ".join(parts)
            where_clauses.append(f"({countries})")

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

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
    {where_sql}
    ORDER BY cosine_distance DESC
    LIMIT {limit};
    """
    return pd.DataFrame(run_query(query))



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
        # prefer Description_merged if present, else Description
        desc = ""
        if "Description_merged" in df.columns:
            desc = getattr(row, "Description_merged")
        elif "Description" in df.columns:
            desc = getattr(row, "Description")
        items.append({
            "index": int(idx) if pd.notna(idx) else pos,
            "current_rank": pos,
            "Company_Name": str(name) if pd.notna(name) else "",
            "Description": str(desc) if pd.notna(desc) else ""
        })
    return items


def _make_rerank_prompt(thesis: str, items_json: str) -> str:
    return f"""Rerank the following startups STRICTLY by how well their Description matches this investment thesis:

THESIS:
{thesis}

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
ITEMS:
{items_json}
"""
def rerank_with_openai_model(df: pd.DataFrame, thesis: str, model_name: str) -> Dict[str, any]:
    items = _build_items_for_model(df)
    prompt = _make_rerank_prompt(thesis, json.dumps(items, ensure_ascii=False))

    # GPT-5 family doesn’t allow temperature=0
    supports_temperature = not model_name.startswith("gpt-5")

    t0 = time.perf_counter()
    kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a precise reranker. Follow the instructions exactly."},
            {"role": "user", "content": prompt}
        ]
    }
    if supports_temperature:
        kwargs["temperature"] = 0

    resp = openai_client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content.strip()
    parsed = _extract_json_object(content)
    elapsed = time.perf_counter() - t0

    if not isinstance(parsed, dict) or "order" not in parsed:
        raise ValueError(f"Model {model_name} response missing a valid 'order' list.")

    proposed_order = [int(x) for x in parsed["order"]]
    original_ids = [int(r["index"]) for r in items]

    if len(proposed_order) != len(original_ids) or set(proposed_order) != set(original_ids):
        inter = [i for i in proposed_order if i in set(original_ids)]
        missing = [i for i in original_ids if i not in set(inter)]
        proposed_order = inter + missing

    df_reordered = df.set_index("index").loc[proposed_order].reset_index()
    return {"df": df_reordered, "seconds": elapsed}



def rerank_with_claude_model(df, thesis: str, model_name: str):
    items = _build_items_for_model(df)
    prompt = _make_rerank_prompt(thesis, json.dumps(items, ensure_ascii=False))

    t0 = time.perf_counter()
    msg = anthropic_client.messages.create(
        model=model_name,
        max_tokens=500,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    # collect text blocks
    chunks = []
    for blk in msg.content:
        if getattr(blk, "type", None) == "text":
            chunks.append(blk.text)
        elif isinstance(blk, dict) and blk.get("type") == "text":
            chunks.append(blk.get("text", ""))
    content = "\n".join(chunks).strip()
    elapsed = time.perf_counter() - t0

    parsed = _extract_json_object(content)
    if not isinstance(parsed, dict) or "order" not in parsed:
        raise ValueError(f"Model {model_name} response missing a valid 'order' list.")

    proposed_order = [int(x) for x in parsed["order"]]
    original_ids = [int(i["index"]) for i in items]
    inter = [i for i in proposed_order if i in set(original_ids)]
    missing = [i for i in original_ids if i not in set(inter)]
    proposed_order = inter + missing

    df_reordered = df.set_index("index").loc[proposed_order].reset_index()
    return {"df": df_reordered, "seconds": elapsed}



def render_model_table(df: pd.DataFrame, model_label: str, selected_cols: set):
    st.subheader(model_label)
    df_to_show = build_display_df(df, selected_cols)

    # AgGrid config (same as yours)
    gb = GridOptionsBuilder.from_dataframe(df_to_show)
    gb.configure_default_column(sortable=True, resizable=True, filter=True, floatingFilter=True)
    gb.configure_side_bar()
    gb.configure_grid_options(
        suppressMenuHide=True,
        enableRangeSelection=True,        # drag across cells, copy with Ctrl+C
        enableCellTextSelection=False,     # allow text highlight & copy
        copyHeadersToClipboard=True,      # include headers when copying
        rowSelection="multiple"           # allow selecting multiple rows
    )
    if "Description" in df_to_show.columns:
        gb.configure_column(
            "Description",
            width=420,
            tooltipField="Description",  # hover still shows full text
            cellStyle={
                "whiteSpace": "normal",   # wrap words
                "overflow": "auto",       # scroll if content too long
                "textOverflow": "ellipsis",
                "lineHeight": "1.5em",    # line height
                "maxHeight": "3em"      # ~2 lines
            }
        )

    grid_options = gb.build()
    grid_options["rowHeight"] = 36

    AgGrid(
        df_to_show,
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED,
        update_mode=GridUpdateMode.MODEL_CHANGED,
        fit_columns_on_grid_load=False,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,   # needed for some copy/paste features
        key=f"aggrid_{model_label}",
    )


# --------------------
# SESSION STATE SETUP
# --------------------
if "base_results" not in st.session_state:
    st.session_state.base_results = {"stack": []}
if "has_run" not in st.session_state:
    st.session_state.has_run = False


# ... (your existing code above unchanged) ...



# Replace the old sidebar code with this
with st.expander("⚙️ Settings"):
    search_limit = st.number_input(
        "Max results to fetch",
        min_value=5,
        max_value=200,
        value=20,
        step=5
    )

    if "selected_cols" not in st.session_state:
        st.session_state.selected_cols = set(default_cols)

    selected_cols = []

    # Create 6 columns for the checkboxes
    cols = st.columns(6)

    for i, (col, (score, new_name)) in enumerate(column_rules.items()):
        label = new_name if new_name else col
        # Decide which column this checkbox should go into
        with cols[i % 6]:
            checked = st.checkbox(
                label,
                value=(col in st.session_state.selected_cols),
                key=f"chk_{col}"
            )
            if checked:
                selected_cols.append(col)

    st.session_state.selected_cols = set(selected_cols)

with st.expander("🔍 Pre-Filters"):
    if "filters" not in st.session_state:
        st.session_state.filters = {"year": [], "stage": [], "country": []}

    # ---- Founded Year checkboxes ----
    st.markdown("**Founded Year**")
    year_cols = st.columns(16)  # spread across 4 columns
    selected_years = []
    for i, y in enumerate(FOUNDED_YEARS):
        with year_cols[i % 16]:
            if st.checkbox(str(y), value=(y in st.session_state.filters["year"]), key=f"year_{y}"):
                selected_years.append(y)

    # ---- Company Stage checkboxes ----
    st.markdown("**Company Stage**")
    stage_cols = st.columns(10)
    selected_stages = []
    for i, s in enumerate(COMPANY_STAGES):
        with stage_cols[i % 10]:
            if st.checkbox(s, value=(s in st.session_state.filters["stage"]), key=f"stage_{s}"):
                selected_stages.append(s)
        
    # ---- Country + Continent checkboxes (with proper uncheck) ----
    st.markdown("**Country / Continent**")
    
    cc_cols = st.columns(10)
    
    # 1) Read continent toggles first
    continents = sorted(continent_map.keys())
    cont_checked = {}
    for i, cont in enumerate(continents):
        with cc_cols[i % 10]:
            cont_checked[cont] = st.checkbox(f"🌍 {cont}", key=f"cont_{cont}")
    
    # 2) Start from previously saved selection (so user manual picks persist)
    prev_selected = set(st.session_state.get("filters", {}).get("country", []))
    
    # 3) Apply continent toggles to derive the target set for this run
    target_selected = set(prev_selected)
    for cont, is_on in cont_checked.items():
        members = set(continent_map[cont])
        if is_on:
            target_selected |= members     # add all if continent is checked
        else:
            target_selected -= members     # remove all if continent is unchecked
    
    # 4) Push the computed selection into session_state BEFORE rendering country boxes
    for c in COUNTRIES:
        st.session_state[f"country_{c}"] = (c in target_selected)
    
    # 5) Render country checkboxes (reflecting the computed state)
    selected_countries = []
    for i, c in enumerate(COUNTRIES):
        with cc_cols[(i + len(continents)) % 10]:
            # value is taken from st.session_state[country_{c}] we set above
            if st.checkbox(c, key=f"country_{c}"):
                selected_countries.append(c)
    

    # ---- Save selections ----
    st.session_state.filters = {
        "year": selected_years,
        "stage": selected_stages,
        "country": selected_countries,
    }



# --------------------
# SINGLE ACTION BUTTON -> does everything, shows only final table
# --------------------
cols = st.columns([2,2,1])  # left: 1 part, middle: 2 parts, right: 1 part
with cols[1]:
    run = st.button("🔍 Find & Rerank Startups")

def build_display_df(base_df: pd.DataFrame, selected_cols: set) -> pd.DataFrame:
    cols_in_df = base_df.columns
    display_cols = []

    # If original_rank exists, always keep it first
    if "original_rank" in cols_in_df:
        display_cols.append("original_rank")

    # Keep only those in selected_cols, in the order of column_rules
    for col, (score, new_name) in column_rules.items():
        if col in cols_in_df and col in selected_cols:
            display_cols.append(col)

    # Always include score if present
    if "score" in cols_in_df and "score" not in display_cols:
        display_cols.append("score")


    # Fallback if nothing selected
    if not display_cols:
        fallback = [c for c, (score, _) in column_rules.items() if score == 1 and c in cols_in_df]
        display_cols = fallback or list(cols_in_df)[:10]

    # Slice dataframe
    df_out = base_df[display_cols].copy()

    # Apply renaming (only those with a friendly name defined)
    rename_map = {col: new for col, (_, new) in column_rules.items() if new}
    df_out.rename(columns=rename_map, inplace=True)

    return df_out


if run:
    st.session_state.has_run = True
    with st.spinner("Working..."):
        # --- enhance once ---
        default_prompt_template = (
            "You are an assistant helping to improve search queries for investment thesis matching.\n"
            "The user provided this query: \"{user_input}\"\n\n"
            "Rephrase the query in a way that makes it highly precise for vector search:\n"
            "- Preserve all niche / domain-specific details\n"
            "- Avoid generalization\n"
            "- Optimize for embeddings and retrieval of relevant companies\n"
            "- Keep it concise but information-rich\n"
        )
        st.text('Enhancing query')
        _, enhanced_query = enhance_query(user_input, default_prompt_template)

        # --- embed once ---
        st.text('Vectorization')
        enhanced_vector = openai_client.embeddings.create(
            input=[enhanced_query], model=model
        ).data[0].embedding

        # --- search once ---
        st.text('Searching database')
        df_enhanced = run_vector_search(
            enhanced_vector,
            limit=search_limit,
            filters=st.session_state.get("filters", None)
        )


        # --- optional score from cosine distances (computed from df_enhanced) ---
        score_series = None
        if "cosine_distance" in df_enhanced.columns:
            d = pd.to_numeric(df_enhanced["cosine_distance"], errors="coerce")
            if d.notna().any():
                d_min = float(d.min()); d_max = float(d.max())
                top_score = random.randint(95, 100)
                if d_max > d_min:
                    norm = (d - d_min) / (d_max - d_min)
                    score_series = ((top_score - 5) + norm * 5).round().astype(int)
                else:
                    score_series = pd.Series(top_score, index=d.index).astype(int)

        # --- models to test ---
        openai_models = ["gpt-4.1", "gpt-4o", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5-mini", "gpt-5-nano", "gpt-5"]
        claude_models = ["claude-opus-4-1-20250805", "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"]
        
        # dedupe while preserving order
        openai_models = list(dict.fromkeys(openai_models))
        claude_models = list(dict.fromkeys(claude_models))
        
        # Combine with provider tag in the order you want to show them
        model_plan = [("Claude", m) for m in claude_models] + [("OpenAI", m) for m in openai_models]
        
        # Prepare a section per model so you see “Queued…” immediately,
        # and then each fills in as soon as that model completes.
        st.markdown("### 🧪 Model Benchmarks (reranking only)")
        sections = {}  # (provider, model) -> {"container": container, "status": status_placeholder}
        
        for provider, model_name in model_plan:
            block = st.container()
            with block:
                st.subheader(f"{provider} · {model_name}")
                status = st.empty()   # will flip from Queued -> Running -> Done/Failed
                status.info("Queued…")
            sections[(provider, model_name)] = {"container": block, "status": status}
        
        # We'll also keep an accumulating stack so re-renders (e.g., checkbox changes) keep results.
        if "base_results" not in st.session_state or not isinstance(st.session_state.base_results, dict):
            st.session_state.base_results = {"stack": []}
        
        # Optional overall progress bar
        progress = st.progress(0)
        done = 0
        total = len(model_plan)
        
        # Helper: add “score” column derived once from df_enhanced distances
        def _attach_score(df_r: pd.DataFrame) -> pd.DataFrame:
            if score_series is None:
                return df_r
            df_r = df_r.copy()
            try:
                # align by row order; if lengths mismatch, trim
                if score_series.index.equals(df_r.index):
                    df_r["score"] = score_series.loc[df_r.index].values
                else:
                    df_r["score"] = np.rint(score_series.values[:len(df_r)]).astype(int)
            except Exception:
                pass
            return df_r
        
        # Run models SEQUENTIALLY but render EACH as soon as it's done
        for provider, model_name in model_plan:
            slot = sections[(provider, model_name)]
            slot["status"].warning("Running…")
        
            try:
                if provider == "Claude":
                    out = rerank_with_claude_model(df=df_enhanced.copy(), thesis=user_input, model_name=model_name)
                else:
                    out = rerank_with_openai_model(df=df_enhanced.copy(), thesis=user_input, model_name=model_name)
        
                df_r = _attach_score(out["df"])
        
                # persist this model’s result so future reruns (e.g., checkbox toggles) can rebuild
                st.session_state.base_results["stack"].append((provider, model_name, df_r, out["seconds"]))
        
                # render IMMEDIATELY into this model's container
                with slot["container"]:
                    slot["status"].success(f"Done in {out['seconds']:.2f}s")
                    render_model_table(df_r, f"{provider} · {model_name}", st.session_state.selected_cols)
        
            except Exception as e:
                with slot["container"]:
                    slot["status"].error(f"Failed: {e}")
        
            # update overall progress
            done += 1
            progress.progress(int(done * 100 / total))

# ---- Re-render on sidebar changes (if user toggles columns later) ----
# If we already have results in session_state, reprint them below (keeps UI consistent after interaction)
if st.session_state.get("base_results", {}).get("stack"):
    st.markdown("### 🔁 Results (cached)")
    for provider, model_name, df_model, secs in st.session_state.base_results["stack"]:
        st.caption(f"{provider} · **{model_name}** — rerank time: {secs:.2f}s")
        render_model_table(df_model, f"{provider} · {model_name}", st.session_state.selected_cols)
