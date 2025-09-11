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

# Sidebar options that affect ONLY final output
st.sidebar.header("⚙️ Display Options")
search_limit = st.sidebar.number_input(
    "Max results to fetch",
    min_value=5,
    max_value=200,
    value=20,
    step=5
)

if "selected_cols" not in st.session_state:
    st.session_state.selected_cols = set(default_cols)

selected_cols = []
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("✅ Select All"):
        st.session_state.selected_cols = set(all_columns)
with col2:
    if st.button("❌ Clear All"):
        st.session_state.selected_cols = set()

for col in all_columns:
    checked = st.sidebar.checkbox(col, value=(col in st.session_state.selected_cols))
    if checked:
        selected_cols.append(col)
st.session_state.selected_cols = set(selected_cols)

# --------------------
# CORE FUNCTIONS
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
    gb.configure_grid_options(suppressMenuHide=True)
    if "Description_merged" in df_to_show.columns:
        gb.configure_column(
            "Description_merged",
            width=520,
            suppressSizeToFit=True,
            wrapText=True,
            autoHeight=False
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

# --------------------
# SINGLE ACTION BUTTON -> does everything, shows only final table
# --------------------
run = st.button("🔍 Find & Rerank Startups")

def build_display_df(base_df: pd.DataFrame, selected_cols: set) -> pd.DataFrame:
    cols_in_df = base_df.columns
    display_cols = []
    if "original_rank" in cols_in_df:
        display_cols.append("original_rank")
    display_cols += [
        c for c in all_columns
        if c in cols_in_df and c in selected_cols and c != "cosine_distance"
    ]
    if "score" in cols_in_df:
        display_cols.append("score")
    # if user selected none, fall back to some sane defaults
    if not display_cols:
        fallback = [c for c in default_cols if c in cols_in_df]
        display_cols = fallback or list(cols_in_df)[:10]
    return base_df[display_cols]


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
        df_enhanced = run_vector_search(enhanced_vector, limit=search_limit)

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
