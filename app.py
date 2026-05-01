import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re

from dotenv import load_dotenv
from utils import load_csv, save_to_sqlite, clean_column_names

from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_groq import ChatGroq

# ---------------- LOAD ENV ---------------- #
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(page_title="AI SQL Analyst", layout="wide")
st.title("📊 AI SQL Data Analyst Agent")

# ---------------- MODEL LOADER ---------------- #
def get_llm(api_key):
    models = [
        "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768"
    ]
    
    for model in models:
        try:
            llm = ChatGroq(
                model=model,
                temperature=0,
                groq_api_key=api_key
            )
            return llm, model
        except Exception:
            continue
    
    raise Exception("❌ No working Groq model found")

# ---------------- FILE UPLOAD ---------------- #
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
question = st.text_input("Ask your question")

# ---------------- MAIN FLOW ---------------- #
if uploaded_file:
    # Load and clean data
    df = load_csv(uploaded_file)
    df = clean_column_names(df)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Save to SQLite
    engine = save_to_sqlite(df)
    db = SQLDatabase(engine)

    # Load LLM
    try:
        llm, model_used = get_llm(groq_api_key)
        st.success(f"Using model: {model_used}")
    except Exception as e:
        st.error(str(e))
        st.stop()

    db_chain = SQLDatabaseChain.from_llm(
        llm,
        db,
        verbose=False,
        return_intermediate_steps=True
    )

    # ---------------- QUERY ---------------- #
    if question:
        with st.spinner("🤖 Thinking..."):
            try:
                # -------- CUSTOM PROMPT -------- #
                custom_prompt = f"""
You are a SQL expert.

Convert the given question into a SQL query.

Rules:
- Use table name: data
- Return ONLY SQL query
- No explanation

Format:
SQLQuery: <your query>

Question: {question}
"""

                result = db_chain.invoke({"query": custom_prompt})

                st.subheader("📌 Answer")
                st.write(result["result"])

                # -------- SQL EXTRACTION -------- #
                sql_query = None

                # 1. Try LangChain intermediate steps
                steps = result.get("intermediate_steps", [])
                for step in steps:
                    if isinstance(step, dict) and "query" in step:
                        sql_query = step["query"]
                        break

                # 2. Extract from SQLQuery format
                if not sql_query:
                    match = re.search(
                        r"SQLQuery:\s*(SELECT .*)",
                        result["result"],
                        re.IGNORECASE | re.DOTALL
                    )
                    if match:
                        sql_query = match.group(1).strip()

                # 3. Generic fallback
                if not sql_query:
                    match = re.search(
                        r"(SELECT .* FROM .*?)(?:$|\n)",
                        result["result"],
                        re.IGNORECASE | re.DOTALL
                    )
                    if match:
                        sql_query = match.group(1).strip()

                # 4. Clean formatting
                if sql_query:
                    sql_query = (
                        sql_query
                        .replace("```sql", "")
                        .replace("```", "")
                        .strip()
                    )

                # -------- EXECUTION -------- #
                if sql_query:
                    st.subheader("🧾 Generated SQL")
                    st.code(sql_query, language="sql")

                    query_df = pd.read_sql(sql_query, engine)

                    if not query_df.empty:
                        st.subheader("📊 Visualization")

                        if len(query_df.columns) >= 2:
                            x_col = query_df.columns[0]
                            y_col = query_df.columns[1]

                            fig = px.bar(query_df, x=x_col, y=y_col)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Not enough columns for chart")

                        # -------- DOWNLOAD -------- #
                        st.download_button(
                            "📥 Download Results",
                            query_df.to_csv(index=False),
                            file_name="result.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No data returned")

                else:
                    st.warning("❌ Could not extract SQL query")

            except Exception as e:
                st.error(f"Error: {e}")