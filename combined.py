from analytics_engine import (
    compute_full_analytics,
    plot_trend, plot_category_bar,
    plot_risk_heatmap, plot_state_map, plot_pie_chart,
    get_risk_scores, get_severity_index,
    get_fatalities, get_incident_count
)

import streamlit as st
import pandas as pd
import numpy as np
import os
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud
import matplotlib
import random
from dotenv import load_dotenv
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from analytics_engine import (
    compute_full_analytics,
    plot_trend,
    plot_category_bar,
    plot_risk_heatmap,
    plot_state_map,
    plot_pie_chart,
    get_risk_scores,
    get_severity_index,
    get_fatalities,
    get_incident_count
)

warnings.filterwarnings("ignore")
matplotlib.use('Agg')

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from sentence_transformers import CrossEncoder
import markdown2

load_dotenv()

DEFAULT_GROQ_API_KEY = os.getenv("GROQ_API_KEY")


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

st.set_page_config(
    page_title="Mining Accident Analysis & RAG System",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="expanded"
)

if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'bm25_retriever' not in st.session_state:
    st.session_state.bm25_retriever = None
if 'vector_retriever' not in st.session_state:
    st.session_state.vector_retriever = None
if 'ensemble_retriever' not in st.session_state:
    st.session_state.ensemble_retriever = None
if 'reranker' not in st.session_state:
    st.session_state.reranker = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'rag_ready' not in st.session_state:
    st.session_state.rag_ready = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'loaded_embedding_model' not in st.session_state:
    st.session_state.loaded_embedding_model = None
if 'loaded_reranker_model' not in st.session_state:
    st.session_state.loaded_reranker_model = None
if 'loaded_groq_model' not in st.session_state:
    st.session_state.loaded_groq_model = None
if 'col_map' not in st.session_state:
    st.session_state.col_map = {}
if 'all_dfs' not in st.session_state:
    st.session_state.all_dfs = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def load_models(embedding_model, reranker_model):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        reranker = CrossEncoder(reranker_model)
        return embeddings, reranker
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def create_documents_from_csv(df, chunk_size=1000, chunk_overlap=200):
    documents = []
    try:
        for idx, row in df.iterrows():
            source_file = str(row.get('_source_file', 'unknown'))
            metadata = {"source": "csv", "row_id": idx, "source_file": source_file}
            content_parts = [f"Source File: {source_file}"]
            
            for col in df.columns:
                if col in ['Description', 'Suggestions']:
                    continue
                val = row[col]
                if not pd.isna(val):
                    content_parts.append(f"{col}: {val}")
                    metadata[col.lower().replace(' ', '_')] = str(val)
            
            desc = '' if pd.isna(row.get('Description', '')) else str(row.get('Description', ''))
            sugg = '' if pd.isna(row.get('Suggestions', '')) else str(row.get('Suggestions', ''))
            
            content_parts.append(f"Incident Description: {desc}")
            content_parts.append(f"Safety Suggestions: {sugg}")
            
            content = "\n".join(content_parts)
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    except Exception as e:
        st.error(f"Error creating documents: {str(e)}")
        return []

def build_retrievers(documents, embeddings, bm25_k=30, vector_k=30, bm25_weight=0.7):
    try:
        bm25_retriever = BM25Retriever.from_documents(documents, k=bm25_k)
        vectorstore = FAISS.from_documents(documents, embeddings)
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": vector_k})
        
        vector_weight = 1.0 - bm25_weight
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[bm25_weight, vector_weight]
        )
        
        return bm25_retriever, vector_retriever, ensemble_retriever
    except Exception as e:
        st.error(f"Error building retrievers: {str(e)}")
        return None, None, None

def rerank_documents(query, documents, reranker, top_k=5):
    if not documents or not reranker:
        return documents[:top_k]
    
    try:
        pairs = [(query, doc.page_content) for doc in documents]
        scores = reranker.predict(pairs)
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in doc_score_pairs[:top_k]]
    except Exception as e:
        st.warning(f"Reranking failed: {str(e)}")
        return documents[:top_k]

def setup_llm(groq_api_key, groq_model):
    try:
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=groq_model,
            temperature=0.7,
            max_tokens=2048
        )
        return llm
    except Exception as e:
        st.error(f"Error setting up LLM: {str(e)}")
        return None

AGGREGATION_TERMS = ['how many', 'most', 'count', 'total', 'frequent', 'repeated', 'worst']
DETAIL_TERMS = [
    'why', 'what happened', 'what caused', 'describe', 'who was involved',
    'circumstances', 'details', 'detail', 'explain', 'cause of', 'caused by'
]

def has_aggregation_intent(question):
    q = question.lower()
    return any(term in q for term in AGGREGATION_TERMS)

def has_detail_intent(question):
    q = question.lower()
    return any(term in q for term in DETAIL_TERMS)

def get_group_field_for_question(question, col_map):
    q = question.lower()
    if any(word in q for word in ['mine', 'colliery', 'site']):
        return 'mine', col_map.get('mine')
    if any(word in q for word in ['state', 'region', 'location', 'province']):
        return 'location', col_map.get('location')
    if any(word in q for word in ['category', 'type', 'cause']):
        return 'category', col_map.get('category')
    if 'district' in q:
        return 'district', col_map.get('district')
    if any(word in q for word in ['owner', 'company', 'operator']):
        return 'owner', col_map.get('owner')
    return 'mine', col_map.get('mine')

def find_mentioned_entities(question, df, col_map):
    q = question.lower()
    stop_tokens = {
        'mine', 'mines', 'colliery', 'accident', 'accidents', 'state',
        'category', 'owner', 'company', 'district', 'region', 'type'
    }
    matches = []
    for field in ['mine', 'location', 'district', 'category', 'owner']:
        col = col_map.get(field)
        if not col or col not in df.columns:
            continue
        values = df[col].dropna().astype(str).str.strip()
        for value in values.unique():
            value_lower = value.lower()
            if len(value_lower) < 3 or value_lower in ['nan', 'none', 'unknown']:
                continue
            value_tokens = [
                token for token in value_lower.replace('-', ' ').replace('_', ' ').split()
                if len(token) > 3 and token not in stop_tokens
            ]
            if value_lower in q:
                matches.append((field, col, value))
    return matches

def build_breakdown_text(filtered_df, col_map):
    parts = []
    for label, field in [('category', 'category'), ('state/location', 'location'), ('mine', 'mine')]:
        col = col_map.get(field)
        if col and col in filtered_df.columns:
            counts = filtered_df[col].dropna().astype(str).value_counts().head(5)
            if not counts.empty:
                formatted = ", ".join([f"{idx}: {int(val)}" for idx, val in counts.items()])
                parts.append(f"{label} breakdown: {formatted}")
    return "; ".join(parts)

def answer_aggregation_from_dataframe(question, df):
    col_map = detect_column_mapping(df)
    matches = find_mentioned_entities(question, df, col_map)
    q = question.lower()
    wants_fatalities = any(word in q for word in ['fatality', 'fatalities', 'death', 'deaths', 'killed'])
    fatality_col = col_map.get('fatality')

    if matches:
        mask = pd.Series(False, index=df.index)
        matched_labels = []
        for field, col, value in matches:
            mask = mask | df[col].astype(str).str.lower().str.contains(value.lower(), na=False, regex=False)
            matched_labels.append(f"{value} ({field})")
        filtered = df[mask]
        breakdown = build_breakdown_text(filtered, col_map)
        answer = f"Using the full dataset, count for {', '.join(dict.fromkeys(matched_labels))}: {len(filtered)} accident records."
        if wants_fatalities and fatality_col and fatality_col in filtered.columns:
            total_fatalities = int(filtered[fatality_col].apply(count_fatalities).sum())
            answer += f" Total fatalities: {total_fatalities}."
        if breakdown:
            answer += f" {breakdown}."
        return answer

    group_label, group_col = get_group_field_for_question(question, col_map)
    if wants_fatalities and fatality_col and fatality_col in df.columns and any(term in q for term in ['total', 'count', 'how many']):
        total_fatalities = int(df[fatality_col].apply(count_fatalities).sum())
        return f"Using the full dataset, total fatalities: {total_fatalities}."

    if group_col and group_col in df.columns and any(term in q for term in ['most', 'frequent', 'repeated', 'worst']):
        if wants_fatalities and fatality_col and fatality_col in df.columns:
            if wants_fatalities and fatality_col and fatality_col in df.columns:
                print(f"DEBUG: wants_fatalities={wants_fatalities}, fatality_col={fatality_col}")
            df['_fat_count'] = df[fatality_col].apply(count_fatalities)
            counts = df.groupby(group_col)['_fat_count'].sum().sort_values(ascending=False).head(10)
        else:
            counts = df[group_col].dropna().astype(str).value_counts().head(10)
        if counts.empty:
            return f"I found the {group_label} column, but there are no usable values to count."
        top_name = counts.index[0]
        top_count = int(counts.iloc[0])
        top_list = ", ".join([f"{idx}: {int(val)}" for idx, val in counts.items()])
        metric = "fatalities" if wants_fatalities else "accident records"
        return f"Using the full dataset, the {group_label} with the most {metric} is {top_name} with {top_count}. Top {metric}: {top_list}."

    if group_col and group_col in df.columns and any(term in q for term in ['count by', 'breakdown', 'what types', 'which types']):
        counts = df[group_col].dropna().astype(str).value_counts().head(10)
        if not counts.empty:
            formatted = ", ".join([f"{idx}: {int(val)}" for idx, val in counts.items()])
            return f"Using the full dataset, {group_label} counts are: {formatted}."

    return f"Using the full dataset, total accident records: {len(df)}."

def answer_question(question, ensemble_retriever, reranker, llm, rerank_top_k=5, full_df=None):
    """Answer counts from full data and incident details from RAG."""
    try:
        aggregate_answer = None
        use_aggregation = full_df is not None and has_aggregation_intent(question)
        use_rag = not use_aggregation or has_detail_intent(question)

        if use_aggregation:
            df_to_use = full_df
            if full_df is not None and '_source_file' in full_df.columns:
                files_in_df = full_df['_source_file'].unique()
                is_compare = any(word in question.lower() for word in ['compare', 'both', 'each', 'versus', 'vs'])
                matched_file = None
                if not is_compare:
                    for fname in files_in_df:
                        if fname.lower() in question.lower() or fname.lower().replace('.xlsx','').replace('.csv','') in question.lower():
                            matched_file = fname
                            break
                if matched_file:
                    df_to_use = full_df[full_df['_source_file'] == matched_file].copy()
                elif any(word in question.lower() for word in ['compare', 'both', 'each', 'versus', 'vs']):
                    answers = []
                    for fname in files_in_df:
                        fdf = full_df[full_df['_source_file'] == fname].copy()
                        ans = answer_aggregation_from_dataframe(question, fdf)
                        answers.append(f"{fname}: {ans}")
                    aggregate_answer = "\n".join(answers)
                    df_to_use = None
            if df_to_use is not None:
                aggregate_answer = answer_aggregation_from_dataframe(question, df_to_use)

        if not use_rag:
            return {
                "answer": aggregate_answer,
                "source_documents": [],
                "num_sources": 0
            }

        retrieved_docs = ensemble_retriever.invoke(question)
        reranked_docs = rerank_documents(question, retrieved_docs, reranker, rerank_top_k)
        
        context = "\n\n".join([
    f"Document {i+1}: {doc.page_content[:1500]}"
    for i, doc in enumerate(reranked_docs)
    ])
        
        prompt_template = ChatPromptTemplate.from_template("""
You are an expert mining safety analyst analyzing historical mining accident records.

Context retrieved from accident database:
{context}

Question: {question}

Instructions:
- Answer ONLY using EXACT information present in the context above
- Do NOT mix up job roles with activities — job role is only what 
  is listed after the person's name in "Persons Killed" field
- NEVER infer or guess details not explicitly stated in the context
- If unsure about any detail, say "not mentioned in retrieved records"
- Include specific details: mine names, locations, dates, death counts wherever available
- If numbers are present in context, always mention them
- Do not answer count, total, most, or frequency questions from retrieved documents.
- If the user asked for counts, totals, most frequent items, or other aggregations, those are provided separately from the full dataset.
- Do NOT make up any information
- Each document has a "Source File:" field at the top — ALWAYS mention 
  the source file name when answering questions
- When multiple source files exist in context, explicitly compare them 
  by file name and highlight any differences between files
- NEVER infer or guess details not explicitly in the context
- If unsure, say "not mentioned in retrieved records"

Answer:
""")
        
        formatted_prompt = prompt_template.format(context=context, question=question)
        answer = llm.invoke(formatted_prompt)
        final_answer = answer.content
        if aggregate_answer:
            final_answer = f"{aggregate_answer}\n\nDetails from retrieved matching records:\n{final_answer}"
        
        return {
            "answer": final_answer,
            "source_documents": reranked_docs,
            "num_sources": len(reranked_docs)
        }
    except Exception as e:
        return {"error": f"Error answering question: {str(e)}"}

def perform_chi_squared_test(df):
    try:
        categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].nunique() < 15:
                categorical_cols.append(col)
        categorical_cols = list(dict.fromkeys(categorical_cols))
        
        if len(categorical_cols) < 2:
            return None, None
        
        n_cols = len(categorical_cols)
        p_values = np.ones((n_cols, n_cols))
        chi2_stats = np.zeros((n_cols, n_cols))
        
        for i, col1 in enumerate(categorical_cols):
            for j, col2 in enumerate(categorical_cols):
                if i != j:
                    try:
                        contingency_table = pd.crosstab(df[col1].fillna('Unknown'), 
                                                      df[col2].fillna('Unknown'))
                        
                        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                        p_values[i, j] = p_value
                        chi2_stats[i, j] = chi2
                    except Exception as e:
                        p_values[i, j] = 1.0
                        chi2_stats[i, j] = 0.0
        
        p_values_df = pd.DataFrame(p_values, 
                                  index=categorical_cols, 
                                  columns=categorical_cols)
        
        return p_values_df, categorical_cols
    except Exception as e:
        st.warning(f"Error in chi-squared test: {str(e)}")
        return None, None

def perform_clustering_analysis(df):
    try:
        # Select numerical columns for clustering
        numerical_cols = []
        
        # Try to get numerical data
        for col in ['Year', 'Death_Count', 'Persons Killed', 'Other_People_Involved', 'Age']:
            try:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():
                        numerical_cols.append(col)
            except:
                continue
        
        if len(numerical_cols) < 2:
            return None, None, None, None
        
        
        cluster_data = df[numerical_cols].dropna()
        
        if len(cluster_data) < 10:
            return None, None, None, None
        
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)
        
        n_clusters = min(5, len(cluster_data) // 5)  # Ensure reasonable cluster size
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        
        cluster_results = cluster_data.copy()
        cluster_results['Cluster'] = cluster_labels
        cluster_results['PCA1'] = pca_data[:, 0]
        cluster_results['PCA2'] = pca_data[:, 1]
        
        return cluster_results, kmeans, pca, numerical_cols
    except Exception as e:
        st.warning(f"Error in clustering analysis: {str(e)}")
        return None, None, None, None

def count_fatalities(val):
    """Count fatalities from numeric or numbered text fields."""
    try:
        if pd.isna(val):
            return 0
        if isinstance(val, (int, float, np.integer, np.floating)):
            return int(float(val))
        text = str(val).strip()
        if not text:
            return 0
        try:
            return int(float(text))
        except:
            import re
            numbered_entries = re.findall(r'^\s*\d+\.', text, re.MULTILINE)
            return len(numbered_entries) if numbered_entries else 1
    except:
        return 0
    
def detect_column_mapping(df):
    col_lower = {col.lower().strip(): col for col in df.columns}

    def find_col(keywords):
        for kw in keywords:
            for col_l, col_orig in col_lower.items():
                if kw in col_l:
                    return col_orig
        return None

    return {
        'date':        find_col(['accident date', 'incident date', 'date', 'occurred']),
        'year':        find_col(['year', 'yr']),
        'location':    find_col(['state', 'region', 'province', 'location', 'territory', 'area']),
        'district':    find_col(['district', 'county', 'lga', 'shire', 'municipality']),
        'mine':        find_col(['mine', 'site', 'pit', 'colliery', 'facility', 'operation']),
        'owner':       find_col(['owner', 'company', 'operator', 'employer', 'organisation', 'organization']),
        'fatality':    find_col(['death', 'fatal', 'kill', 'dead', 'casualt', 'deceased']),
        'description': find_col(['description', 'narrative', 'detail', 'summary', 'notes', 'incident']),
        'suggestions': find_col(['suggestion', 'recommendation', 'prevention', 'control', 'action']),
        'gender':      find_col(['gender', 'sex']),
        'age':         find_col(['age']),
        'job':         find_col(['job', 'occupation', 'role', 'position', 'trade']),
        'shift':       find_col(['shift', 'period', 'session']),
        'category':    find_col(['category name', 'accident type', 'category name', 'type name']) or find_col(['category', 'type', 'class', 'kind']),
    }

def detect_year_column(df, col_map=None):
    col_map = col_map or detect_column_mapping(df)
    year_col = col_map.get('year')
    if year_col in df.columns:
        return year_col
    for candidate in ['Year', 'year', 'Yr', 'yr']:
        if candidate in df.columns:
            return candidate
    date_col = col_map.get('date')
    if date_col in df.columns:
        return date_col
    return None

def get_yearly_fatality_summary(df, col_map=None):
    col_map = col_map or detect_column_mapping(df)
    year_col = detect_year_column(df, col_map)
    fatality_col = col_map.get('fatality')
    if not year_col or year_col not in df.columns:
        return None, "I could not find a year/date column in this dataset."
    if not fatality_col or fatality_col not in df.columns:
        return None, "I could not find a fatality/death column in this dataset."

    temp = df[[year_col, fatality_col]].copy()
    if pd.api.types.is_datetime64_any_dtype(temp[year_col]):
        temp['Year'] = temp[year_col].dt.year
    else:
        numeric_years = pd.to_numeric(temp[year_col], errors='coerce')
        plausible_years = numeric_years.where(numeric_years.between(1800, 2200))
        parsed_dates = pd.to_datetime(temp[year_col], errors='coerce', dayfirst=True)
        temp['Year'] = plausible_years.fillna(parsed_dates.dt.year)

    temp['Fatalities'] = temp[fatality_col].apply(count_fatalities)
    yearly = (
        temp.dropna(subset=['Year'])
        .assign(Year=lambda x: x['Year'].astype(int))
        .groupby('Year', as_index=False)['Fatalities']
        .sum()
        .sort_values('Year')
    )
    if yearly.empty:
        return None, "I found the columns, but there is no usable yearly fatality data."
    return yearly, None

def answer_analysis_question_deterministic(question, all_dfs):
    q_lower = question.lower()
    wants_death = any(w in q_lower for w in ['death', 'fatal', 'killed', 'casualty', 'casualties'])
    wants_year = any(w in q_lower for w in ['year', 'annum'])
    wants_high = any(w in q_lower for w in ['highest', 'maximum', 'max', 'most', 'worst'])
    wants_low = any(w in q_lower for w in ['lowest', 'minimum', 'min', 'least'])

    if wants_death and wants_year and (wants_high or wants_low):
        answers = []
        for fname, fdf in all_dfs.items():
            yearly, error = get_yearly_fatality_summary(fdf, detect_column_mapping(fdf))
            if error:
                answers.append(f"{fname}: {error}")
                continue
            max_row = yearly.loc[yearly['Fatalities'].idxmax()]
            min_row = yearly.loc[yearly['Fatalities'].idxmin()]
            total = int(yearly['Fatalities'].sum())
            parts = [f"{fname}:"]
            if wants_high:
                parts.append(f"highest fatalities were in {int(max_row['Year'])} with {int(max_row['Fatalities'])} deaths")
            if wants_low:
                parts.append(f"lowest fatalities were in {int(min_row['Year'])} with {int(min_row['Fatalities'])} deaths")
            answers.append("; ".join(parts) + f". Total deaths across {yearly['Year'].nunique()} years: {total}.")
        return "\n\n".join(answers), True

    return None, False

def get_chart_intent(question):
    q = question.lower()
    chart_words = ['show', 'plot', 'chart', 'graph', 'visualize', 'visualise', 'pie', 'heatmap', 'trend']
    if not any(word in q for word in chart_words):
        return None
    if 'pie' in q:
        if any(word in q for word in ['state', 'region', 'location', 'province']):
            return ('pie', 'state')
        return ('pie', 'category')
    if 'heatmap' in q:
        return ('heatmap', None)
    if 'line chart' in q or 'trend' in q:
        return ('trend', None)
    if 'by state' in q or ('bar chart' in q and any(word in q for word in ['state', 'region', 'location', 'province'])):
        return ('state_bar', None)
    if 'by category' in q or 'bar chart' in q or ('chart' in q and any(word in q for word in ['category', 'type', 'cause'])):
        return ('category_bar', None)
    return None

def select_chart_dfs(question, all_dfs):
    q = question.lower()
    exact_filename_matches = {
        fname: fdf for fname, fdf in all_dfs.items()
        if fname.lower() in q
    }
    if exact_filename_matches:
        return exact_filename_matches

    exact_stem_matches = {
        fname: fdf for fname, fdf in all_dfs.items()
        if os.path.splitext(fname.lower())[0] in q
    }
    if exact_stem_matches:
        return exact_stem_matches

    mentioned = {}
    for fname, fdf in all_dfs.items():
        name = fname.lower()
        stem = os.path.splitext(name)[0]
        tokens = [tok for tok in stem.replace('_', ' ').replace('-', ' ').split() if len(tok) > 2]
        if name in q or stem in q or any(tok in q for tok in tokens):
            mentioned[fname] = fdf
    return mentioned if mentioned else all_dfs

def render_requested_analysis_chart(question, all_dfs):
    intent = get_chart_intent(question)
    if not intent:
        return False
    chart_type, field = intent
    selected_dfs = select_chart_dfs(question, all_dfs)
    for fname, fdf in selected_dfs.items():
        fig = None
        if chart_type == 'pie':
            fig = plot_pie_chart(fdf, fname, field=field)
        elif chart_type == 'trend':
            fig = plot_trend(fdf, fname)
        elif chart_type == 'heatmap':
            fig = plot_risk_heatmap(fdf, fname)
        elif chart_type == 'state_bar':
            fig = plot_state_map(fdf, fname)
        elif chart_type == 'category_bar':
            fig = plot_category_bar(fdf, fname)

        if fig:
            st.plotly_chart(fig, use_container_width=True, key=f"analysis_{chart_type}_{fname}")
        else:
            st.info(f"{fname}: requested chart could not be created from the available columns.")
    return True

def convert_report_to_pdf(markdown_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            rightMargin=inch, leftMargin=inch,
                            topMargin=inch, bottomMargin=inch)
    styles = getSampleStyleSheet()
    story = []
    for line in markdown_text.split('\n'):
        line = line.strip()
        if not line:
            story.append(Spacer(1, 0.15 * inch))
        elif line.startswith('# '):
            story.append(Paragraph(line[2:], styles['Heading1']))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], styles['Heading2']))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], styles['Heading3']))
        else:
            clean = line.replace('**','').replace('*','').replace('`','').replace('|','')
            if clean.strip():
                story.append(Paragraph(clean, styles['Normal']))
    doc.build(story)
    return buffer.getvalue()

def row_to_text(row):
    try:
        text_parts = []
        for col in row.index:
            try:
                if pd.notna(row[col]):
                    text_parts.append(f"{col}: {row[col]}")
            except:
                continue
        return "\n".join(text_parts)
    except:
        return ""

def generate_safety_audit_report_with_calculations(df, groq_api_key, groq_model):
    
    try:
        
        def extract_causes(description):
            if pd.isna(description):
                return []
            
            causes = []
            description = str(description).lower()
            
            cause_patterns = {
                'roof collapse': ['roof', 'collapse', 'fall', 'caving'],
                'explosion': ['explosion', 'blast', 'detonation'],
                'fire': ['fire', 'burning', 'flame'],
                'flooding': ['flood', 'water', 'inundation'],
                'equipment failure': ['equipment', 'machinery', 'mechanical'],
                'electrical': ['electrical', 'electrocution', 'electric'],
                'gas leak': ['gas', 'methane', 'carbon monoxide'],
                'transportation': ['transport', 'vehicle', 'truck', 'conveyor'],
                'human error': ['negligence', 'human error', 'mistake'],
                'structural failure': ['structural', 'building', 'wall']
            }
            
            for cause, keywords in cause_patterns.items():
                if any(keyword in description for keyword in keywords):
                    causes.append(cause)
            
            return causes if causes else ['other']
        
        # Data preprocessing and calculations
        try:
            df['Fatalities'] = df['Death_Count'].apply(count_fatalities)
        except:
            df['Fatalities'] = 0
            
        # Clean state and district data
        try:
            df['State'] = df['State'].astype(str).str.strip()
            df['District'] = df['District'].astype(str).str.strip()
            df['Owner'] = df['Owner'].astype(str).str.strip()
        except:
            pass
        
        # Fix common typos
        fix_typos = {
            "Chhattisgar h": "Chhattisgarh",
            "Telangan a": "Telangana", 
            "Maharasht ra": "Maharashtra"
        }
        
        try:
            df['State'] = df['State'].replace(fix_typos)
        except:
            pass
        
        # Extract causes from descriptions
        try:
            df['Causes'] = df['Description'].apply(extract_causes)
        except:
            df['Causes'] = [['other']] * len(df)
        
        # Flatten causes for analysis
        all_causes = []
        for causes_list in df['Causes']:
            all_causes.extend(causes_list)
        
        from collections import Counter
        cause_counts = Counter(all_causes)
        
        # CALCULATE KEY METRICS
        total_incidents = len(df)
        total_fatalities = df['Fatalities'].sum()
        
        try:
            date_range = f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
            years_span = (df['Date'].max() - df['Date'].min()).days / 365.25
        except:
            date_range = "Date range unavailable"
            years_span = 1
        
        fatality_rate = total_fatalities / total_incidents if total_incidents > 0 else 0
        avg_incidents_per_year = total_incidents / years_span if years_span > 0 else 0
        avg_fatalities_per_year = total_fatalities / years_span if years_span > 0 else 0
        
        zero_fatality_incidents = len(df[df['Fatalities'] == 0])
        zero_fatality_rate = zero_fatality_incidents / total_incidents * 100 if total_incidents > 0 else 0
        
        high_fatality_incidents = len(df[df['Fatalities'] > 5])
        high_fatality_rate = high_fatality_incidents / total_incidents * 100 if total_incidents > 0 else 0
        
        # GEOGRAPHIC ANALYSIS WITH CALCULATIONS
        try:
            state_analysis = df.groupby('State').agg({
                'Date': 'count',
                'Fatalities': 'sum',
                'Owner': 'nunique'
            }).rename(columns={'Date': 'Incidents', 'Owner': 'Unique_Owners'})
            state_analysis['Fatality_Rate'] = state_analysis['Fatalities'] / state_analysis['Incidents']
            state_analysis['Incidents_per_Owner'] = state_analysis['Incidents'] / state_analysis['Unique_Owners']
            state_analysis['Risk_Score'] = (
                state_analysis['Fatality_Rate'] * 0.4 +
                (state_analysis['Incidents'] / state_analysis['Incidents'].max()) * 0.3 +
                (state_analysis['Incidents_per_Owner'] / state_analysis['Incidents_per_Owner'].max()) * 0.3
            )
            state_analysis = state_analysis.sort_values('Incidents', ascending=False)
        except:
            state_analysis = pd.DataFrame()
        
        # START BUILDING REPORT
        calculated_report = f"""
            # Mining Safety Audit Report - Comprehensive Analysis

            ## Executive Summary

            ### Key Performance Indicators
            - **Total Mining Incidents**: {total_incidents:,}
            - **Total Fatalities**: {total_fatalities:,}
            - **Reporting Period**: {date_range}
            - **Data Coverage**: {years_span:.1f} years
            - **Average Fatality Rate**: {fatality_rate:.2f} fatalities per incident
            - **Average Incidents per Year**: {avg_incidents_per_year:.1f}
            - **Average Fatalities per Year**: {avg_fatalities_per_year:.1f}

            ### Incident Severity Distribution
            - **Zero Fatality Incidents**: {zero_fatality_incidents:,} ({zero_fatality_rate:.1f}%)
            - **High Fatality Incidents (>5 deaths)**: {high_fatality_incidents:,} ({high_fatality_rate:.1f}%)

            ## 1. Geographic Analysis

            ### Critical Findings
            """

        if not state_analysis.empty:
            calculated_report += f"""
                - **Highest Risk State**: {state_analysis.loc[state_analysis['Risk_Score'].idxmax()].name} (Risk Score: {state_analysis['Risk_Score'].max():.2f})
                - **Most Incidents**: {state_analysis.iloc[0].name} with {state_analysis.iloc[0]['Incidents']:,} incidents
                - **Highest Fatality Rate**: {state_analysis.loc[state_analysis['Fatality_Rate'].idxmax()].name} ({state_analysis['Fatality_Rate'].max():.2f} fatalities per incident)

                ### Top 10 States by Incident Count
                | Rank | State | Incidents | Fatalities | Fatality Rate | Companies | Risk Score |
                |------|-------|-----------|------------|---------------|-----------|------------|
                """
            for i, (state, row) in enumerate(state_analysis.head(10).iterrows(), 1):
                calculated_report += f"| {i} | {state} | {row['Incidents']:,} | {row['Fatalities']:,} | {row['Fatality_Rate']:.2f} | {row['Unique_Owners']} | {row['Risk_Score']:.2f} |\n"
        
        calculated_report += f"""

            ## 2. Accident Causes Analysis

            ### Primary Causes of Mining Accidents
            | Cause | Frequency | Percentage |
            |-------|-----------|------------|
            """
        for cause, count in cause_counts.most_common(10):
            percentage = (count / len(all_causes)) * 100 if len(all_causes) > 0 else 0
            calculated_report += f"| {cause.title()} | {count:,} | {percentage:.1f}% |\n"
        
        calculated_report += """

            ## 3. Recommendations

            Based on the comprehensive analysis of mining accident data, the following recommendations are provided:

            1. **Enhanced Roof Support Systems**: Given the prevalence of roof collapse incidents, implement more robust systematic support rules
            2. **Improved Training Programs**: Focus on safety protocols for high-risk activities
            3. **Regular Safety Audits**: Increase frequency of safety inspections in high-risk areas
            4. **Technology Integration**: Deploy modern monitoring systems for early detection of hazardous conditions
            5. **Emergency Response**: Strengthen emergency response capabilities and protocols

            ---
            *Report generated with comprehensive statistical analysis and AI-powered insights*
            """
        
        return calculated_report

    except Exception as e:
        return f"Error generating enhanced safety audit report: {str(e)}"
def render_file_analysis(df, file_name, col_map):
    st.subheader(f"📄 Overview — {file_name}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        try:
            date_col = col_map.get('date')
            if date_col and date_col in df.columns:
                st.metric("Date Range", f"{df[date_col].min().year} to {df[date_col].max().year}")
            else:
                st.metric("Date Range", "N/A")
        except:
            st.metric("Date Range", "N/A")

    st.dataframe(df.head())

    try:
        fatality_col = col_map.get('fatality')
        if fatality_col and fatality_col in df.columns:
            df["Fatalities"] = df[fatality_col].apply(count_fatalities)
        else:
            df["Fatalities"] = 0
    except:
        df["Fatalities"] = 0

    try:
        location_col = col_map.get('location')
        if location_col and location_col in df.columns:
            df[location_col] = df[location_col].astype(str).str.strip()
            df_cleaned = df[df[location_col].str.match(r'^[A-Za-z\s]+$', na=False)]
            df_cleaned = df_cleaned[df_cleaned[location_col].str.len() < 50]
        else:
            df_cleaned = df
    except:
        df_cleaned = df

    st.subheader("📈 Visual Analysis")

    try:
        location_col = col_map.get('location')
        if location_col and location_col in df_cleaned.columns:
            loc_counts = df_cleaned[location_col].value_counts().reset_index()
            loc_counts.columns = [location_col, 'Number of Accidents']
            fig_loc = px.bar(loc_counts.sort_values('Number of Accidents'),
                x='Number of Accidents', y=location_col, orientation='h',
                title=f'Accidents per {location_col}',
                color='Number of Accidents', color_continuous_scale='Viridis')
            st.plotly_chart(fig_loc, use_container_width=True, key=f"{file_name}_loc")
    except Exception as e:
        st.warning(f"Location chart error: {e}")

    try:
        if 'Owner' in df.columns:
            df_owner = df.copy()
            df_owner['Owner'] = df_owner['Owner'].astype(str).fillna("Unknown")
            df_owner = df_owner[df_owner['Owner'].str.lower() != 'nan']
            fig_owner = px.pie(df_owner, names='Owner', title='Accidents by Owner')
            st.plotly_chart(fig_owner, use_container_width=True, key=f"{file_name}_owner")
    except Exception as e:
        st.warning(f"Owner chart error: {e}")

    try:
        if 'Date' in df.columns and 'Fatalities' in df.columns:
            fatal_by_month = df.groupby(df['Date'].dt.to_period("M"))['Fatalities'].sum().reset_index()
            fatal_by_month['Date'] = fatal_by_month['Date'].astype(str)
            fig_trend = px.line(fatal_by_month, x='Date', y='Fatalities', title='Monthly Fatalities Trend')
            st.plotly_chart(fig_trend, use_container_width=True, key=f"{file_name}_trend")
    except Exception as e:
        st.warning(f"Trend chart error: {e}")

    try:
        if 'District' in df.columns:
            top_districts = df['District'].value_counts().nlargest(10)
            fig_district = px.bar(x=top_districts.index, y=top_districts.values,
                title="Top 10 Districts", labels={"x": "District", "y": "Incidents"})
            st.plotly_chart(fig_district, use_container_width=True, key=f"{file_name}_district")
    except Exception as e:
        st.warning(f"District chart error: {e}")

    try:
        if 'Category Name' in df.columns:
            category_counts = df['Category Name'].value_counts().head(10)
            fig_category = px.pie(values=category_counts.values,
                names=category_counts.index, title="Top 10 Accident Categories")
            st.plotly_chart(fig_category, use_container_width=True, key=f"{file_name}_category")
    except Exception as e:
        st.warning(f"Category chart error: {e}")
        
        st.subheader("🔥 Categorical Association Analysis (Chi-squared Test P-values)")
    p_values_df, categorical_cols = perform_chi_squared_test(df)
    if p_values_df is not None:
        try:
            fig_chi2, ax_chi2 = plt.subplots(figsize=(12, 10))
            sns.heatmap(p_values_df, annot=True, fmt='.3f', cmap='RdYlBu_r',
                cbar_kws={'label': 'P-value'}, ax=ax_chi2, vmin=0, vmax=1)
            ax_chi2.set_title('Chi-squared Test P-values\n(Lower = stronger association)', fontsize=14)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig_chi2)
            plt.close(fig_chi2)
        except Exception as e:
            st.warning(f"Chi-squared error: {e}")
    else:
        st.info("Not enough categorical data for chi-squared analysis")
        # Risk & Severity Analytics
    st.subheader("⚠️ Risk & Severity Analytics")
    try:
        analytics = compute_full_analytics(df, file_name)
        if "error" not in analytics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Incidents", analytics['incident_count'])
            with col2:
                st.metric("Total Fatalities", analytics['total_fatalities'])
            with col3:
                st.metric("Avg Fatalities/Incident", analytics['average_fatalities_per_incident'])
            with col4:
                st.metric("Severity Index", analytics['severity_index'])

            st.subheader("🎯 Category Risk Ranking")
            risk = analytics['risk_scores']
            if isinstance(risk, dict) and 'frequency' in risk:
                risk_df = pd.DataFrame(risk).T.reset_index()
                risk_df.columns = ['Category', 'Frequency', 'Total Fatalities', 'Severity Index', 'Risk Score']
                risk_df = risk_df.sort_values('Risk Score', ascending=False)
                st.dataframe(risk_df, use_container_width=True)

            fig_trend = plot_trend(df, file_name)
            if fig_trend:
                st.plotly_chart(fig_trend, use_container_width=True, key=f"{file_name}_risk_trend")

            fig_heatmap = plot_risk_heatmap(df, file_name)
            if fig_heatmap:
                st.subheader("🔥 Accident Heatmap (Category × Year)")
                st.plotly_chart(fig_heatmap, use_container_width=True, key=f"{file_name}_heatmap")

            anomalies = analytics['anomalies']
            if isinstance(anomalies, list) and len(anomalies) > 0:
                st.subheader("🚨 Anomaly Years Detected")
                st.dataframe(pd.DataFrame(anomalies), use_container_width=True)
            else:
                st.info("No anomaly years detected in this dataset.")
    except Exception as e:
        st.warning(f"Risk analytics error: {e}")
        
def main():
    st.title("⚒️ Mining Accident Analysis & RAG System")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Data Analysis", 
        "🔍 Hybrid RAG System", 
        "📋 Safety Audit Report", 
        "ℹ️ About",
        "📥 Get Datasets"
    ])

    with st.sidebar:
        st.header("📂 Data Upload")
        uploaded_files = st.file_uploader(
            "Upload Mining Data Files (max 6)", 
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if len(uploaded_files) > 6:
                st.error("❌ Maximum 6 files allowed!")
            else:
                all_dfs = {}
                for uploaded_file in uploaded_files:
                    try:
                        if uploaded_file.name.endswith(('.xlsx', '.xls')):
                            df = pd.read_excel(uploaded_file, engine="openpyxl")
                        else:
                            df = pd.read_csv(uploaded_file, engine="python")

                        df.columns = [col.strip() for col in df.columns]
                        df['_source_file'] = uploaded_file.name

                        col_map_temp = detect_column_mapping(df)
                        detected_date = col_map_temp.get('date')
                        try:
                            if detected_date and detected_date in df.columns:
                                df[detected_date] = pd.to_datetime(df[detected_date], errors='coerce', dayfirst=True)
                                df = df[df[detected_date].notna()]
                        except:
                            pass

                        df = df.convert_dtypes()
                        all_dfs[uploaded_file.name] = df
                        st.success(f"✅ {uploaded_file.name}: {len(df)} records")

                    except Exception as e:
                        st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")

                if all_dfs:
                    st.session_state.all_dfs = all_dfs
                    combined_df = pd.concat(all_dfs.values(), ignore_index=True)
                    st.session_state.df = combined_df
                    col_map = detect_column_mapping(combined_df)
                    st.session_state.col_map = col_map
                    st.info(f"📊 Total records across all files: {len(combined_df)}")

        st.header("⚙️ RAG Settings")
        groq_api_key = st.text_input("Groq API Key", value=DEFAULT_GROQ_API_KEY or "", type="password")
        st.subheader("Models")
        embedding_model = st.selectbox(
            "Embedding Model",
            ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"],
            index=0
        )
        reranker_model = st.selectbox(
            "Reranker Model",
            [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "cross-encoder/ms-marco-MiniLM-L-12-v2",
            ],
            index=0
        )
        groq_model = st.selectbox(
            "ChatGroq Model",
            [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
            ],
            index=0
        )
        chunk_size = 1000
        chunk_overlap = 200
        bm25_weight = 0.5
        rerank_top_k = 5

        if st.session_state.rag_ready:
            embedding_changed = st.session_state.loaded_embedding_model != embedding_model
            reranker_changed = st.session_state.loaded_reranker_model != reranker_model
            groq_changed = st.session_state.loaded_groq_model != groq_model

            if embedding_changed or reranker_changed:
                st.warning("⚠️ Embedding/Reranker changed! Click '🔧 Full Rebuild' in RAG tab.")
                st.session_state.rag_ready = False
            elif groq_changed:
                st.info("💡 Chat model changed! Click '⚡ Update Chat Model Only' in RAG tab.")

    # Tab1: Data Analysis
    with tab1:
        st.header("📊 Mining Incident Data Analysis")
        if st.session_state.df is not None:
            all_dfs = st.session_state.get('all_dfs', {})
            col_map = st.session_state.col_map

            if all_dfs and len(all_dfs) >= 1:
                tab_names = [f"📁 {name}" for name in all_dfs.keys()]
                tab_names.append("💬 Chat Analysis")
                file_tabs = st.tabs(tab_names)

                for i, (file_name, file_df) in enumerate(all_dfs.items()):
                    with file_tabs[i]:
                        file_col_map = detect_column_mapping(file_df)
                        render_file_analysis(file_df.copy(), file_name, file_col_map)

                with file_tabs[-1]:
                    st.subheader("💬 Ask Questions About Your Data")
                    st.info("Ask statistical questions — exact answers from full data!")

                    if 'analysis_chat' not in st.session_state:
                        st.session_state.analysis_chat = []

                    user_q = st.text_input("Your question:", key="analysis_question")
                    if st.button("🔍 Get Answer", key="analysis_btn"):
                        if user_q:
                            try:
                                deterministic_answer, _ = answer_analysis_question_deterministic(user_q, all_dfs)

                                if deterministic_answer:
                                    st.session_state.analysis_chat.append({
                                        "q": user_q,
                                        "a": deterministic_answer
                                    })
                                    render_requested_analysis_chart(user_q, all_dfs)
                                else:
                                    if not groq_api_key:
                                        st.warning("Please enter Groq API key for open-ended analysis questions. Exact statistical questions can still be answered without it.")
                                        st.stop()

                                    from langchain_groq import ChatGroq

                                    chat_llm = ChatGroq(
                                        groq_api_key=groq_api_key,
                                        model_name=groq_model
                                    )

                                    # Use verified analytics engine
                                    analytics_context = ""
                                    for fname, fdf in all_dfs.items():
                                        analytics = compute_full_analytics(fdf, fname)
                                        analytics_context += f"\n=== {fname} ===\n"
                                        if "error" in analytics:
                                            analytics_context += f"Analytics Error: {analytics['error']}\n"
                                            continue
                                        analytics_context += f"Incident Count: {analytics['incident_count']}\n"
                                        analytics_context += f"Total Fatalities: {analytics['total_fatalities']}\n"
                                        analytics_context += f"Average Fatalities Per Incident: {analytics['average_fatalities_per_incident']}\n"
                                        analytics_context += f"Severity Index: {analytics['severity_index']}\n"
                                        analytics_context += f"Incidents by Category: {analytics['incidents_by_category']}\n"
                                        analytics_context += f"Fatalities by Category: {analytics['fatalities_by_category']}\n"
                                        risk = analytics['risk_scores']
                                        if isinstance(risk, dict) and 'risk_score' in risk:
                                            analytics_context += f"Top Risk Categories: {dict(list(risk['risk_score'].items())[:5])}\n"
                                        trend = analytics['yearly_trend']
                                        if isinstance(trend, list):
                                            analytics_context += f"Yearly Trend: {trend}\n"
                                        state = analytics['state_ranking']
                                        if isinstance(state, dict) and 'incidents' in state:
                                            analytics_context += f"Top 5 States: {dict(list(state['incidents'].items())[:5])}\n"
                                        analytics_context += f"Anomalies: {analytics['anomalies']}\n"

                                    prompt = f"""You are an Advanced Mining Accident Analytics Engine.
Your task is to analyze structured mining accident datasets accurately using verified statistical computation principles.

CRITICAL RULES:
1. NEVER hallucinate calculations.
2. NEVER assume missing values.
3. NEVER confuse incident count, fatalities, category frequency, average fatalities, severity metrics.
4. If information is unavailable, explicitly say: "Insufficient structured data available."
5. All numerical analytics must come ONLY from the verified data statistics provided below.

ANALYTICS DEFINITIONS:
- Incident Count: Total number of accident records
- Fatalities: Total number of deaths across records
- Average Fatalities Per Incident: Total Fatalities / Total Incidents
- Severity Index: Total Fatalities / Incident Count
- Risk Score: Frequency x Severity

VERIFIED ANALYTICS DATA:
{analytics_context}

Question: {user_q}

Give a direct, precise answer using exact numbers from above. Show formula used where relevant.
CRITICAL: 
- The user may refer to a dataset by filename, country, or any keyword.
- Match their question to the most relevant === filename === block in the context above.
- If only one dataset is uploaded, use that one.
- NEVER say 'Insufficient structured data available' if Incident Count is present in context.
- Answer in one or two sentences only. No explanation of how you matched the dataset."""

                                    response = chat_llm.invoke(prompt)
                                    st.session_state.analysis_chat.append({
                                        "q": user_q,
                                        "a": response.content
                                    })

                                    render_requested_analysis_chart(user_q, all_dfs)

                            except Exception as e:
                                st.error(f"Error: {str(e)}")

                    for chat in reversed(st.session_state.analysis_chat):
                        st.markdown(f"**Q:** {chat['q']}")
                        st.markdown(f"**A:** {chat['a']}")
                        st.divider()

            else:
                single_df = list(all_dfs.values())[0] if all_dfs else st.session_state.df
                single_name = list(all_dfs.keys())[0] if all_dfs else "Uploaded File"
                render_file_analysis(single_df.copy(), single_name, col_map)

        else:
            st.info("Please upload an Excel/CSV file to begin analysis.")

    # Tab2: Hybrid RAG System
    with tab2:
        st.header("🔍 Hybrid RAG System")
        st.markdown("""
        **Hybrid RAG System** — sparse + dense retrieval with reranking:
        - **BM25** for keyword-based search
        - **FAISS Vector embeddings** for semantic search
        - **Cross-encoder reranking** for selecting most relevant documents
        - **ChatGroq LLM** for generating natural language answers
        - **Chat history** tracked throughout your session
        - 🔧 **Full Rebuild** — use when Embedding or Reranker model changes
        - ⚡ **Update Chat Model Only** — use when only Groq model changes (2-3 sec)
        """)

        st.info("""
💡 **What to ask here vs Data Analysis tab:**

✅ **Ask RAG** — specific incident details:
- "What happened at KUSUNDA mine?"
- "Describe a roof fall accident in Jharkhand"
- "What safety suggestions were given for transportation accidents?"
- "Tell me about accidents at Singareni Collieries"

📊 **Use Data Analysis tab instead** — statistics & counts:
- "Which state has most accidents?"
- "Total fatalities across all years?"
- "Which year was worst?"
- "How many accidents happened in Jharkhand?"
        """)
        if st.session_state.df is not None:
            df = st.session_state.df
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                if st.button("🔧 Full Rebuild", type="primary", use_container_width=True):
                    if not groq_api_key:
                        st.error("Please enter Groq API key!")
                    else:
                        with st.spinner("Building full RAG system... (30-60 sec)"):
                            embeddings, reranker = load_models(embedding_model, reranker_model)
                            if embeddings is None or reranker is None:
                                st.stop()
                            documents = create_documents_from_csv(df, chunk_size, chunk_overlap)
                            if not documents:
                                st.error("Failed to create documents")
                                st.stop()
                            bm25_ret, vector_ret, ensemble_ret = build_retrievers(
                                documents, embeddings, bm25_weight=bm25_weight
                            )
                            if ensemble_ret is None:
                                st.stop()
                            llm = setup_llm(groq_api_key, groq_model)
                            if llm is None:
                                st.stop()
                            st.session_state.documents = documents
                            st.session_state.bm25_retriever = bm25_ret
                            st.session_state.vector_retriever = vector_ret
                            st.session_state.ensemble_retriever = ensemble_ret
                            st.session_state.reranker = reranker
                            st.session_state.llm = llm
                            st.session_state.embeddings = embeddings
                            st.session_state.rag_ready = True
                            st.session_state.rerank_top_k = rerank_top_k
                            st.session_state.loaded_embedding_model = embedding_model
                            st.session_state.loaded_reranker_model = reranker_model
                            st.session_state.loaded_groq_model = groq_model
                            st.success(f"✅ Full RAG built with {len(documents)} documents!")

            with col_btn2:
                if st.button("⚡ Update Chat Model Only", use_container_width=True):
                    if not groq_api_key:
                        st.error("Please enter Groq API key!")
                    elif not st.session_state.rag_ready:
                        st.error("Please do Full Rebuild first!")
                    else:
                        with st.spinner("Updating chat model... (2-3 sec)"):
                            llm = setup_llm(groq_api_key, groq_model)
                            if llm is None:
                                st.stop()
                            st.session_state.llm = llm
                            st.session_state.loaded_groq_model = groq_model
                            st.success(f"✅ Chat model updated to: {groq_model}")
            if st.session_state.rag_ready:
                st.header("💬 Ask Questions")
                question = st.text_area(
                    "Your Question:",
                    value="",
                    height=100,
                    placeholder="Ask anything about your mining accident data..."
                )
                if st.button("🔍 Get Answer", type="primary") and question:
                    with st.spinner("Getting answer..."):
                        result = answer_question(
                            question,
                            st.session_state.ensemble_retriever,
                            st.session_state.reranker,
                            st.session_state.llm,
                            st.session_state.rerank_top_k,
                            full_df=st.session_state.df
                        )
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            st.session_state.chat_history.append({
                                "question": question,
                                "answer": result["answer"]
                            })
                            st.subheader("🤖 Answer")
                            st.write(result["answer"])

                            if result.get("source_documents"):
                                with st.expander("📄 View Source Documents Used"):
                                    for i, doc in enumerate(result["source_documents"]):
                                        st.markdown(f"**📌 Source {i+1}:**")
                                        st.text(doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content)
                                        st.divider()

                if st.session_state.chat_history:
                    st.subheader("💬 Chat History")
                    for chat in reversed(st.session_state.chat_history):
                        with st.expander(f"🗨️ {chat['question'][:80]}"):
                            st.markdown(f"**Answer:** {chat['answer']}")
                    if st.button("🗑️ Clear History"):
                        st.session_state.chat_history = []
                        st.rerun()
        else:
            st.info("Please upload an Excel/CSV file to begin using the RAG system.")

    # Tab3: Safety Audit Report
    with tab3:
        st.header("📋 Safety Audit Report Generation")
        if st.session_state.df is not None and groq_api_key:
            df = st.session_state.df
            if st.button("📊 Generate Safety Audit Report", type="primary"):
                with st.spinner("Generating comprehensive safety audit report..."):
                    report = generate_safety_audit_report_with_calculations(df, groq_api_key, groq_model)
                    st.subheader("📄 Safety Audit Report")
                    st.markdown(report)
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            label="📄 Download as Markdown",
                            data=report,
                            file_name="mining_safety_audit_report.md",
                            mime="text/markdown"
                        )
                    with col_dl2:
                        try:
                            pdf_bytes = convert_report_to_pdf(report)
                            st.download_button(
                                label="📥 Download as PDF",
                                data=pdf_bytes,
                                file_name="mining_safety_audit_report.pdf",
                                mime="application/pdf"
                            )
                        except Exception as e:
                            st.warning(f"PDF generation failed: {e}")
        else:
            if st.session_state.df is None:
                st.info("Please upload an Excel/CSV file to generate the safety audit report.")
            if not groq_api_key:
                st.info("Please enter your Groq API key to generate the safety audit report.")

    # Tab4: About
    with tab4:
        st.header("ℹ️ About This Application")
        st.markdown("""
        This application combines four powerful components for mining accident analysis from **any country's dataset**.

        ---

        ### 📂 **Universal Data Support**
        - Works with mining data from **any country** — India, Australia, USA, South Africa, and more
        - Auto-detects column roles (location, fatality, date, owner, etc.) regardless of column names
        - Supports Excel (.xlsx, .xls) and CSV file formats
        - Shows detected column mapping on upload so you know what was found

        ---

        ### 📊 **Data Analysis & Visualization**
        - Interactive charts: location-wise accidents, owner distribution, monthly trends
        - Age, gender, job role, shift-wise analysis (auto-detected from dataset)
        - Chi-squared test heatmap for categorical correlations (auto-detects categorical columns)
        - K-means clustering with PCA visualization (auto-detects numerical columns)
        - Word cloud generation for incident descriptions and safety suggestions

        ---

        ### 🔍 **Hybrid RAG System**
        - Combines **BM25** (keyword search) + **FAISS** (semantic search) for best retrieval
        - **Cross-Encoder reranking** for selecting most relevant documents
        - **ChatGroq LLM** for natural language answers
        - Shows **source documents** used for each answer — full transparency
        - **Chat history** — track all your previous questions in one session
        - **Smart rebuild system:**
            - 🔧 Full Rebuild — use when embedding or reranker model changes
            - ⚡ Update Chat Model Only — use when only Groq model changes (fast, 2-3 sec)
        - Automatic warning when model is changed after RAG is built

        ---

        ### 🤖 **Models Available**
        | Type | Option 1 | Option 2 |
        |------|----------|----------|
        | Embedding | all-MiniLM-L6-v2 (fast) | all-mpnet-base-v2 (better quality) |
        | Reranker | ms-marco-MiniLM-L-6-v2 | ms-marco-MiniLM-L-12-v2 (best quality) |
        | Chat LLM | llama-3.1-8b-instant (fast) | llama-3.1-70b-versatile (best quality) |

        ---

        ### 📋 **Safety Audit Report**
        - Auto-generates comprehensive report with KPIs, fatality trends, geographic analysis
        - Root cause analysis from incident descriptions
        - AI-written safety recommendations
        - Download as **Markdown** or **PDF** (professional format)

        ---

        **Built with:** Streamlit, LangChain, FAISS, BM25, CrossEncoder, ChatGroq (LLaMA), HuggingFace Embeddings, Plotly, Seaborn, WordCloud, Scikit-learn, SciPy, ReportLab
        """)

# Tab5: Get Datasets
    with tab5:
        st.header("📥 Sample Datasets")
        st.markdown("""
        Download the sample datasets below, upload them into this app, and start exploring.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🇮🇳 India Mining Accidents")
            st.markdown("""
            - **Records:** 337  
            - **Period:** 2016 – 2022  
            - **File:** `accidents.xlsx`
            """)
            st.link_button(
                "⬇️ Download India Dataset",
                "https://docs.google.com/spreadsheets/d/1Zc0ECyZeXScu5umT4MaBqehDCaBsdk1I/export?format=xlsx",
                use_container_width=True
            )

        with col2:
            st.subheader("🇦🇺 Australia Mining Accidents")
            st.markdown("""
            - **Records:** 350  
            - **Period:** 1882 – 2024  
            - **File:** `Australia_Mining_Accidents_350.xlsx`
            """)
            st.link_button(
                "⬇️ Download Australia Dataset",
                "https://docs.google.com/spreadsheets/d/1dQjdoePkBya-pfisfdGz4jWnZZwiVKVA/export?format=xlsx",
                use_container_width=True
            )

        st.divider()
        st.subheader("📋 How to use")
        st.markdown("""
        1. Click **Download** above for one or both datasets.
        2. Save the `.xlsx` file to your computer.
        3. Come back to the **📊 Data Analysis** tab.
        4. Upload the file using the **Data Upload** panel on the left sidebar.
        5. Explore charts, run RAG queries.
        """)

        st.info("💡 You can upload both files at once — the app supports multi-file analysis.")

if __name__ == "__main__":
    main()
