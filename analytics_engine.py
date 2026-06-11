import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ─────────────────────────────────────────
# SCHEMA MAPPING
# ─────────────────────────────────────────
SCHEMA_MAP = {
    "fatalities": ["persons killed", "deaths", "fatalities", "killed", "death count"],
    "state":      ["state", "province", "region", "location"],
    "year":       ["year", "yr"],
    "mine":       ["mine", "project", "colliery", "mine name"],
    "category":   ["category name", "accident type", "category", "type"],
    "district":   ["district", "area", "zone"],
    "owner":      ["owner", "company", "operator"],
    "date":       ["date", "accident date", "incident date"],
}

def resolve_column(df, field):
    """Find actual column name from schema map."""
    aliases = SCHEMA_MAP.get(field, [field])
    for alias in aliases:
        for col in df.columns:
            if alias.lower() == col.lower().strip():
                return col
    return None


# ─────────────────────────────────────────
# DATA VALIDATION
# ─────────────────────────────────────────
def validate_dataframe(df):
    """Validate df before analytics."""
    issues = []
    if df is None or len(df) == 0:
        return False, ["DataFrame is empty or None"]
    if len(df.columns) < 2:
        issues.append("Too few columns for meaningful analysis")
    null_pct = df.isnull().mean()
    high_null = null_pct[null_pct > 0.8].index.tolist()
    if high_null:
        issues.append(f"Columns with >80% missing values: {high_null}")
    return len(issues) == 0, issues


def extract_fatality_count(value):
    """Extract numeric fatality count from string like '1. John, Male, 35 Years'."""
    if pd.isna(value):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    import re
    entries = re.findall(r'^\d+\.', str(value).strip(), re.MULTILINE)
    return len(entries) if entries else 1


# ─────────────────────────────────────────
# CORE ANALYTICS FUNCTIONS
# ─────────────────────────────────────────
def get_incident_count(df):
    """Total number of accident records."""
    return len(df)


def get_fatalities(df):
    """Total fatalities across all records."""
    fatality_col = resolve_column(df, "fatalities")
    if not fatality_col:
        return "Insufficient structured data available"
    total = df[fatality_col].apply(extract_fatality_count).sum()
    return int(total)


def get_average_fatalities(df):
    """Average fatalities per incident."""
    fatality_col = resolve_column(df, "fatalities")
    if not fatality_col:
        return "Insufficient structured data available"
    fatalities = df[fatality_col].apply(extract_fatality_count)
    incidents = len(df)
    if incidents == 0:
        return "Insufficient structured data available"
    avg = fatalities.sum() / incidents
    return round(avg, 4)


def get_severity_index(df):
    """Severity Index = Total Fatalities / Total Incidents"""
    fatalities = get_fatalities(df)
    incidents = get_incident_count(df)
    if isinstance(fatalities, str) or incidents == 0:
        return "Insufficient structured data available"
    return round(fatalities / incidents, 4)


def get_incident_by_category(df):
    """Incident count per category."""
    cat_col = resolve_column(df, "category")
    if not cat_col:
        return "Insufficient structured data available"
    return df[cat_col].value_counts().to_dict()


def get_fatalities_by_category(df):
    """Fatalities per category."""
    cat_col = resolve_column(df, "category")
    fatality_col = resolve_column(df, "fatalities")
    if not cat_col or not fatality_col:
        return "Insufficient structured data available"
    df = df.copy()
    df['_fat'] = df[fatality_col].apply(extract_fatality_count)
    return df.groupby(cat_col)['_fat'].sum().sort_values(ascending=False).to_dict()


def get_risk_scores(df):
    """Risk Score = Frequency × Severity per category."""
    cat_col = resolve_column(df, "category")
    fatality_col = resolve_column(df, "fatalities")
    if not cat_col or not fatality_col:
        return "Insufficient structured data available"
    df = df.copy()
    df['_fat'] = df[fatality_col].apply(extract_fatality_count)
    grouped = df.groupby(cat_col).agg(
        frequency=('_fat', 'count'),
        total_fatalities=('_fat', 'sum')
    )
    grouped['severity_index'] = grouped['total_fatalities'] / grouped['frequency']
    grouped['risk_score'] = (grouped['frequency'] * 0.5) + (grouped['severity_index'] * 0.5)
    return grouped.sort_values('risk_score', ascending=False).round(4).to_dict()


def get_yearly_trend(df):
    """Year-wise accident and fatality counts."""
    year_col = resolve_column(df, "year")
    fatality_col = resolve_column(df, "fatalities")
    if not year_col:
        return "Insufficient structured data available"
    df = df.copy()
    if fatality_col:
        df['_fat'] = df[fatality_col].apply(extract_fatality_count)
        trend = df.groupby(year_col).agg(
            incidents=(year_col, 'count'),
            fatalities=('_fat', 'sum')
        ).reset_index()
    else:
        trend = df.groupby(year_col).size().reset_index(name='incidents')
    return trend.sort_values(year_col).to_dict(orient='records')


def get_state_ranking(df):
    """Rank states by incident count and fatalities."""
    state_col = resolve_column(df, "state")
    fatality_col = resolve_column(df, "fatalities")
    if not state_col:
        return "Insufficient structured data available"
    df = df.copy()
    if fatality_col:
        df['_fat'] = df[fatality_col].apply(extract_fatality_count)
        ranking = df.groupby(state_col).agg(
            incidents=(state_col, 'count'),
            fatalities=('_fat', 'sum')
        ).sort_values('incidents', ascending=False)
    else:
        ranking = df.groupby(state_col).size().reset_index(name='incidents')
    return ranking.head(15).to_dict()


def get_correlation_matrix(df):
    """Correlation between numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return "Insufficient structured data available"
    return numeric_df.corr().round(3).to_dict()


def detect_anomalies(df):
    """Detect anomaly years using Z-score on incident counts."""
    year_col = resolve_column(df, "year")
    if not year_col:
        return "Insufficient structured data available"
    yearly = df.groupby(year_col).size().reset_index(name='count')
    yearly['zscore'] = np.abs(stats.zscore(yearly['count']))
    anomalies = yearly[yearly['zscore'] > 2.0]
    return anomalies.to_dict(orient='records')


# ─────────────────────────────────────────
# VISUALIZATION FUNCTIONS
# ─────────────────────────────────────────
def plot_trend(df, file_name=""):
    """Line chart — yearly trend."""
    trend = get_yearly_trend(df)
    if isinstance(trend, str):
        return None
    year_col = resolve_column(df, "year")
    if not year_col:
        return None
    trend_df = pd.DataFrame(trend)
    trend_df = trend_df.sort_values(year_col)
    fig = px.line(
        trend_df, x=year_col, y='incidents',
        title=f'Yearly Accident Trend — {file_name}',
        markers=True, color_discrete_sequence=['#FF6B6B']
    )
    return fig


def plot_category_bar(df, file_name=""):
    """Bar chart — incidents by category."""
    cat_col = resolve_column(df, "category")
    if not cat_col:
        return None
    counts = df[cat_col].value_counts().reset_index()
    counts.columns = ['Category', 'Count']
    fig = px.bar(
        counts, x='Count', y='Category',
        orientation='h',
        title=f'Accidents by Category — {file_name}',
        color='Count', color_continuous_scale='Reds'
    )
    return fig


def plot_risk_heatmap(df, file_name=""):
    """Heatmap — category vs year."""
    year_col = resolve_column(df, "year")
    cat_col = resolve_column(df, "category")
    if not cat_col or not year_col:
        return None
    pivot = df.groupby([cat_col, year_col]).size().unstack(fill_value=0)
    fig = px.imshow(
        pivot,
        title=f'Accident Heatmap (Category × Year) — {file_name}',
        color_continuous_scale='YlOrRd',
        aspect='auto'
    )
    return fig


def plot_state_map(df, file_name=""):
    """Bar chart — top states by incidents."""
    state_col = resolve_column(df, "state")
    if not state_col:
        return None
    df_clean = df[df[state_col].astype(str).str.match(r'^[A-Za-z\s]+$', na=False)]
    df_clean = df_clean[df_clean[state_col].str.len() < 50]
    counts = df_clean[state_col].value_counts().head(15).reset_index()
    counts.columns = ['State', 'Incidents']
    fig = px.bar(
        counts, x='Incidents', y='State',
        orientation='h',
        title=f'Top States by Incidents — {file_name}',
        color='Incidents', color_continuous_scale='Viridis'
    )
    return fig


# ─────────────────────────────────────────
# MASTER ANALYTICS SUMMARY
# ─────────────────────────────────────────
def plot_pie_chart(df, file_name="", field="category"):
    col = resolve_column(df, field)
    if not col:
        return None
    counts = df[col].value_counts().reset_index()
    counts.columns = [col, 'Count']
    if len(counts) > 8:
        top = counts.head(8)
        other = pd.DataFrame([{col: 'Other', 'Count': counts.iloc[8:]['Count'].sum()}])
        counts = pd.concat([top, other], ignore_index=True)
    fig = px.pie(
        counts,
        names=col,
        values='Count',
        title=f'{field.title()} Distribution — {file_name}',
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    return fig


def compute_full_analytics(df, file_name=""):
    """Compute all verified analytics for a dataframe."""
    valid, issues = validate_dataframe(df)
    if not valid:
        return {"error": f"Validation failed: {issues}"}

    return {
        "file_name": file_name,
        "incident_count": get_incident_count(df),
        "total_fatalities": get_fatalities(df),
        "average_fatalities_per_incident": get_average_fatalities(df),
        "severity_index": get_severity_index(df),
        "incidents_by_category": get_incident_by_category(df),
        "fatalities_by_category": get_fatalities_by_category(df),
        "risk_scores": get_risk_scores(df),
        "yearly_trend": get_yearly_trend(df),
        "state_ranking": get_state_ranking(df),
        "anomalies": detect_anomalies(df),
    }
