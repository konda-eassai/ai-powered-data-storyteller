# dashboard/utils.py
"""
Reusable helpers for EDA, modeling, plotting and simple insights.
Save as dashboard/utils.py
"""
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score, roc_curve, precision_recall_curve)
import joblib

sns.set(style="whitegrid")

# -----------------------
# I/O & load
# -----------------------
def find_csv(data_dir: Path = Path('../data')) -> Path:
    data_dir = Path(data_dir)
    files = list(data_dir.glob('*.csv'))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir.resolve()}")
    return files[0]

def load_data(path: Optional[Path|str|Any] = None) -> pd.DataFrame:
    if path is None:
        path = find_csv()
    if hasattr(path, "read"):
        return pd.read_csv(path)
    return pd.read_csv(path)

# -----------------------
# Summaries
# -----------------------
def basic_info(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "shape": df.shape,
        "n_rows": df.shape[0],
        "n_columns": df.shape[1],
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.apply(lambda x: str(x)).to_dict()
    }

def missing_values(df: pd.DataFrame) -> pd.Series:
    return df.isnull().sum().sort_values(ascending=False)

def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number]).describe().T

def categorical_value_counts(df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.Series]:
    out = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        out[col] = df[col].value_counts().head(top_n)
    return out

# -----------------------
# Correlation & plots
# -----------------------
def correlation_matrix(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if numeric_cols is None:
        num = df.select_dtypes(include=[np.number])
    else:
        num = df[numeric_cols]
    return num.corr()

def plot_correlation_matplotlib(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None,
                                figsize=(10,8), annot: bool = True, cmap: str = 'coolwarm') -> plt.Figure:
    corr = correlation_matrix(df, numeric_cols)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=annot, fmt='.2f', cmap=cmap, ax=ax)
    ax.set_title("Correlation matrix")
    return fig

def plot_correlation_plotly(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None):
    corr = correlation_matrix(df, numeric_cols)
    fig = px.imshow(corr, text_auto='.2f', aspect='auto', title='Correlation matrix')
    return fig

def plot_top_categories(df: pd.DataFrame, col: str, top_n: int = 10, kind: str = 'plotly'):
    vc = df[col].value_counts().nlargest(top_n)
    tmp = vc.reset_index()
    tmp.columns = [col, 'count']
    title = f"Top {top_n} categories in {col}"
    if kind == 'plotly':
        fig = px.bar(tmp, x=col, y='count', title=title, text='count')
        fig.update_traces(textposition='outside')
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8, max(4, 0.35*len(tmp))))
        sns.barplot(data=tmp, x='count', y=col, palette='viridis', ax=ax)
        ax.set_title(title)
        return fig

def plot_numeric_hist(df: pd.DataFrame, col: str, bins: int = 30, kind: str = 'plotly'):
    title = f"Distribution of {col}"
    if kind == 'plotly':
        fig = px.histogram(df, x=col, nbins=bins, title=title, marginal='box')
        return fig
    else:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(df[col].dropna(), bins=bins, kde=False, ax=ax)
        ax.set_title(title)
        return fig

def plot_scatter(df: pd.DataFrame, x: str, y: str, color: Optional[str] = None, kind: str = 'plotly'):
    title = f"{y} vs {x}"
    if kind == 'plotly':
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
        return fig
    else:
        fig, ax = plt.subplots(figsize=(7,5))
        sns.scatterplot(data=df, x=x, y=y, hue=color, ax=ax)
        ax.set_title(title)
        return fig

# -----------------------
# Saving helpers
# -----------------------
def save_matplotlib_figure(fig: plt.Figure, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches='tight')

def save_plotly_figure(fig, out_path: str):
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(out_path)
    except Exception as e:
        raise RuntimeError("Saving Plotly images requires 'kaleido'. Install with: pip install kaleido") from e

# -----------------------
# Simple rule-based insights (fallback)
# -----------------------
def _find_attrition_col(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c.lower() == 'attrition':
            return c
    return None

def _attrition_to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    return s.map({'yes': 1, 'y': 1, 'no': 0, 'n': 0}).fillna(0).astype(int)

def generate_simple_insights(df: pd.DataFrame) -> List[str]:
    insights = []
    insights.append(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    mv = missing_values(df)
    top_missing = mv[mv > 0].head(5)
    if not top_missing.empty:
        insights.append("Top missing columns: " + "; ".join([f"{c} ({int(v)})" for c, v in top_missing.items()]))

    attr_col = _find_attrition_col(df)
    if attr_col:
        attr_num = _attrition_to_numeric(df[attr_col])
        rate = round(100 * attr_num.sum() / len(df), 2)
        insights.append(f"Attrition rate: {rate}% ({int(attr_num.sum())}/{len(df)})")
        if 'JobRole' in df.columns:
            temp = df[['JobRole', attr_col]].copy()
            temp['__a'] = _attrition_to_numeric(temp[attr_col])
            by_role = temp.groupby('JobRole')['__a'].agg(['sum','count'])
            by_role['rate'] = (by_role['sum'] / by_role['count']*100).round(2)
            top_roles = by_role.sort_values('rate', ascending=False).head(3)
            insights.append("Top roles by attrition rate: " + "; ".join([f"{idx} ({row['rate']}%)" for idx, row in top_roles.iterrows()]))

        numeric = df.select_dtypes(include=[np.number]).copy()
        if not numeric.empty:
            try:
                corr_with_attr = numeric.corrwith(attr_num).abs().sort_values(ascending=False)
                top_corr = corr_with_attr[corr_with_attr > 0].head(5)
                if not top_corr.empty:
                    insights.append("Top numeric features correlated with attrition: " + ", ".join([f"{c} ({corr_with_attr[c]:.2f})" for c in top_corr.index]))
            except Exception:
                pass

    if 'Age' in df.columns:
        insights.append(f"Average age: {df['Age'].mean():.1f}, median age: {df['Age'].median():.1f}")

    if 'MonthlyIncome' in df.columns:
        insights.append(f"Average monthly income: {df['MonthlyIncome'].mean():.0f}")

    return insights

# -----------------------
# Preprocessing & modeling helpers
# -----------------------
def encode_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int)
    s = series.astype(str).str.strip().str.lower()
    return s.map({'yes': 1, 'y': 1, '1': 1, 'true': 1, 'no': 0, 'n': 0, '0': 0, 'false': 0}).fillna(0).astype(int)

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, stratify: bool = True, random_state: int = 42
              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"{target_col} not in dataframe")
    X = df.drop(columns=[target_col]).copy()
    y = encode_target(df[target_col])
    strat = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=strat)

def build_preprocessor(numeric_features: Optional[List[str]] = None,
                       categorical_features: Optional[List[str]] = None,
                       impute_strategy: str = 'median',
                       scaler: Optional[Any] = None,
                       drop_first_ohe: bool = True) -> ColumnTransformer:
    if scaler is None:
        scaler = StandardScaler()

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=impute_strategy)),
        ('scaler', scaler)
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', drop='first' if drop_first_ohe else None))
    ])

    transformers = []
    if numeric_features:
        transformers.append(('num', num_transformer, numeric_features))
    if categorical_features:
        transformers.append(('cat', cat_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', sparse_threshold=0)
    return preprocessor

def build_pipeline(preprocessor: ColumnTransformer, estimator=None) -> Pipeline:
    if estimator is None:
        estimator = LogisticRegression(max_iter=1000, solver='liblinear')
    return Pipeline(steps=[('preprocessor', preprocessor), ('estimator', estimator)])

def train_and_evaluate_pipeline(pipeline: Pipeline,
                                X_train: pd.DataFrame, X_test: pd.DataFrame,
                                y_train: pd.Series, y_test: pd.Series,
                                cv: int = 5) -> Dict[str, Any]:
    pipeline.fit(X_train, y_train)
    try:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_mean = float(np.mean(cv_scores))
    except Exception:
        cv_scores = None
        cv_mean = None

    y_pred = pipeline.predict(X_test)
    y_prob = None
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None,
        "cv_mean_accuracy": cv_mean,
        "cv_scores": cv_scores.tolist() if isinstance(cv_scores, np.ndarray) else cv_scores
    }

    return {"pipeline": pipeline, "metrics": metrics, "y_pred": y_pred, "y_prob": y_prob}

def _get_feature_names_from_preprocessor(preprocessor: ColumnTransformer,
                                         numeric_features: List[str],
                                         categorical_features: List[str]) -> List[str]:
    names = []
    if numeric_features:
        names.extend(numeric_features)
    if categorical_features:
        cat = preprocessor.named_transformers_.get('cat')
        if cat is not None:
            ohe = None
            if hasattr(cat, 'named_steps') and 'ohe' in cat.named_steps:
                ohe = cat.named_steps['ohe']
            elif isinstance(cat, OneHotEncoder):
                ohe = cat
            if ohe is not None:
                ohe_names = list(ohe.get_feature_names_out(categorical_features))
                names.extend(ohe_names)
    return names

def select_kbest_features(preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series,
                          numeric_features: List[str], categorical_features: List[str],
                          k: int = 10, score_func=f_classif) -> List[str]:
    X_prep = preprocessor.fit_transform(X)
    selector = SelectKBest(score_func=score_func, k=min(k, X_prep.shape[1]))
    selector.fit(X_prep, y)
    names = _get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features)
    selected = np.array(names)[selector.get_support()]
    return selected.tolist()

def rfe_select_features(estimator, preprocessor: ColumnTransformer, X: pd.DataFrame, y: pd.Series,
                        numeric_features: List[str], categorical_features: List[str], n_features: int = 5) -> List[str]:
    X_prep = preprocessor.fit_transform(X)
    rfe = RFE(estimator, n_features_to_select=min(n_features, X_prep.shape[1]))
    rfe.fit(X_prep, y)
    names = _get_feature_names_from_preprocessor(preprocessor, numeric_features, categorical_features)
    selected = np.array(names)[rfe.support_]
    return selected.tolist()

def plot_learning_curve_estimator(estimator, X: pd.DataFrame, y: pd.Series, scoring: str = 'accuracy', cv: int = 5,
                                  train_sizes: Optional[np.ndarray] = None, figsize=(6,5)) -> plt.Figure:
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring, n_jobs=-1, train_sizes=train_sizes)
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(train_sizes, train_mean, 'o-', label="Training Score")
    ax.plot(train_sizes, test_mean, 'o-', label="Validation Score")
    ax.set_xlabel("Training Size")
    ax.set_ylabel(scoring)
    ax.set_title("Learning Curve")
    ax.legend()
    return fig

def save_pipeline(pipeline: Pipeline, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_path)
    return out_path

def load_pipeline(path: str):
    return joblib.load(path)
