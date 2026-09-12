import os
import math
from typing import Dict, List, Tuple

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from openai import OpenAI

try:
    import networkx as nx
except Exception:
    nx = None

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None


# ============================================================
# 設定
# ============================================================
DB_DIR = r"C:\clollesys_data\chroma_db"
EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
DEFAULT_MODEL = "gpt-4o-mini"

course_list = [
    "ALL_NODES",
    "周易一",
    "影音技術",
    "新媒體藝術概論",
    "科技藝術講座",
    "紅樓夢",
    "義大利文",
    "資訊科學與數位生活",
    "電腦程式設計",
]

PROMPT_TEMPLATES = {
    "總覽整理": "請根據 {scope}，整理這個主題的核心內容、關鍵概念、重要關鍵字與脈絡。若資料不足請直接說明。主題：{topic}",
    "名詞解釋": "請根據 {scope}，解釋「{topic}」的定義、背景、相關概念與實例，並標示它主要來自哪些節點資料。",
    "比較分析": "請根據 {scope}，比較「{topic}」中涉及的兩個或多個概念的異同，請整理成清楚條列或表格，並指出出處。",
    "重點摘錄": "請從 {scope} 中，找出與「{topic}」最相關的節點資料，摘錄重點並用繁體中文整理。",
    "考前複習": "請依據 {scope}，把「{topic}」整理成考前複習重點，包含必背觀念、易混淆點、可能考題。",
    "結構探索": "請從 {scope} 中，找出和「{topic}」最相關的課程節點、分類、檔案與概念之間的關係，像知識圖譜一樣說明。",
    "檔案定位": "請幫我在 {scope} 中定位與「{topic}」最相關的資料。請列出對應的課程節點、分類、檔名與摘要。",
}

QUICK_PROMPTS = [
    "先總覽這個資料庫有哪些課程節點、子分類與主要內容。",
    "列出目前資料庫中每個課程節點底下有哪些子分類，並簡短說明各自內容。",
    "幫我找出和『期中考重點』最相關的資料，並整理成清單。",
    "比較兩個不同課程節點中，對同一個概念的說明有何異同。",
    "如果我想找某個名詞的定義，怎麼問可以最精準？請先給我提問模板。",
    "幫我列出某個主題最相關的檔名、分類與摘要。",
]


# ============================================================
# 初始化
# ============================================================
print("=" * 64)
print("Second Brain RAG Atlas // Step 3")
print("正在載入向量資料庫與 Embedding 模型...")
print("=" * 64)

vectorstore = None
embeddings = None
STARTUP_MESSAGE = ""

if os.path.isdir(DB_DIR):
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings,
        )
        startup_test = vectorstore.similarity_search("111-1 課程資料", k=1)
        STARTUP_MESSAGE = (
            "資料庫已成功載入。"
            f" 啟動測試可檢索 {len(startup_test)} 筆結果。"
        )
        print("ChromaDB 啟動測試成功")
    except Exception as e:
        STARTUP_MESSAGE = f"資料庫載入失敗：{type(e).__name__}: {e}"
        print("❌ ChromaDB 啟動測試失敗")
        print(STARTUP_MESSAGE)
else:
    STARTUP_MESSAGE = (
        "目前找不到 ChromaDB 路徑。\n"
        f"請確認資料庫存在於：{DB_DIR}"
    )
    print("⚠️ 找不到 ChromaDB 路徑")


# ============================================================
# 第二大腦資料快取
# ============================================================
BRAIN_CACHE: Dict[str, object] = {
    "df": pd.DataFrame(),
    "catalog": pd.DataFrame(),
    "status": STARTUP_MESSAGE,
}


def _normalize_text(value: str, limit: int = 140) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."



def load_collection_dataframe() -> pd.DataFrame:
    if vectorstore is None:
        return pd.DataFrame(
            columns=[
                "id",
                "course",
                "subcategory",
                "file_name",
                "content",
                "preview",
                "char_count",
                "embedding",
            ]
        )

    collection = vectorstore._collection
    payload = collection.get(include=["metadatas", "documents", "embeddings"])

    # Chroma may return embeddings as a NumPy ndarray. An ndarray cannot be
    # evaluated with `value or []`, because its truth value is ambiguous.
    ids = payload.get("ids")
    metadatas = payload.get("metadatas")
    documents = payload.get("documents")
    embedding_list = payload.get("embeddings")

    ids = [] if ids is None else ids
    metadatas = [] if metadatas is None else metadatas
    documents = [] if documents is None else documents
    embedding_list = [] if embedding_list is None else embedding_list

    rows = []
    for idx, doc_id in enumerate(ids):
        metadata = metadatas[idx] or {}
        content = documents[idx] if idx < len(documents) else ""
        emb = embedding_list[idx] if idx < len(embedding_list) else None
        rows.append(
            {
                "id": doc_id,
                "course": metadata.get("course", "UNKNOWN_NODE"),
                "subcategory": metadata.get("subcategory", "UNKNOWN_CATEGORY"),
                "file_name": metadata.get("file_name", "UNKNOWN_SOURCE"),
                "content": content or "",
                "preview": _normalize_text(content or ""),
                "char_count": len(content or ""),
                "embedding": emb,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "id",
                "course",
                "subcategory",
                "file_name",
                "content",
                "preview",
                "char_count",
                "embedding",
            ]
        )
    return df



def build_catalog_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["course", "subcategory", "file_name", "chunks", "characters"]
        )

    catalog = (
        df.groupby(["course", "subcategory", "file_name"], dropna=False)
        .agg(chunks=("id", "count"), characters=("char_count", "sum"))
        .reset_index()
        .sort_values(["course", "subcategory", "chunks"], ascending=[True, True, False])
    )
    return catalog



def refresh_brain_cache() -> Dict[str, object]:
    df = load_collection_dataframe()
    catalog = build_catalog_dataframe(df)

    if df.empty:
        status = STARTUP_MESSAGE + "\n目前沒有可供視覺化的資料。"
    else:
        status = (
            f"已載入 {len(df)} 個知識片段、"
            f"{df['course'].nunique()} 個課程節點、"
            f"{df['subcategory'].nunique()} 個子分類、"
            f"{df['file_name'].nunique()} 個來源檔案。"
        )

    BRAIN_CACHE["df"] = df
    BRAIN_CACHE["catalog"] = catalog
    BRAIN_CACHE["status"] = status
    return BRAIN_CACHE


refresh_brain_cache()


# ============================================================
# 視覺化工具
# ============================================================

def get_filtered_df(course: str, category: str) -> pd.DataFrame:
    df = BRAIN_CACHE.get("df", pd.DataFrame())
    if df.empty:
        return df

    filtered = df.copy()
    if course and course != "ALL_NODES":
        filtered = filtered[filtered["course"] == course]
    if category and category != "ALL_CATEGORIES":
        filtered = filtered[filtered["subcategory"] == category]
    return filtered.reset_index(drop=True)



def get_category_choices(course: str) -> List[str]:
    df = BRAIN_CACHE.get("df", pd.DataFrame())
    if df.empty:
        return ["ALL_CATEGORIES"]

    if course == "ALL_NODES":
        categories = sorted(df["subcategory"].dropna().astype(str).unique().tolist())
    else:
        categories = sorted(
            df.loc[df["course"] == course, "subcategory"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    return ["ALL_CATEGORIES"] + categories



def grayscale_palette(n: int) -> List[str]:
    if n <= 1:
        return ["#111111"]
    palette = []
    for i in range(n):
        value = int(30 + (190 * i / max(n - 1, 1)))
        hex_value = f"#{value:02x}{value:02x}{value:02x}"
        palette.append(hex_value)
    return palette



def build_stats_html(df: pd.DataFrame) -> str:
    if df.empty:
        cards = [
            ("Knowledge Chunks", "0"),
            ("Course Nodes", "0"),
            ("Categories", "0"),
            ("Source Files", "0"),
        ]
    else:
        cards = [
            ("Knowledge Chunks", f"{len(df):,}"),
            ("Course Nodes", f"{df['course'].nunique():,}"),
            ("Categories", f"{df['subcategory'].nunique():,}"),
            ("Source Files", f"{df['file_name'].nunique():,}"),
        ]

    card_html = []
    for label, value in cards:
        card_html.append(
            f"""
            <div class=\"metric-card\">
                <div class=\"metric-label\">{label}</div>
                <div class=\"metric-value\">{value}</div>
            </div>
            """
        )
    return f"<div class='metrics-grid'>{''.join(card_html)}</div>"



def build_overview_note(df: pd.DataFrame) -> str:
    if df.empty:
        return "### Second Brain Note\n目前沒有可用資料。請先建立向量資料庫後再啟動此介面。"

    top_courses = (
        df.groupby("course")
        .size()
        .sort_values(ascending=False)
        .head(4)
        .items()
    )
    top_categories = (
        df.groupby("subcategory")
        .size()
        .sort_values(ascending=False)
        .head(6)
        .items()
    )

    course_text = "、".join([f"{name}（{count}）" for name, count in top_courses]) or "無"
    category_text = "、".join([f"{name}（{count}）" for name, count in top_categories]) or "無"

    return (
        "### Second Brain Note\n"
        f"- 目前最主要的課程節點：{course_text}\n"
        f"- 最常出現的子分類：{category_text}\n"
        "- 若你想更精準查詢，建議先鎖定課程節點，再用名詞、問題或任務型句子提問。"
    )



def build_course_distribution_plot(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=18, family="Times New Roman"),
        )
        fig.update_layout(template="simple_white", height=280)
        return fig

    counts = (
        df.groupby("course")
        .size()
        .reset_index(name="chunks")
        .sort_values("chunks", ascending=True)
    )

    fig = px.bar(
        counts,
        x="chunks",
        y="course",
        orientation="h",
        text="chunks",
    )
    fig.update_traces(
        marker_color="#111111",
        hovertemplate="Course: %{y}<br>Chunks: %{x}<extra></extra>",
        textposition="outside",
    )
    fig.update_layout(
        template="simple_white",
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        title="Knowledge Distribution by Course",
        font=dict(family="Times New Roman"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis_title="Chunk Count",
        yaxis_title="",
    )
    return fig



def build_vector_map_plot(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Vector map will appear when embeddings are available.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=18, family="Times New Roman"),
        )
        fig.update_layout(template="simple_white", height=460)
        return fig

    emb_rows = []
    valid_indices = []
    for idx, emb in enumerate(df["embedding"].tolist()):
        if emb is None:
            continue
        arr = np.array(emb, dtype=float)
        if arr.ndim == 1 and arr.size >= 2:
            emb_rows.append(arr)
            valid_indices.append(idx)

    if len(valid_indices) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough embedding vectors for projection.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=18, family="Times New Roman"),
        )
        fig.update_layout(template="simple_white", height=460)
        return fig

    matrix = np.vstack(emb_rows)
    if PCA is not None and matrix.shape[1] >= 2:
        coords = PCA(n_components=2).fit_transform(matrix)
    else:
        coords = matrix[:, :2]

    plot_df = df.iloc[valid_indices].copy().reset_index(drop=True)
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]

    courses = sorted(plot_df["course"].astype(str).unique().tolist())
    colors = grayscale_palette(len(courses))
    color_map = {course: colors[idx] for idx, course in enumerate(courses)}

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="course",
        color_discrete_map=color_map,
        hover_data={
            "course": True,
            "subcategory": True,
            "file_name": True,
            "preview": True,
            "x": False,
            "y": False,
        },
    )
    fig.update_traces(
        marker=dict(size=11, line=dict(color="#ffffff", width=0.6), opacity=0.82),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Category: %{customdata[1]}<br>"
            "File: %{customdata[2]}<br>"
            "Preview: %{customdata[3]}<extra></extra>"
        ),
    )
    fig.update_layout(
        template="simple_white",
        height=480,
        margin=dict(l=20, r=20, t=36, b=20),
        title="Vector Space Projection",
        font=dict(family="Times New Roman"),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis_title="Vector Axis 1",
        yaxis_title="Vector Axis 2",
        legend_title="Course",
    )
    return fig



def build_brain_network_plot(df: pd.DataFrame) -> go.Figure:
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Knowledge graph will appear when metadata is available.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=18, family="Times New Roman"),
        )
        fig.update_layout(template="simple_white", height=520)
        return fig

    if nx is None:
        fig = go.Figure()
        fig.add_annotation(
            text="networkx is not installed. Please install networkx to render the graph.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, family="Times New Roman"),
        )
        fig.update_layout(template="simple_white", height=520)
        return fig

    relation_df = (
        df.groupby(["course", "subcategory"], dropna=False)
        .size()
        .reset_index(name="weight")
        .sort_values("weight", ascending=False)
    )

    graph = nx.Graph()

    course_counts = df.groupby("course").size().to_dict()
    category_counts = df.groupby("subcategory").size().to_dict()

    for course, count in course_counts.items():
        graph.add_node(f"course::{course}", label=course, kind="course", size=count)

    for category, count in category_counts.items():
        graph.add_node(f"category::{category}", label=category, kind="category", size=count)

    for _, row in relation_df.iterrows():
        graph.add_edge(
            f"course::{row['course']}",
            f"category::{row['subcategory']}",
            weight=float(row["weight"]),
        )

    pos = nx.spring_layout(graph, seed=42, k=1.2 / max(math.sqrt(max(graph.number_of_nodes(), 1)), 1))

    edge_x = []
    edge_y = []
    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#b9b9b9"),
        hoverinfo="none",
        mode="lines",
        showlegend=False,
    )

    node_x = []
    node_y = []
    texts = []
    sizes = []
    colors = []
    outlines = []

    for node, data in graph.nodes(data=True):
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        label = data.get("label", node)
        kind = data.get("kind", "category")
        count = int(data.get("size", 1))
        texts.append(f"{label}<br>{kind.title()} • {count} chunks")
        sizes.append(18 + min(count, 30) * 1.7)
        if kind == "course":
            colors.append("#111111")
            outlines.append("#111111")
        else:
            colors.append("#ffffff")
            outlines.append("#111111")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        hoverinfo="text",
        text=[data.get("label") for _, data in graph.nodes(data=True)],
        textposition="top center",
        hovertext=texts,
        marker=dict(
            size=sizes,
            color=colors,
            line=dict(color=outlines, width=1.5),
        ),
        textfont=dict(family="Times New Roman", size=12, color="#111111"),
        showlegend=False,
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        template="simple_white",
        height=560,
        margin=dict(l=20, r=20, t=36, b=20),
        title="Second Brain Knowledge Graph",
        font=dict(family="Times New Roman"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
    )
    return fig



def build_catalog_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["course", "subcategory", "file_name", "chunks", "characters"]
        )
    return build_catalog_dataframe(df)


# ============================================================
# 提問引導
# ============================================================

def scope_label(course: str, category: str) -> str:
    if course == "ALL_NODES" and category == "ALL_CATEGORIES":
        return "整個資料庫"
    if course != "ALL_NODES" and category == "ALL_CATEGORIES":
        return f"課程節點「{course}」"
    if course == "ALL_NODES" and category != "ALL_CATEGORIES":
        return f"整個資料庫中屬於「{category}」的內容"
    return f"課程節點「{course}」中的「{category}」分類"



def generate_prompt_template(intent: str, course: str, category: str, topic: str) -> str:
    topic = (topic or "想查詢的主題").strip()
    scope = scope_label(course, category)
    template = PROMPT_TEMPLATES.get(intent, PROMPT_TEMPLATES["總覽整理"])
    return template.format(scope=scope, topic=topic)



def build_query_guide(course: str, category: str) -> str:
    scope = scope_label(course, category)
    examples = [
        f"- 在 {scope} 中，先幫我總覽有哪些重點內容。",
        f"- 在 {scope} 中，『某個名詞』的定義與出處是什麼？",
        f"- 請從 {scope} 找出和『某個問題』最相關的節點、分類與檔案。",
        f"- 請比較 {scope} 裡兩個概念的差異，並整理成表格。",
    ]
    tips = [
        "1. 先鎖定課程節點，可以大幅減少檢索噪音。",
        "2. 問題盡量包含：主題 + 你想做什麼（解釋 / 比較 / 摘錄 / 定位）。",
        "3. 如果想找到原始材料，直接要求輸出節點、分類、檔名與摘要。",
        "4. 如果回答太廣，請再加上子分類或用途，例如『考前複習』『名詞解釋』。",
    ]
    return (
        "### Query Guide\n"
        f"目前檢索範圍：**{scope}**\n\n"
        "**推薦提問方式**\n"
        + "\n".join(examples)
        + "\n\n**精準查詢技巧**\n"
        + "\n".join([f"- {item}" for item in tips])
    )


# ============================================================
# RAG 對話
# ============================================================

def retrieve_context(message: str, selected_course: str, selected_category: str, k_retrievals: int):
    if vectorstore is None:
        return [], "資料庫尚未載入，無法檢索。", "### Retrieved Context\n資料庫尚未載入，無法檢索。"

    search_kwargs = {}
    filters = {}
    if selected_course != "ALL_NODES":
        filters["course"] = selected_course
    if selected_category != "ALL_CATEGORIES":
        filters["subcategory"] = selected_category
    if filters:
        search_kwargs["filter"] = filters

    results = vectorstore.similarity_search(
        query=message,
        k=int(k_retrievals),
        **search_kwargs,
    )

    if not results:
        return [], "未檢索到相關節點資料。", "### Retrieved Context\n未檢索到相關節點資料。"

    blocks = []
    source_lines = ["### Retrieved Context"]
    for idx, doc in enumerate(results, start=1):
        source = doc.metadata.get("file_name", "UNKNOWN_SOURCE")
        course = doc.metadata.get("course", "UNKNOWN_NODE")
        subcategory = doc.metadata.get("subcategory", "UNKNOWN_CATEGORY")
        preview = _normalize_text(doc.page_content, limit=240)

        blocks.append(
            f"| NODE: {course} | CATEGORY: {subcategory} | FILE: {source} |\n{doc.page_content}"
        )
        source_lines.append(
            f"**{idx}. {course} / {subcategory} / {source}**  \n{preview}"
        )

    context_str = "\n\n--------------------\n\n".join(blocks)
    source_md = "\n\n".join(source_lines)
    return results, context_str, source_md



def run_rag_answer(message: str, api_key: str, model_choice: str, selected_course: str, selected_category: str, k_retrievals: int):
    clean_key = (api_key or DEFAULT_API_KEY).strip()
    if not clean_key or not clean_key.startswith("sk-"):
        return (
            "⚡ [SYSTEM_ERROR]\n\nAPI Key 無效或尚未輸入。\n請在左側欄位輸入 sk- 開頭的 OpenAI API Key。",
            "### Retrieved Context\n目前尚未檢索，請先輸入正確的 API Key。",
        )

    if vectorstore is None:
        return (
            "⚡ [SYSTEM_ERROR]\n\n目前找不到向量資料庫，請確認 DB_DIR 是否正確，並先建立資料庫。",
            "### Retrieved Context\n資料庫未載入。",
        )

    try:
        _, context_str, source_md = retrieve_context(
            message,
            selected_course,
            selected_category,
            k_retrievals,
        )

        prompt = f"""你是一個高等個人知識庫檢索核心。
請根據以下【節點資料】精準回答使用者【查詢】。

規則：
1. 優先使用節點資料回答。
2. 不要捏造資料。
3. 若資料不足，直接聲明資料缺失。
4. 若有多個課程節點，請清楚區分來源。
5. 回答使用繁體中文。
6. 若適合，先簡短回答，再條列整理。

【節點資料】：
{context_str}

【查詢目標】：
{message}
"""

        client = OpenAI(api_key=clean_key)
        response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise personal knowledge retrieval engine.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content
        return answer, source_md
    except Exception as e:
        return (
            f"⚡ [CRITICAL_FAILURE]\n\n{type(e).__name__}: {e}",
            "### Retrieved Context\n發生錯誤，請查看終端機訊息。",
        )



def submit_chat(message, history, api_key, model_choice, selected_course, selected_category, k_retrievals):
    history = history or []
    clean_message = (message or "").strip()
    if not clean_message:
        return history, "", "### Retrieved Context\n請先輸入查詢內容。"

    answer, source_md = run_rag_answer(
        clean_message,
        api_key,
        model_choice,
        selected_course,
        selected_category,
        k_retrievals,
    )

    history = history + [
        {"role": "user", "content": clean_message},
        {"role": "assistant", "content": answer},
    ]
    return history, "", source_md


# ============================================================
# UI 更新函式
# ============================================================

def refresh_dashboard(course: str, category: str):
    df = get_filtered_df(course, category)
    status = BRAIN_CACHE.get("status", STARTUP_MESSAGE)
    status_md = f"### System Status\n{status}"
    stats_html = build_stats_html(df)
    overview_md = build_overview_note(df)
    guide_md = build_query_guide(course, category)
    course_plot = build_course_distribution_plot(df if course == "ALL_NODES" else get_filtered_df("ALL_NODES", "ALL_CATEGORIES"))
    vector_plot = build_vector_map_plot(df)
    brain_plot = build_brain_network_plot(df)
    catalog_df = build_catalog_table(df)
    return status_md, stats_html, overview_md, guide_md, course_plot, vector_plot, brain_plot, catalog_df



def on_course_change(course: str):
    categories = get_category_choices(course)
    return gr.update(choices=categories, value="ALL_CATEGORIES")



def rebuild_everything(course: str):
    refresh_brain_cache()
    categories = get_category_choices(course)
    dashboard = refresh_dashboard(course, "ALL_CATEGORIES")
    return (*dashboard, gr.update(choices=categories, value="ALL_CATEGORIES"))



def inject_quick_prompt(prompt_text: str):
    return prompt_text



def compose_prompt(intent: str, course: str, category: str, topic: str):
    return generate_prompt_template(intent, course, category, topic)


custom_css = """
:root {
    --bg: #f5f4f0;
    --panel: rgba(255,255,255,0.92);
    --text: #111111;
    --muted: #666666;
    --line: #d8d5ce;
    --soft: #ebe8e1;
    --accent: #111111;
}

* {
    box-sizing: border-box !important;
    font-family: "Times New Roman", "Noto Serif TC", serif !important;
}

body, .gradio-container {
    background:
        linear-gradient(rgba(17,17,17,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(17,17,17,0.03) 1px, transparent 1px),
        var(--bg) !important;
    background-size: 24px 24px, 24px 24px, auto !important;
    color: var(--text) !important;
}

.gradio-container {
    max-width: 1480px !important;
    margin: 0 auto !important;
    padding: 22px 22px 38px !important;
}

h1, h2, h3, h4, p, span, label {
    color: var(--text) !important;
}

h1 {
    font-size: 2rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    margin-bottom: 0.2rem !important;
}

h3 {
    font-size: 0.95rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

.hero-panel, .node-panel {
    background: var(--panel) !important;
    border: 1px solid var(--line) !important;
    border-radius: 18px !important;
    box-shadow: 0 12px 34px rgba(17,17,17,0.05) !important;
    backdrop-filter: blur(8px) !important;
}

.hero-panel {
    padding: 22px 24px !important;
    margin-bottom: 18px !important;
}

.node-panel {
    padding: 18px !important;
}

.node-panel-tight {
    padding: 14px !important;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}

.metric-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 14px 16px;
    min-height: 90px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-label {
    color: var(--muted);
    font-size: 0.85rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-value {
    color: var(--text);
    font-size: 2rem;
    line-height: 1;
    font-weight: 700;
}

input, textarea, select, .gradio-dropdown, .gradio-textbox {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    box-shadow: none !important;
}

input:focus, textarea:focus, select:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px var(--accent) !important;
}

button {
    border-radius: 12px !important;
    box-shadow: none !important;
    font-weight: 700 !important;
    transition: all 0.2s ease !important;
}

button.primary, button[variant="primary"] {
    background: #111111 !important;
    color: #ffffff !important;
    border: 1px solid #111111 !important;
}

button.secondary, button[variant="secondary"] {
    background: #ffffff !important;
    color: #111111 !important;
    border: 1px solid var(--line) !important;
}

button:hover {
    transform: translateY(-1px);
}

.message {
    border-radius: 16px !important;
    border: 1px solid var(--line) !important;
    font-size: 1rem !important;
    line-height: 1.68 !important;
}

.bot-row .message {
    background: #ffffff !important;
    color: #111111 !important;
}

.user-row .message {
    background: #111111 !important;
    border-color: #111111 !important;
    color: #ffffff !important;
}

.user-row .message * {
    color: #ffffff !important;
}

.brain-tag {
    display: inline-block;
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 6px 10px;
    margin-right: 8px;
    margin-bottom: 8px;
    background: #ffffff;
    font-size: 0.88rem;
}

footer {
    display: none !important;
}

@media (max-width: 1100px) {
    .metrics-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
"""


with gr.Blocks(title="Second Brain RAG Atlas") as demo:
    with gr.Column(elem_classes=["hero-panel"]):
        gr.Markdown("# Second Brain RAG Atlas")
        gr.Markdown(
            "把你的向量資料庫變成可閱讀、可探索、可精準提問的第二大腦。"
        )
        gr.HTML(
            "<span class='brain-tag'>Vector Retrieval</span>"
            "<span class='brain-tag'>Knowledge Graph</span>"
            "<span class='brain-tag'>Prompt Guidance</span>"
            "<span class='brain-tag'>Times New Roman UI</span>"
        )

    with gr.Row(equal_height=False):
        with gr.Column(scale=3):
            with gr.Column(elem_classes=["node-panel"]):
                gr.Markdown("### Knowledge Atlas")
                stats_html = gr.HTML()
                overview_md = gr.Markdown()
                system_status_md = gr.Markdown()

            with gr.Tabs():
                with gr.Tab("Atlas Overview"):
                    with gr.Column(elem_classes=["node-panel"]):
                        course_dist_plot = gr.Plot(label="Course Distribution")
                    with gr.Column(elem_classes=["node-panel"]):
                        vector_map_plot = gr.Plot(label="Vector Map")
                with gr.Tab("Second Brain Graph"):
                    with gr.Column(elem_classes=["node-panel"]):
                        brain_graph_plot = gr.Plot(label="Knowledge Graph")
                with gr.Tab("Content Catalogue"):
                    with gr.Column(elem_classes=["node-panel"]):
                        catalog_table = gr.Dataframe(
                            headers=["course", "subcategory", "file_name", "chunks", "characters"],
                            interactive=False,
                            wrap=True,
                            row_count=(10, "dynamic"),
                            column_count=5,
                        )

        with gr.Column(scale=2):
            with gr.Column(elem_classes=["node-panel"]):
                gr.Markdown("### Controls")
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-proj-...",
                    type="password",
                    value=DEFAULT_API_KEY,
                )
                model_selector = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o"],
                    value=DEFAULT_MODEL,
                    label="Model",
                )
                course_filter = gr.Dropdown(
                    choices=course_list,
                    value="ALL_NODES",
                    label="Course Node",
                )
                category_filter = gr.Dropdown(
                    choices=["ALL_CATEGORIES"],
                    value="ALL_CATEGORIES",
                    label="Category",
                )
                k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Retrieval Depth (k)",
                )
                refresh_button = gr.Button("Refresh Second Brain", variant="secondary")

            with gr.Column(elem_classes=["node-panel"]):
                gr.Markdown("### Prompt Lab")
                prompt_intent = gr.Dropdown(
                    choices=list(PROMPT_TEMPLATES.keys()),
                    value="總覽整理",
                    label="提問目的",
                )
                prompt_topic = gr.Textbox(
                    label="主題 / 關鍵字",
                    placeholder="例如：蒙太奇、紅樓夢人物關係、資料庫建構流程...",
                )
                compose_button = gr.Button("生成提問模板", variant="primary")
                message_input = gr.Textbox(
                    label="提問欄",
                    placeholder="在這裡輸入問題，或先用上方的 Prompt Lab 自動產生。",
                    lines=5,
                )
                with gr.Row():
                    send_button = gr.Button("Send", variant="primary")
                    clear_button = gr.Button("Clear Chat", variant="secondary")
                query_guide_md = gr.Markdown()

            with gr.Accordion("Quick Prompt Suggestions", open=True):
                quick_buttons = []
                for text in QUICK_PROMPTS:
                    btn = gr.Button(text, variant="secondary")
                    quick_buttons.append((btn, text))

            with gr.Column(elem_classes=["node-panel"]):
                gr.Markdown("### Dialogue")
                chatbot = gr.Chatbot(height=460)
                retrieved_context_md = gr.Markdown("### Retrieved Context\n尚未檢索。")

    demo.load(
        fn=lambda: (
            *refresh_dashboard("ALL_NODES", "ALL_CATEGORIES"),
            gr.update(choices=get_category_choices("ALL_NODES"), value="ALL_CATEGORIES"),
        ),
        inputs=None,
        outputs=[
            system_status_md,
            stats_html,
            overview_md,
            query_guide_md,
            course_dist_plot,
            vector_map_plot,
            brain_graph_plot,
            catalog_table,
            category_filter,
        ],
    )

    course_filter.change(
        fn=on_course_change,
        inputs=[course_filter],
        outputs=[category_filter],
    ).then(
        fn=refresh_dashboard,
        inputs=[course_filter, category_filter],
        outputs=[
            system_status_md,
            stats_html,
            overview_md,
            query_guide_md,
            course_dist_plot,
            vector_map_plot,
            brain_graph_plot,
            catalog_table,
        ],
    )

    category_filter.change(
        fn=refresh_dashboard,
        inputs=[course_filter, category_filter],
        outputs=[
            system_status_md,
            stats_html,
            overview_md,
            query_guide_md,
            course_dist_plot,
            vector_map_plot,
            brain_graph_plot,
            catalog_table,
        ],
    )

    refresh_button.click(
        fn=rebuild_everything,
        inputs=[course_filter],
        outputs=[
            system_status_md,
            stats_html,
            overview_md,
            query_guide_md,
            course_dist_plot,
            vector_map_plot,
            brain_graph_plot,
            catalog_table,
            category_filter,
        ],
    )

    compose_button.click(
        fn=compose_prompt,
        inputs=[prompt_intent, course_filter, category_filter, prompt_topic],
        outputs=[message_input],
    )

    for btn, text in quick_buttons:
        btn.click(fn=lambda t=text: inject_quick_prompt(t), outputs=[message_input])

    send_button.click(
        fn=submit_chat,
        inputs=[
            message_input,
            chatbot,
            api_key_input,
            model_selector,
            course_filter,
            category_filter,
            k_slider,
        ],
        outputs=[chatbot, message_input, retrieved_context_md],
    )

    message_input.submit(
        fn=submit_chat,
        inputs=[
            message_input,
            chatbot,
            api_key_input,
            model_selector,
            course_filter,
            category_filter,
            k_slider,
        ],
        outputs=[chatbot, message_input, retrieved_context_md],
    )

    clear_button.click(
        fn=lambda: ([], "", "### Retrieved Context\n尚未檢索。"),
        outputs=[chatbot, message_input, retrieved_context_md],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, css=custom_css)
