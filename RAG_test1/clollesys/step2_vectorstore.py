import os
from pathlib import Path
import shutil

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ============================================================
# 設定
# ============================================================

# 課程 Markdown / TXT 原始資料
SOURCE_DIR = Path(
    r"C:\Users\lzhix\OneDrive\桌面\clollesys\111_1_Course_Materials_MD"
)

# ChromaDB 刻意放在 OneDrive 外，避免 HNSW index 被同步/鎖定
DB_DIR = Path(r"C:\clollesys_data\chroma_db")

EMBEDDING_MODEL = "shibing624/text2vec-base-chinese"

# 是否在建庫前自動刪除舊資料庫
REBUILD_DATABASE = True


def get_course_name(relative_path: Path) -> str:
    """SOURCE_DIR 下一層即為課程名稱。"""
    parts = relative_path.parts
    return parts[0] if parts else "未分類"


def get_subcategory(relative_path: Path) -> str:
    """取得課程下的子資料夾；若沒有則標記為主目錄。"""
    parts = relative_path.parts
    if len(parts) <= 2:
        return "主目錄"
    return "/".join(parts[1:-1])


def get_doc_type(file_path: Path) -> str:
    """依檔名判斷文件類型。"""
    name = file_path.name.lower()

    if "筆記" in file_path.name or "note" in name or "class-note" in name:
        return "note"

    if any(word in file_path.name for word in ["作業", "報告", "心得", "反思", "評測", "閱讀"]):
        return "assignment"

    if file_path.suffix.lower() in [".ck", ".py", ".pde", ".js", ".java"]:
        return "code"

    return "general"


def main():
    print("=" * 60)
    print("RAG Knowledge Core // Step 2")
    print("建立 ChromaDB 向量資料庫")
    print("=" * 60)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"找不到來源資料夾：\n{SOURCE_DIR}"
        )

    # --------------------------------------------------------
    # 1. 搜尋文本
    # --------------------------------------------------------
    print("\n[1/5] 開始載入並優化文本資料...")

    target_files = sorted(
        list(SOURCE_DIR.rglob("*.md")) +
        list(SOURCE_DIR.rglob("*.txt"))
    )

    raw_documents = []

    for file_path in target_files:
        try:
            content = file_path.read_text(encoding="utf-8")

            if not content.strip():
                continue

            relative_path = file_path.relative_to(SOURCE_DIR)

            raw_documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(file_path),
                        "relative_path": relative_path.as_posix(),
                    },
                )
            )

        except Exception as e:
            print(f"⚠ 讀取檔案失敗 [{file_path}]: {e}")

    print(f"成功讀取 {len(raw_documents)} 個文本檔案。")

    if not raw_documents:
        raise ValueError(
            "未在資料夾內找到任何非空的 .md 或 .txt 檔案！"
        )

    # --------------------------------------------------------
    # 2. Metadata
    # --------------------------------------------------------
    print("\n[2/5] 建立 Metadata...")

    docs_with_metadata = []

    for doc in raw_documents:
        file_path = Path(doc.metadata["source"])
        relative_path = file_path.relative_to(SOURCE_DIR)

        course_name = get_course_name(relative_path)
        subcategory = get_subcategory(relative_path)
        file_name = file_path.name
        doc_type = get_doc_type(file_path)

        doc.metadata.update({
            "semester": "111-1",
            "course": course_name,
            "subcategory": subcategory,
            "doc_type": doc_type,
            "file_name": file_name,
        })

        # 把課程與檔案資訊放進文本，增加 RAG 語境
        header_context = (
            f"[學期: 111-1] "
            f"[課程: {course_name}] "
            f"[分類: {subcategory}] "
            f"[文件類型: {doc_type}] "
            f"[檔案: {file_name}]\n"
        )

        doc.page_content = header_context + doc.page_content
        docs_with_metadata.append(doc)

    courses = sorted(
        set(doc.metadata["course"] for doc in docs_with_metadata)
    )

    print(f"辨識到 {len(courses)} 門課程：")
    for course in courses:
        print(f"  - {course}")

    # --------------------------------------------------------
    # 3. Chunking
    # --------------------------------------------------------
    print("\n[3/5] 進行文本切片...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    chunks = text_splitter.split_documents(docs_with_metadata)

    print(f"切片完成！共生成 {len(chunks)} 個帶有 Metadata 標籤的切片。")

    if not chunks:
        raise ValueError("切片結果為空，無法建立向量資料庫。")

    # --------------------------------------------------------
    # 4. Embedding
    # --------------------------------------------------------
    print("\n[4/5] 正在載入 Embedding 模型...")
    print(f"Model: {EMBEDDING_MODEL}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    # --------------------------------------------------------
    # 5. ChromaDB
    # --------------------------------------------------------
    print("\n[5/5] 正在建立 ChromaDB...")

    DB_DIR.parent.mkdir(parents=True, exist_ok=True)

    if REBUILD_DATABASE and DB_DIR.exists():
        print(f"刪除舊 ChromaDB：{DB_DIR}")

        try:
            shutil.rmtree(DB_DIR)
        except Exception as e:
            raise RuntimeError(
                f"無法刪除舊 ChromaDB。\n"
                f"如果資料夾正在被其他 Python 程式使用，請先關閉 step3。\n"
                f"原始錯誤：{e}"
            )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_metadata={
            "hnsw:space": "cosine",
        },
    )

    # 強制做一次簡單檢索測試，確認 index 可以被讀取
    test_results = vectorstore.similarity_search(
        "課程資料",
        k=1,
    )

    print("\n" + "=" * 60)
    print("階段二完成！")
    print(f"ChromaDB：{DB_DIR}")
    print(f"Chunks：{len(chunks)}")
    print(f"測試檢索：{'成功' if test_results else '無結果'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
