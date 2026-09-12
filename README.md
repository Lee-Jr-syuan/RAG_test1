<img width="1418" height="879" alt="螢幕擷取畫面 2026-09-12 192725" src="https://github.com/user-attachments/assets/7d5a9941-823c-47ae-8f8c-d964697602da" />
<img width="774" height="560" alt="newplot" src="https://github.com/user-attachments/assets/cd4ee3a3-1de7-4ffc-9859-ef4ec44e3c36" />

Second Brain RAG Atlas

一套在 Windows 本機執行的個人知識庫系統。它會將課程 PDF 轉成 Markdown、切割成可檢索的知識片段、寫入 ChromaDB，最後透過 Gradio 提供資料視覺化、提示詞引導與 RAG 對話介面。

主要功能

將資料夾內的 PDF 批次轉換為 Markdown，保留原始目錄結構。

為文件加入學期、課程、分類、文件類型與檔名等 Metadata。

使用 shibing624/text2vec-base-chinese 建立中文語意向量。

使用 ChromaDB 儲存並進行相似度檢索。

顯示課程資料量、向量空間投影、知識關聯圖與內容目錄。

依課程、子分類與 Retrieval Depth 縮小檢索範圍。

透過 Prompt Lab 產生總覽、比較、名詞解釋、考前複習等查詢。

使用 OpenAI 模型根據檢索結果回答，並顯示實際引用的 Retrieved Context。

系統流程

flowchart TD
    A[原始 PDF 課程資料] --> B[step1_convert.py]
    B --> C[Markdown 文字資料]
    C --> D[step2_vectorstore.py]
    D --> E[ChromaDB 向量資料庫]
    E --> F[step3_gui_app.py]
    F --> G[知識視覺化與 RAG 對話]

專案結構

clollesys/
├─ 111_1_Course_Materials/       # 原始課程資料
│  ├─ 111-1/                     # 學期
│  │  ├─ 周易一/
│  │  ├─ 影音技術/
│  │  ├─ 新媒體藝術概論/
│  │  ├─ 科技藝術講座/
│  │  ├─ 紅樓夢/
│  │  ├─ 義大利文/
│  │  ├─ 資訊科學與數位生活/
│  │  └─ 電腦程式設計/
│  ├─ 完整索引.md
│  └─ 檔案結構.txt
├─ 111_1_Course_Materials_MD/    # 轉換後的 Markdown 資料
├─ 0911_TestResult/              # UI 畫面、操作錄影與回答測試
├─ step1_convert.py              # PDF → Markdown
├─ step2_vectorstore.py          # Markdown/TXT → ChromaDB
├─ step3_gui_app.py              # 第二大腦視覺化與 RAG UI
└─ README.md

向量資料庫預設儲存在 OneDrive 之外：

C:\clollesys_data\chroma_db

這樣可降低 OneDrive 同步或鎖定 HNSW 索引檔案的風險。

執行環境

Windows 10／11

Python 3.11 以上

可連線至 Hugging Face Hub，以首次下載 Embedding 模型

OpenAI API Key，用於產生最後回答

安裝

在 PowerShell 進入專案資料夾：

cd C:\Users\lzhix\OneDrive\桌面\clollesys

建議建立獨立虛擬環境：

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

安裝所需套件：

python -m pip install pdfplumber langchain-core langchain-text-splitters langchain-huggingface langchain-chroma chromadb sentence-transformers openai gradio plotly pandas numpy networkx scikit-learn

確認主要套件可正常載入：

python -c "import gradio, chromadb, plotly, pandas, numpy, networkx, sklearn; print('Dependencies OK')"

首次設定：確認三個資料路徑

執行前請先開啟三支 Python 程式，確認路徑符合自己的電腦。

1. step1_convert.py

壓縮檔內的資料夾使用底線 _，建議設定為：

SRC_FOLDER = "./111_1_Course_Materials"
MD_OUTPUT_FOLDER = "./111_1_Course_Materials_MD"

原始程式若仍是 111-1_Course_Materials，會因名稱與實際資料夾不一致而找不到資料。

2. step2_vectorstore.py

建議讓 SOURCE_DIR 直接指向學期資料夾，使下一層正確對應各門課程：

SOURCE_DIR = Path(
    r"C:\Users\lzhix\OneDrive\桌面\clollesys\111_1_Course_Materials_MD\111-1"
)

DB_DIR = Path(r"C:\clollesys_data\chroma_db")

如果 SOURCE_DIR 只指到 111_1_Course_Materials_MD，程式可能會把 111-1 辨識為課程名稱，而不是學期層。

3. step3_gui_app.py

DB_DIR 必須與 Step 2 完全一致：

DB_DIR = r"C:\clollesys_data\chroma_db"

新增課程時，也要將課程名稱加入 course_list，它才會出現在 UI 的課程選單中。

執行方式

Step 1：將 PDF 轉成 Markdown

python step1_convert.py

轉換結果會寫入：

111_1_Course_Materials_MD/

如果 Markdown 資料已經準備完成，可以略過這一步。

目前 Step 1 只會處理 PDF；TXT、PPTX、ODT、圖片與掃描型 PDF 不會自動轉換。要讓 TXT 被收錄，可將檔案依課程分類後放進 Markdown 資料夾。掃描文件則需先進行 OCR。

Step 2：建立向量資料庫

執行前請先關閉正在運行的 step3_gui_app.py，再執行：

python step2_vectorstore.py

程式會依序：

讀取 .md 與 .txt。

加入課程與來源 Metadata。

以 500 字、重疊 50 字切割文本。

建立中文 Embedding。

寫入 ChromaDB 並執行一次檢索測試。

REBUILD_DATABASE = True 會在建庫前刪除既有的 ChromaDB 並完整重建。這只會影響 DB_DIR 指定的向量資料庫，不會刪除原始課程檔案。

Step 3：啟動第二大腦介面

python step3_gui_app.py

啟動成功後，瀏覽器會自動開啟本機 Gradio 網址，通常是：

http://127.0.0.1:7860

OpenAI API Key

可以直接在 UI 的 OpenAI API Key 欄位輸入，或在 PowerShell 先設定環境變數：

$env:OPENAI_API_KEY="sk-proj-你的金鑰"
python step3_gui_app.py

請勿將 API Key 寫進程式、README、截圖或上傳到公開 Git 儲存庫。API 使用可能產生費用，實際計費依 OpenAI 帳戶與所選模型而定。

UI 使用方式

區域

用途

Knowledge Atlas

顯示知識片段、課程、分類與來源檔案數量

Atlas Overview

查看各課程資料量與向量空間投影

Second Brain Graph

查看課程與子分類之間的連線關係

Content Catalogue

查看檔名、分類、片段數與字數

Course Node

限定要搜尋的課程

Category

進一步限定子分類

Retrieval Depth

控制每次檢索的知識片段數量，預設為 4

Prompt Lab

根據目的與主題自動產生完整提問

Retrieved Context

顯示回答前實際檢索到的來源片段

Retrieval Depth 建議

2–3：單一名詞、精準定位、查找特定檔案。

4–6：一般整理、課程複習、同一主題的綜合回答。

7–10：跨課程比較或廣泛總覽，但可能增加無關內容。

提問範例

精準查詢最好包含「範圍＋主題＋任務＋輸出格式」。例如：

請根據「紅樓夢」課程節點，比較賈寶玉與林黛玉的性格與關係，整理成表格，並標示來源檔名。

請從「周易一」中找出和坤卦相關的資料，先解釋核心概念，再列出最相關的分類、檔名與摘要。

請比較「新媒體藝術概論」與「科技藝術講座」對科技和藝術關係的看法，整理共同點、差異與引用來源。

請將「資訊科學與數位生活」整理成考前複習重點，包含必背概念、易混淆處與五題可能考題。

更新資料庫

新增 PDF

將 PDF 放入 111_1_Course_Materials/111-1/課程名稱/。

執行 step1_convert.py。

關閉 Step 3 UI。

執行 step2_vectorstore.py 重建資料庫。

重新啟動 step3_gui_app.py。

新增 Markdown 或 TXT

將檔案放入 111_1_Course_Materials_MD/111-1/課程名稱/。

關閉 Step 3 UI。

執行 step2_vectorstore.py。

重新啟動 Step 3。

UI 中的 Refresh Second Brain 只會重新讀取現有 ChromaDB，不會把新的原始檔案自動轉換或寫入資料庫。

常見問題

ModuleNotFoundError: No module named 'plotly'

目前執行程式的 Python 環境缺少套件：

python -m pip install plotly pandas numpy networkx scikit-learn

ValueError: The truth value of an array ... is ambiguous

這是 Chroma 回傳 NumPy embeddings 時，舊版程式使用 or [] 判斷陣列造成的錯誤。專案目前的 step3_gui_app.py 已改用明確的 None 判斷。

Chatbot.__init__() got an unexpected keyword argument 'type'

這是 Gradio 6 的 API 變更。專案目前已移除 gr.Chatbot(type="messages") 中的 type 參數，並使用新版預設訊息格式。

找不到 ChromaDB

確認：

Step 2 已成功完成。

Step 2 與 Step 3 的 DB_DIR 完全相同。

C:\clollesys_data\chroma_db 確實存在。

無法刪除或重建 ChromaDB

通常是 Step 3 或其他 Python 程式仍在使用資料庫。關閉相關終端機與瀏覽器介面後，再執行 Step 2。

Hugging Face 顯示 unauthenticated warning

這只是速率與下載速度提醒，不會阻止系統啟動。需要較高下載額度時，可自行設定 Hugging Face Token：

$env:HF_TOKEN="你的 Hugging Face Token"

查不到某個檔案內容

依序檢查：

原始檔是否已轉成 Markdown 或 TXT。

檔案是否放在 Step 2 的 SOURCE_DIR 底下。

是否在新增資料後重新執行 Step 2。

課程與分類篩選是否限制了搜尋範圍。

Retrieval Depth 是否太低。

已知限制

PDF 轉換依賴文字層；純掃描 PDF 需要先做 OCR。

Step 1 不會直接解析 PPTX、ODT、圖片、音訊或影片。

PCA 向量圖是高維語意向量的二維投影，只適合觀察相對群聚，不代表絕對距離。

知識圖譜目前主要呈現「課程—子分類」關係，不是自動抽取人物或概念間的語意關係。

回答品質取決於原始文字品質、切片內容、檢索範圍與提問精確度。

隱私與資料使用

此專案包含個人課程作業、筆記、圖片與可能受著作權保護的教材。請勿將完整資料夾、向量資料庫、測試截圖或 API Key 上傳到公開儲存庫。若要公開程式碼，建議只保留三支 Python 程式與去識別化的示例資料。

專案定位

Second Brain RAG Atlas 的目標不是取代原始文件，而是提供一個可以看見資料結構、快速定位來源、建立跨課程連結並以自然語言查詢的個人第二大腦介面。
