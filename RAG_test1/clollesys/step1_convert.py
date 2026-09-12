import os
import sys
import pdfplumber

# ==================== 設定區 ====================
# 1. SRC_FOLDER: 放置你原始 PDF 的根目錄路徑
SRC_FOLDER = "./111-1_Course_Materials"

# 2. MD_OUTPUT_FOLDER: 轉出 Markdown 檔案的目標目錄
MD_OUTPUT_FOLDER = "./111-1_Course_Materials_MD"
# ================================================

def pdf_to_markdown(pdf_path):
    """將單一 PDF 轉為帶有標題階層與表格的 Markdown 語法"""
    md_content = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            md_content.append(f"\n<!-- Page {page_idx + 1} -->\n")
            
            # 1. 提取並轉換表格為 Markdown 格式
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    # 清理內容，替換換行符號
                    clean_table = [[str(cell or '').replace('\n', ' ') for cell in row] for row in table]
                    if clean_table and len(clean_table) > 0:
                        header = clean_table[0]
                        md_table = "| " + " | ".join(header) + " |\n"
                        md_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
                        for row in clean_table[1:]:
                            md_table += "| " + " | ".join(row) + " |\n"
                        md_content.append(md_table)
            
            # 2. 提取文字並自動判定標題階層 (Heuristic Title Detection)
            words = page.extract_words(extra_attrs=['size'])
            if not words:
                continue
                
            sizes = [w['size'] for w in words]
            avg_size = sum(sizes) / len(sizes) if sizes else 10
            
            # 依據縱座標 (top) 將文字分行
            lines = {}
            for w in words:
                top = round(w['top'], 1)
                lines.setdefault(top, []).append(w)
            
            for top in sorted(lines.keys()):
                line_words = lines[top]
                line_text = "".join([w['text'] for w in line_words]).strip()
                max_size = max([w['size'] for w in line_words])
                
                if not line_text:
                    continue
                
                # 依據字體大小相較於平均字體大小判定標題
                if max_size > avg_size * 1.5:
                    md_content.append(f"\n# {line_text}\n")
                elif max_size > avg_size * 1.2:
                    md_content.append(f"\n## {line_text}\n")
                else:
                    md_content.append(line_text)
                    
    return "\n".join(md_content)

def run_conversion():
    if not os.path.exists(SRC_FOLDER):
        print(f"❌ 錯誤：找不到原始資料夾 '{SRC_FOLDER}'，請先確認資料夾名稱與路徑！")
        sys.exit(1)

    print(f"🔍 開始掃描 '{SRC_FOLDER}' 中的 PDF 檔案...\n")
    success_count = 0
    fail_count = 0
    skip_count = 0

    for root, dirs, files in os.walk(SRC_FOLDER):
        for file in files:
            # 1. 自動過濾：隱藏檔 (如 Mac ._) 與暫存檔
            if file.startswith('.') or file.startswith('~$'):
                skip_count += 1
                continue
            
            # 2. 自動過濾：非 PDF 檔案直接跳過
            if not file.lower().endswith('.pdf'):
                skip_count += 1
                continue

            pdf_path = os.path.join(root, file)
            
            # 3. 計算相對路徑，保持目標資料夾結構一致
            rel_path = os.path.relpath(pdf_path, SRC_FOLDER)
            md_rel_path = os.path.splitext(rel_path)[0] + ".md"
            md_out_path = os.path.join(MD_OUTPUT_FOLDER, md_rel_path)
            
            # 建立目標子目錄
            os.makedirs(os.path.dirname(md_out_path), exist_ok=True)
            
            print(f"⏳ 正在轉換: {rel_path}")
            try:
                md_text = pdf_to_markdown(pdf_path)
                with open(md_out_path, "w", encoding="utf-8") as f:
                    f.write(md_text)
                success_count += 1
            except Exception as e:
                print(f"   ❌ 轉換失敗 [{rel_path}]: {e}")
                fail_count += 1

    print("\n" + "="*40)
    print(f"🎉 階段 1 執行完成！")
    print(f"✅ 成功轉換：{success_count} 個 PDF")
    if fail_count > 0:
        print(f"⚠️ 失敗數量：{fail_count} 個")
    print(f"⏭️ 自動跳過（非 PDF / 隱藏檔）：{skip_count} 個")
    print(f"📂 產出檔案位置：{os.path.abspath(MD_OUTPUT_FOLDER)}")
    print("="*40)

if __name__ == "__main__":
    run_conversion()