import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# 1. 載入環境變數
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# ================= 設定區 =================
PROMPT_FOLDER = "running_prompts"
PROMPT_FILENAME = "prompt_v2_completeQ.txt" 
PROMPT_PATH = os.path.join(PROMPT_FOLDER, PROMPT_FILENAME)
INPUT_FILE = "mini_train.json"      # 改成讀取 JSON
OUTPUT_FILE = "structured_dataset.json"
# =========================================

def load_prompt_from_file(filepath):
    """讀取外部 txt 檔案作為 System Prompt"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            print(f"正在載入 Prompt 檔案: {filepath}")
            return f.read()
    except Exception as e:
        print(f"讀取 Prompt 檔案時發生錯誤: {e}")
        exit(1)

# 2. 載入 System Prompt
SYSTEM_PROMPT = load_prompt_from_file(PROMPT_PATH)

def process_item(index, item):
    question = item.get('question', '')
    answer = item.get('answer', '')
    dataset_id = f"gsm8k_train_{index:03d}"
    
    user_content = f"""
    Dataset ID: {dataset_id}
    Question: {question}
    Answer: {answer}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            response_format={"type": "json_object"}, 
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0
        )
        
        # 解析回傳的 JSON 字串
        result_json = json.loads(response.choices[0].message.content)
        
        # ======================================================
        # 【關鍵修改】為了 100% 確保 question 與原文一致
        # 我們不在這裡依賴 AI 的輸出，而是直接用程式碼把原始問題填回去
        # ======================================================
        if 'query' not in result_json:
            result_json['query'] = {}
            
        result_json['query']['question'] = question  # <--- 強制覆寫
        
        # 同樣確保 ID 一致
        if 'context' not in result_json:
            result_json['context'] = {}
        result_json['context']['dataset_id'] = dataset_id

        return result_json

    except Exception as e:
        print(f"Error processing item {index}: {e}")
        return None

def main():
    # 3. 讀取 JSON 檔案
    if not os.path.exists(INPUT_FILE):
        print(f"錯誤：找不到資料檔案 {INPUT_FILE}")
        return

    print(f"Reading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # data 應該是一個 list，如果不確定結構可以用 data[:3] 測試
    data_to_process = data 

    results = []

    print(f"Start processing {len(data_to_process)} items using prompt from: {PROMPT_FILENAME}...")

    for index, item in enumerate(data_to_process):
        print(f"Processing item {index}...")
        structured_data = process_item(index, item)
        
        if structured_data:
            results.append(structured_data)

    # 4. 將結果存檔
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done! Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()