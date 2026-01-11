# GSM8K Data Processing & Classification Experiments

本專案包含針對 GSM8K 資料集的結構化提取（ETL）以及數學題型分類的實驗紀錄。

## 0.前置處理

### 資料集來源
本專案使用 OpenAI 發布的 GSM8K 資料集，原始資料可於 Hugging Face 取得：
- [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k)

### 實驗範圍與取樣說明
考量到儲存空間限制以及 LLM API 的 **Token 成本**，本專案採取小規模取樣進行概念驗證 (POC)，並非全量執行：
- **結構化資料提取 (ETL)**：使用資料集的前 **100 筆** 進行測試。
- **題型分類實驗**：使用資料集的前 **9 筆** 進行深度測試。

## 1. 結構化資料提取 (ETL)

目標是將非結構化的數學題目與詳解，提取為符合特定 Schema 的 JSON 格式。

- **核心腳本**: `process_data.py`

### Prompt Schema 設計
使用 LLM 作為 Data Processing Assistant，將題目解析為 JSON 結構。


## 2. 題型分類實作測試結果

### 實驗 1：Embeddings + Clustering (原先預想)

* **腳本**：`method1_clustering.py`
* **產出結果**：`method1_clustering_results.csv`
* **問題**：只能依照特性分群，無法按照指定的標準標籤（如 "Prealgebra", "Algebra" 等）進行分類。
* **結果**：雖然有分出群組，但缺乏明確定義，可能仍需人工進行類別對照，不符自動化需求。

---

### 實驗 2：Zero-shot & Few-shot (含 Prealgebra 標籤)

* **腳本**：`method2_zero-shot.py`, `method3_few-shot.py`
* **產出結果**：`method2_zero-shot_results.csv`, `method3_few-shot_results.csv`

**測試結果比較：**

| 問題編號 | Zero-shot 結果 | Few-shot 結果 |
| :--- | :--- | :--- |
| gsm8k_train_000 | Prealgebra | Prealgebra |
| gsm8k_train_001 | Prealgebra | Prealgebra |
| gsm8k_train_002 | Prealgebra | Prealgebra |
| gsm8k_train_003 | **Algebra** | Prealgebra |
| gsm8k_train_004 | **Counting & Probability** | Prealgebra |
| gsm8k_train_005 | Prealgebra | Prealgebra |
| gsm8k_train_006 | Prealgebra | Prealgebra |
| gsm8k_train_007 | Prealgebra | Prealgebra |
| gsm8k_train_008 | Prealgebra | Prealgebra |
| gsm8k_train_009 | Prealgebra | Prealgebra |

**結論**：
GSM8K 是小學程度題目，而 MATH Dataset 的分類標準偏向高中競賽。模型可能認為 GSM8K 的題目對它來說太簡單，因此將絕大部分題目都歸類為 "Prealgebra"（前代數/基礎算術），導致分類失去鑑別度。即使 Few-shot 給予範例也無法改善此傾向。

**預計解方**：
嘗試將分類標籤中的 "Prealgebra" 移除，測試模型是否能根據題目特性分辨出其他類型。

---

### 實驗 3：Zero-shot & Few-shot 進階測試（移除 Prealgebra 標籤）

* **產出結果**：`method2_zero-shot_results_new.csv`, `method3_few-shot_results_new.csv`

**測試結果比較 (移除 Prealgebra 後)：**

| 問題編號 | Zero-shot 結果 | Few-shot 結果 |
| :--- | :--- | :--- |
| gsm8k_train_000 | Algebra | Algebra |
| gsm8k_train_001 | Algebra | Algebra |
| gsm8k_train_002 | Algebra | Algebra |
| gsm8k_train_003 | Algebra | Algebra |
| gsm8k_train_004 | Counting & Probability | Counting & Probability |
| gsm8k_train_005 | Counting & Probability | Counting & Probability |
| gsm8k_train_006 | Counting & Probability | Counting & Probability |
| gsm8k_train_007 | Algebra | Algebra |
| gsm8k_train_008 | Algebra | Algebra |
| gsm8k_train_009 | Counting & Probability | Counting & Probability |

**結論**：
移除 "Prealgebra" 選項後，分類效果顯著變好！
1.  **一致性高**：Zero-shot 與 Few-shot 的判斷結果完全相同。
2.  **區分度提升**：模型成功將題目分配到 Algebra 與 Counting & Probability 等不同類別，對照題目內容後初步判斷沒有分錯。
3.  **後續規劃**：雖然效果改善，但仍需進一步驗證這樣的強迫分類是否在所有情況下都準確。
