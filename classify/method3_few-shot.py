import json
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FEW_SHOT_PROMPT = """
You are a math expert. Classify math problems into one of the 7 MATH dataset subjects.

Examples:
1. Algebra: "Find the equation of the line passing through (2,3) and (4,7)."
2. Number Theory: "What is the remainder when 100^100 is divided by 13?"
3. Counting & Probability: "How many ways can 5 distinct books be arranged on a shelf?"
4. Geometry: "In triangle ABC, angle A is 90 degrees and AB=3, AC=4. Find BC."
5. Intermediate Algebra: "Find all real roots of the polynomial x^4 - 5x^2 + 4 = 0."
6. Precalculus: "Find the value of cos(pi/3) + sin(pi/6)."

Now, classify the following problem. Output ONLY the subject name.
"""

def classify_question_fewshot(question):
    user_content = f"Problem: {question}\nSubject:"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": FEW_SHOT_PROMPT},
                {"role": "user", "content": user_content}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return "Error"

# Load Data
input_file = 'data/structured_dataset_completeQ.json'
with open(input_file, 'r', encoding='utf-8') as f:
    # ★ 注意：這裡只取前 10 筆
    data = json.load(f)[:10]

results = []

print("Start Few-shot Classification...")
for item in data:
    q_text = item['query']['question']
    q_id = item['context']['dataset_id']
    
    category = classify_question_fewshot(q_text)
    print(f"Processing {q_id}: {category}")
    
    results.append({
        "question_id": q_id,
        "categories": category
    })

# Save to CSV
output_filename = "method3_few-shot_results_new.csv"
df = pd.DataFrame(results)
df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"Done! Results saved to {output_filename}")