import json
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SUBJECTS = [
    "Algebra", "Number Theory", "Counting & Probability",
    "Geometry", "Intermediate Algebra", "Precalculus"
]

def classify_question_zeroshot(question):
    prompt = f"""
    You are a math expert. Classify the following math problem into EXACTLY ONE of these 7 subjects:
    {', '.join(SUBJECTS)}.
    
    Problem: {question}
    
    Output only the subject name, nothing else.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[{"role": "user", "content": prompt}],
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

print("Start Zero-shot Classification...")
for item in data:
    q_text = item['query']['question']
    q_id = item['context']['dataset_id']
    
    category = classify_question_zeroshot(q_text)
    print(f"Processing {q_id}: {category}")
    
    results.append({
        "question_id": q_id,
        "categories": category
    })

# Save to CSV
output_filename = "method2_zero-shot_results_new.csv"
df = pd.DataFrame(results)
df.to_csv(output_filename, index=False, encoding='utf-8-sig')
print(f"Done! Results saved to {output_filename}")