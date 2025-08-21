from together import Together
import os
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("TOGETHER_API_KEY")

client = Together(api_key=api_key)

def get_ai_response(chat_history, predicted_class=None):
    system_prompt = {
        "role": "system",
        "content": f"Kita sedang membahas penyakit ikan '{predicted_class}'. Jawablah setiap pertanyaan pengguna dengan konteks ini." if predicted_class else
                   "Kamu adalah ahli penyakit ikan air tawar."
    }
    messages = [system_prompt] + chat_history
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        max_tokens=500,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
