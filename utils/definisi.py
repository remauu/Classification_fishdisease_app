from together import Together
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")

client = Together(api_key=api_key)

def get_definisi_llm(penyakit: str) -> str:
    prompt = (
        f"Penyakit yang sedang dibahas adalah '{penyakit}'. "
        f"Jelaskan penyakit ini secara langsung seperti: '{penyakit} adalah penyakit pada ikan air tawar yang...'. "
        f"Gunakan Bahasa Indonesia yang jelas dan sistematis."
        f"Setelah menjelaskan buat poin-poin(3) untuk penanganan {penyakit}"
    )

    messages = [
        {"role": "system", "content": "Kamu adalah pakar penyakit ikan air tawar. Jawablah dengan bahasa Indonesia."},
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Gagal mendapatkan definisi: {e}"