import json
import numpy as np
import requests
import logging
from pathlib import Path
from numpy.linalg import norm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logging.info("開始")

# ===== 設定項目 =====
EMBEDDING_FILE = Path("category_embeddings.jsonl")  # カテゴリの埋め込みファイル
OLLAMA_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "kun432/cl-nagoya-ruri-large:latest"

# ==== ユーザー入力（例）====
# 「量子力学」を含む話題についての検索
user_input = "私は大学時代に量子力学を勉強し、デコヒーレンスについて独自に研究をしていました。"

# ==== 1. ユーザーの入力をEmbedding化 ====
try:
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "input": user_input},
        timeout=30
    )
    response.raise_for_status()
    user_embedding = np.array(response.json()["embeddings"][0], dtype="float32")
except requests.RequestException as e:
    logging.error(f"Ollama APIリクエストに失敗しました: {e}")
    exit(1)
except Exception as e:
    logging.error(f"予期せぬエラーが発生しました: {e}")
    exit(1)

# ==== コサイン類似度計算関数 ====
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b)))

# ==== 2. カテゴリ埋め込みファイルを読み込み、類似度を計算 ====
results = []
with open(EMBEDDING_FILE, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        # JSONLの中身は {"id": str, "title": str, "description": str, "embedding": list}
        cat_embedding = np.array(item["embedding"], dtype="float32")

        sim = cosine_similarity(user_embedding, cat_embedding)
        results.append({
            "id": item["id"],
            "title": item["title"],
            "description": item["description"],
            "similarity": sim
        })

# ==== 3. 類似度が高い順にソート & 上位5件表示 ====
results.sort(key=lambda x: x["similarity"], reverse=True)

print("【マッチング結果（上位5件）】")
for result in results[:5]:
    print(f"- タイトル: {result['title']}, 類似度: {result['similarity']:.4f}")
    print(f"  説明: {result['description']}\n")

logging.info("終了")

