import json
import re
import requests
import logging
import numpy as np
from pathlib import Path
from numpy.linalg import norm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ========= 設定項目 =========
CATEGORY_EMBEDDING_FILE = Path("category_embeddings.jsonl")

# Llama3.3:latest を利用する想定の API URL とモデル名
LLAMA_API_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL = "llama3.3:latest"

# 埋め込み取得API（Ollama等）の設定
EMBED_API_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "kun432/cl-nagoya-ruri-large:latest"

# プロンプトテンプレート。中括弧はエスケープ済み。
PROMPT_TEMPLATE = """次の文章が[社会・公共,自然・科学・技術,文化・芸術・表現,生活・実践・ライフスキル,心理・精神・内面世界]のうち、どのカテゴリーに該当するかを推測せよ。
次にサブカテゴリーを推測し、サブカテゴリーについて{{"categoryname": （サブカテゴリー名）, "description": （サブカテゴリーの説明）}}のjson形式で解答せよ。

{user_text}
"""

# ===== コサイン類似度計算関数 =====
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (norm(a) * norm(b)))

# ===== メイン処理 =====
def main():
    # ユーザー入力のサンプル
    user_input = "私は大学時代に量子力学を勉強し、デコヒーレンスについて独自に研究していました。"

    # 1. ユーザー入力を Llama3.3 で抽象化し、JSONを取得する
    abstracted_json = call_llama_for_category(user_input)
    if not abstracted_json:
        logging.error("LLMが有効なJSONを返しませんでした。処理を中断します。")
        return

    # LLMが返した {"categoryname": ..., "description": ...} を抽出
    category_name = abstracted_json.get("categoryname")
    subcategory_desc = abstracted_json.get("description")
    logging.info(f"カテゴリー推測結果: {category_name}, 説明: {subcategory_desc}")

    # 2. 抽象化された "description" を埋め込み化する
    desc_embedding = call_embed_api(subcategory_desc)
    if desc_embedding is None:
        logging.error("埋め込みAPIが有効なベクトルを返しませんでした。処理を中断します。")
        return
    desc_embedding = np.array(desc_embedding, dtype="float32")

    # 3. 事前に用意したカテゴリ埋め込みファイルとの類似度計算
    results = []
    with open(CATEGORY_EMBEDDING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            cat_emb = np.array(item["embedding"], dtype="float32")
            sim = cosine_similarity(desc_embedding, cat_emb)
            results.append({
                "id": item["id"],
                "title": item["title"],
                "description": item["description"],
                "similarity": sim
            })

    # 類似度順にソートして上位5件を表示
    results.sort(key=lambda x: x["similarity"], reverse=True)
    print("【マッチング結果（上位5件）】")
    for r in results[:5]:
        print(f"- タイトル: {r['title']} (ID: {r['id']}) 類似度: {r['similarity']:.4f}")
        print(f"  説明: {r['description']}\n")

    logging.info("処理終了")

# ===== Llama3.3 への問い合わせ =====
def call_llama_for_category(user_input: str) -> dict:
    prompt = PROMPT_TEMPLATE.format(user_text=user_input)
    payload = {
        "model": LLAMA_MODEL,
        "prompt": prompt,
        "temperature": 0.2,
        "max_tokens": 512,
        "stream": False
    }
    try:
        response = requests.post(
            LLAMA_API_URL,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        # 応答を生テキストとして取得
        raw_response_text = response.text
        logging.info(f"LLM生レスポンス: {raw_response_text}")

        # 生レスポンスから generated_text 部分を取得
        # ここでは API が { "generated_text": "・・・" } の形式を返すと仮定
        try:
            parsed = json.loads(raw_response_text)
            text_output = parsed.get("response", "")
        except json.JSONDecodeError:
            text_output = raw_response_text

        # text_output から JSON 部分のみを抽出
        json_str = extract_json_from_text(text_output)
        if json_str:
            parsed_json = json.loads(json_str)
            return parsed_json
        else:
            return {}
    except requests.RequestException as e:
        logging.error(f"LLM呼び出し失敗: {e}")
        return {}
    except Exception as e:
        logging.error(f"予期せぬエラー: {e}")
        return {}

# ===== JSON 部分を抽出する関数 =====
def extract_json_from_text(text: str) -> str:
    """
    LLMの応答テキストから、"categoryname" と "description" を含む JSON 部分を正規表現で抽出する。
    複数候補がある場合は最初の1件を返す。
    """
    # 改行を含む可能性を考慮して DOTALL オプションを付与
    pattern = r'(\{[^}]*"categoryname"[^}]*"description"[^}]*\})'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        candidate = matches[0].strip()
        try:
            # 正しくパースできるかチェック
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    # 候補が見つからなかった場合は、最初の { から最後の } を抜き出すフォールバック
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx:end_idx + 1].strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            return ""
    return ""

# ===== 埋め込みAPI呼び出し関数 =====
def call_embed_api(text: str) -> list:
    try:
        response = requests.post(
            EMBED_API_URL,
            json={"model": EMBED_MODEL, "input": text},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        return data["embeddings"][0]
    except requests.RequestException as e:
        logging.error(f"埋め込みAPI呼び出し失敗: {e}")
        return None
    except Exception as e:
        logging.error(f"予期せぬエラー: {e}")
        return None

if __name__ == "__main__":
    main()

