import json
import requests
import uuid
from pathlib import Path

# ===== 設定項目 =====
JSON_FILE = Path("categories.json")  # 先のJSONを保存しているファイルパス
OUTPUT_FILE = Path("category_embeddings.jsonl")
LOG_INTERVAL = 10
OLLAMA_URL = "http://localhost:11434/api/embed"
OLLAMA_MODEL = "kun432/cl-nagoya-ruri-large:latest"
MAX_PROMPT_LENGTH = 500  # テキストが長すぎる場合に切り捨てる最大文字数

# ===== 実行処理 =====
def main():
    total_count = 0
    
    # カテゴリJSONを読み込み
    with open(JSON_FILE, "r", encoding="utf-8") as f:
        categories_data = json.load(f)

    # JSONL出力用ファイルを開く
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        # メインカテゴリーを順番に処理
        for main_category_name, main_category_info in categories_data.items():
            # メインカテゴリー用IDを発行
            main_cat_id = str(uuid.uuid4())
            # テキストを "カテゴリ名 + 説明" の形式で作成
            main_text = f"{main_category_name}。{main_category_info['description']}"
            if len(main_text) > MAX_PROMPT_LENGTH:
                main_text = main_text[:MAX_PROMPT_LENGTH]

            # embedding API コール
            embedding = call_ollama_embed(main_text)
            if embedding is not None:
                # JSONL形式で出力
                out_f.write(json.dumps({
                    "id": main_cat_id,
                    "title": main_category_name,
                    "description": main_category_info['description'],
                    "embedding": embedding
                }, ensure_ascii=False) + "\n")
                total_count += 1

                # LOG_INTERVALごとに進捗表示
                if total_count % LOG_INTERVAL == 0:
                    print(f"Processed {total_count} categories...")

            # サブカテゴリーの処理
            if "subcategories" in main_category_info:
                for subcategory_info in main_category_info["subcategories"]:
                    sub_cat_id = str(uuid.uuid4())  # サブカテゴリごとにユニークなIDを付与
                    category_name = subcategory_info["category"]
                    description = subcategory_info["description"]

                    sub_text = f"{category_name}。{description}"
                    if len(sub_text) > MAX_PROMPT_LENGTH:
                        sub_text = sub_text[:MAX_PROMPT_LENGTH]

                    # embedding API コール
                    embedding = call_ollama_embed(sub_text)
                    if embedding is not None:
                        # JSONL形式で出力
                        out_f.write(json.dumps({
                            "id": sub_cat_id,
                            "title": category_name,
                            "description": description,
                            "embedding": embedding
                        }, ensure_ascii=False) + "\n")
                        total_count += 1

                        if total_count % LOG_INTERVAL == 0:
                            print(f"Processed {total_count} categories...")

    print(f"Embedding complete! Total processed: {total_count}")


def call_ollama_embed(text: str):
    """
    Ollama の /api/embed に対して embedding を取得するヘルパー関数。
    失敗時は None を返す。
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "input": text
            },
            timeout=30
        )
        response.raise_for_status()
        # Ollamaのレスポンスに応じてパース（下記は "embeddings" キーを前提）
        data = response.json()
        return data["embeddings"][0]
    except requests.RequestException as e:
        print(f"Failed embedding: {text[:30]}...")  # テキストの先頭30文字だけ表示
        if e.response is not None:
            print(f"Status Code: {e.response.status_code}")
            print(f"Response Body: {e.response.text}")
        else:
            print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Failed embedding (unexpected error): {text[:30]}... - {e}")
        return None


if __name__ == "__main__":
    main()

