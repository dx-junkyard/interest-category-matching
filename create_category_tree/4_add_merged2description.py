import json
import requests
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ===== Ollama API設定 =====
LLAMA_API_URL = "http://localhost:11434/api/generate"  # 適宜変更
LLAMA_MODEL = "llama3.3:latest"                        # 適宜変更

# ===== LLM呼び出し用プロンプト =====
PROMPT_TEMPLATE = """以下のカテゴリー名について、簡潔な説明文を日本語で作成してください。

カテゴリー名: {category_name}

出力は説明文のみを返してください。
"""

def call_ollama_for_description(category_name: str) -> str:
    """
    Ollama APIに問い合わせ、カテゴリー名に対する説明文を生成する。
    """
    prompt = PROMPT_TEMPLATE.format(category_name=category_name)
    payload = {
        "model": LLAMA_MODEL,
        "prompt": prompt,
        "temperature": 0.2,
        "max_tokens": 256,
        "stream": False
    }
    try:
        response = requests.post(LLAMA_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        raw_response_text = response.text.strip()
        logging.info(f"[LLM raw] {raw_response_text}")
        try:
            data = json.loads(raw_response_text)
            if isinstance(data, dict) and "response" in data:
                return data["response"].strip()
            if isinstance(data, list):
                return "".join(item.get("response", "") for item in data).strip()
            return raw_response_text
        except json.JSONDecodeError:
            return raw_response_text
    except requests.RequestException as e:
        logging.error(f"Ollama API呼び出し失敗: {e}")
        return ""
    except Exception as e:
        logging.error(f"予期せぬエラー: {e}")
        return ""

def collect_ordered_unique_categories(data: list) -> list:
    """
    JSONツリーを再帰的に走査し、最初に出現した順序を保持した状態で
    全てのカテゴリー名を収集する（重複を排除）。
    """
    seen = set()
    ordered_categories = []

    def traverse(node: dict):
        cat = node.get("categoryname", "")
        if cat and cat not in seen:
            ordered_categories.append(cat)
            seen.add(cat)
        for child in node.get("sub-category", []):
            traverse(child)

    for node in data:
        traverse(node)
    return ordered_categories

def update_tree_with_descriptions(node: dict, description_map: dict, id_map: dict):
    """
    各ノードの categoryname をキーにして、description_mapから説明文を取得し、
    ノードの "description" キーに設定する。サブカテゴリーも再帰的に処理する。
    """
    cat = node.get("categoryname", "")
    node["description"] = description_map.get(cat, "")
    node["id"] = id_map.get(cat, None)
    for child in node.get("sub-category", []):
        update_tree_with_descriptions(child, description_map, id_map)

def load_progress(progress_path: Path) -> list:
    """
    進捗ファイルが存在すればその内容をリストとして返す。
    """
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            progress_list = json.load(f)
        logging.info(f"進捗ファイル {progress_path} を読み込みました。")
        return progress_list
    except Exception as e:
        logging.error(f"進捗ファイルの読み込みエラー: {e}")
        return []

def save_progress(progress_path: Path, cat_info_list: list):
    """
    進捗ファイルに cat_info_list を上書き保存する。
    """
    try:
        with open(progress_path, "w", encoding="utf-8") as pf:
            json.dump(cat_info_list, pf, ensure_ascii=False, indent=2)
        logging.info(f"進捗ファイル {progress_path} に保存しました。")
    except Exception as e:
        logging.error(f"進捗ファイルの保存エラー: {e}")

def main():
    progress_file = Path("./progress.json")

    # --- 1) merged.json の読み込み ---
    merged_file = Path("./merged.json")
    with open(merged_file, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    # --- 2) 進捗ファイルの存在確認 ---
    if progress_file.exists():
        # 既存の progress.json をそのまま順序維持して読み込む
        cat_info_list = load_progress(progress_file)
        logging.info("進捗ファイルが存在するため、既存の項目順をそのまま使用します。")
    else:
        # 新規作成の場合：JSONツリーから順序を保持してユニークなカテゴリーを抽出
        ordered_categories = collect_ordered_unique_categories(merged_data)
        cat_info_list = [
            {"id": idx + 1, "categoryname": cat, "description": ""}
            for idx, cat in enumerate(ordered_categories)
        ]
        logging.info("進捗ファイルが存在しなかったため、新たに進捗リストを作成しました。")

    # --- 3) 処理再開位置の特定 ---
    resume_index = None
    for idx, item in enumerate(cat_info_list):
        if item["description"] == "":
            resume_index = idx
            break

    if resume_index is None:
        logging.info("すでに全ての項目の説明文が生成されています。")
        resume_index = len(cat_info_list)
    else:
        logging.info(
            f"処理再開位置: {resume_index + 1} 番目の項目（カテゴリー名: {cat_info_list[resume_index]['categoryname']}）から再開します。"
        )

    # --- 4) 未処理のカテゴリーに対して説明文生成 ---
    processed_count = 0  # この変数は今回の実行分のカウント
    total = len(cat_info_list)
    for idx in range(resume_index, total):
        cat_info = cat_info_list[idx]
        if cat_info["description"] == "":
            desc = call_ollama_for_description(cat_info["categoryname"])
            cat_info["description"] = desc
            processed_count += 1
            logging.info(f"Processed {idx + 1} / {total}: {cat_info['categoryname']}")
        # 100回の処理ごとに progress.json に上書き保存
        if processed_count > 0 and processed_count % 100 == 0:
            save_progress(progress_file, cat_info_list)
    # 最終状態も保存
    save_progress(progress_file, cat_info_list)

    # --- 5) 説明文マッピングを作成し、元のJSONツリーに反映 ---
    description_map = {item["categoryname"]: item["description"] for item in cat_info_list}
    id_map = {item["categoryname"]: item["id"] for item in cat_info_list}  # 追加：idマッピングの作成
    for node in merged_data:
        update_tree_with_descriptions(node, description_map, id_map)

    output_file = Path("./merged_with_description.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    logging.info(f"説明文付きJSONを {output_file} として出力しました。")

if __name__ == "__main__":
    main()

