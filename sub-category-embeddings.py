#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

def load_json_file(filename):
    """指定したファイルから JSON データを読み込む"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_embeddings(filename):
    """
    embeddings.jsonl を読み込み、各行の id をキー、embedding を値とする辞書を生成する。
    各行は {"id": <id>, "embedding": <embedding>} の形式を仮定する。
    """
    embeddings = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    rec_id = record.get("id")
                    if rec_id is not None:
                        embeddings[rec_id] = record.get("embedding")
                except json.JSONDecodeError:
                    continue
    return embeddings

def extract_level2_categories(merged_data):
    """
    merged_with_description.json のトップレベルの各ノードの sub-category 内の
    各サブカテゴリー（第２階層）の情報（categoryname と id）を抽出する。
    """
    level2_categories = []
    for top_node in merged_data:
        sub_categories = top_node.get("sub-category", [])
        for sub in sub_categories:
            cat_name = sub.get("categoryname")
            cat_id = sub.get("id")
            if cat_name is not None and cat_id is not None:
                level2_categories.append({"id": cat_id, "categoryname": cat_name})
    return level2_categories

def main():
    merged_filename = "merged_with_description.json"
    embeddings_filename = "embeddings.jsonl"
    output_filename = "sub-category-embedding.jsonl"

    # merged_with_description.json を読み込む
    merged_data = load_json_file(merged_filename)
    # embeddings.jsonl を id をキーとした辞書として読み込む
    embeddings_dict = load_embeddings(embeddings_filename)
    # 第２階層のサブカテゴリー情報を抽出する
    level2_categories = extract_level2_categories(merged_data)

    # 各サブカテゴリーに対して、id と categoryname、対応する embedding を出力ファイルへ書き出す
    with open(output_filename, 'w', encoding='utf-8') as out_file:
        for category in level2_categories:
            cat_id = category["id"]
            embedding = embeddings_dict.get(cat_id)  # 存在しなければ None
            out_record = {
                "id": cat_id,
                "categoryname": category["categoryname"],
                "embedding": embedding
            }
            out_file.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    print(f"出力完了: {output_filename}")

if __name__ == "__main__":
    main()

