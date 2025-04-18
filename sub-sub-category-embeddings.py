#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def load_json(filename):
    """指定した JSON ファイルを読み込んで返す"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_embeddings(embeddings_filename):
    """
    embeddings.jsonl を読み込み、各行の id をキー、embedding を値とする辞書を生成する。
    各行は {"id": <id>, "embedding": ...} の形式を仮定する。
    """
    embeddings = {}
    with open(embeddings_filename, 'r', encoding='utf-8') as f:
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

def gather_descendants(node, embeddings):
    """
    指定ノードのサブカテゴリー（sub-category）以下を再帰的に走査し、
    各ノードについて、id, categoryname, description, embedding を含む辞書を返すリストを生成する。
    なお、ここでは node の直下の子ノード群から収集するため、
    node 自身は含めず、その下層（第３階層以降）を収集する前提です。
    """
    results = []
    children = node.get("sub-category", [])
    for child in children:
        rec_id = child.get("id")
        rec = {
            "id": rec_id,
            "categoryname": child.get("categoryname"),
            "description": child.get("description", ""),
            # embeddings から id に対応する embedding を取り出す（なければ None）
            "embedding": embeddings.get(rec_id)
        }
        results.append(rec)
        # 子ノードの下層も再帰的に収集
        results.extend(gather_descendants(child, embeddings))
    return results

def main():
    merged_filename = "merged_with_description.json"
    embeddings_filename = "embeddings.jsonl"

    # JSONファイル群を読み込み
    merged_data = load_json(merged_filename)
    embeddings_dict = load_embeddings(embeddings_filename)

    # merged_with_description.json はトップレベルがリストになっている前提
    # 各トップレベルノード内の「第２階層」＝ top_node["sub-category"] の各要素について処理する
    for top_node in merged_data:
        second_level = top_node.get("sub-category", [])
        for second in second_level:
            second_id = second.get("id")
            second_name = second.get("categoryname")
            if second_id is None or second_name is None:
                continue  # id や categoryname がない場合はスキップ

            # 第二階層の下（第３階層以降）の全てのカテゴリーを収集する
            # 注意：ここでは第二階層そのものは除外し、その下にあるノードのみを対象にする
            descendants = gather_descendants(second, embeddings_dict)

            # ファイル名を「{id}-{categoryname}.jsonl」とする。ファイル名に不都合な文字があれば適宜置換してください。
            filename = f"./embeddings/{second_id}-{second_name}.jsonl"
            
            # ファイル出力（各行が 1 つの JSON オブジェクト）
            with open(filename, 'w', encoding='utf-8') as f_out:
                for record in descendants:
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"出力完了: {filename}")

if __name__ == "__main__":
    main()

