#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import requests
import re
import numpy as np
from numpy.linalg import norm
from pathlib import Path

# ===== ログ設定 =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ===== 定数 =====
LLAMA_API_URL = "http://localhost:11434/api/generate"       # LLM のエンドポイント
LLAMA_MODEL = "llama3.3:latest"                               # LLM のモデル名

EMBED_API_URL = "http://localhost:11434/api/embed"           # 埋め込み API のエンドポイント
EMBED_MODEL = "kun432/cl-nagoya-ruri-large:latest"            # 埋め込み API のモデル名

PROMPT_TEMPLATE = """次の文章から、該当するメインカテゴリーおよびサブカテゴリーを推測し、サブカテゴリーの説明文も作成せよ。
回答は以下の JSON 配列形式で、メインカテゴリーとそのサブカテゴリー（説明付き）を返すこと。
[
  {{
    "categoryname": "（メインカテゴリー名）",
    "sub-category": [
      {{
        "categoryname": "（サブカテゴリー名）",
        "description": "（サブカテゴリーの説明）"
      }}
    ]
  }}
]
文章:
{user_text}
"""

# ===== ユーティリティクラス =====
class Utils:
    @staticmethod
    def sanitize_filename(name: str) -> str:
        """
        ファイル名に使える文字列に変換。
        英数字、ハイフン、アンダースコアに加え、日本語（漢字、ひらがな、カタカナ）を許容する。
        """
        return re.sub(r'[^\w\u4e00-\u9faf\u3040-\u309f\u30a0-\u30ff\-]', '_', name)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """2つのベクトル間のコサイン類似度を計算する"""
        return float(np.dot(a, b) / (norm(a) * norm(b)))


# ===== LLM API 呼び出しクラス =====
class LLMClient:
    def __init__(self, api_url=LLAMA_API_URL, model=LLAMA_MODEL):
        self.api_url = api_url
        self.model = model

    def get_categories(self, user_text: str):
        prompt = PROMPT_TEMPLATE.format(user_text=user_text)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": 0.2,
            "max_tokens": 512,
            "stream": False
        }
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            raw_response_text = response.text.strip()
            logging.info(f"LLM 生レスポンス: {raw_response_text}")
            try:
                parsed = json.loads(raw_response_text)
                if isinstance(parsed, dict) and "response" in parsed:
                    text_output = parsed.get("response", "")
                    return json.loads(text_output)
                else:
                    return parsed
            except json.JSONDecodeError:
                logging.error("LLMの返答のパースに失敗しました。")
                return None
        except requests.RequestException as e:
            logging.error(f"LLM呼び出し失敗: {e}")
            return None


# ===== 埋め込み API 呼び出しクラス =====
class EmbedClient:
    def __init__(self, api_url=EMBED_API_URL, model=EMBED_MODEL):
        self.api_url = api_url
        self.model = model

    def get_embedding(self, text: str) -> list:
        try:
            payload = {"model": self.model, "input": text}
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["embeddings"][0]
        except requests.RequestException as e:
            logging.error(f"埋め込みAPI呼び出し失敗: {e}")
            return None
        except Exception as e:
            logging.error(f"予期せぬエラー: {e}")
            return None


# ===== JSONL ファイル読み込みクラス =====
class FileLoader:
    @staticmethod
    def load_jsonl(filepath: Path) -> list:
        records = []
        if not filepath.exists():
            logging.error(f"ファイルが存在しません: {filepath}")
            return records
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        return records


# ===== カテゴリー検索クラス =====
class CategorySearcher:
    def __init__(self, llm_client: LLMClient, embed_client: EmbedClient, sub_embed_filepath: Path):
        self.llm_client = llm_client
        self.embed_client = embed_client
        self.sub_embed_filepath = sub_embed_filepath

    def get_top_sub_candidates(self, sub_cat_name: str, sub_cat_desc: str, top_n: int = 3) -> list:
        """
        embeddings/sub-category-embedding.jsonl から、予測されたサブカテゴリー名・説明に対する上位候補（上位 top_n 件）を返す。
        完全一致する候補は類似度 1.0 として評価する。
        """
        candidates = FileLoader.load_jsonl(self.sub_embed_filepath)
        if not candidates:
            logging.error("embeddings/sub-category-embedding.jsonl の読み込みに失敗しました。")
            return []

        sub_emb = self.embed_client.get_embedding(sub_cat_desc)
        if sub_emb is None:
            logging.error("サブカテゴリー説明文の埋め込みに失敗しました。")
            return []
        sub_emb = np.array(sub_emb, dtype="float32")

        for candidate in candidates:
            candidate_name = candidate.get("categoryname", "").strip()
            if candidate_name == sub_cat_name:
                candidate["similarity"] = 1.0
            else:
                candidate_emb = candidate.get("embedding")
                if candidate_emb is None:
                    candidate["similarity"] = 0.0
                else:
                    candidate_emb = np.array(candidate_emb, dtype="float32")
                    candidate["similarity"] = Utils.cosine_similarity(sub_emb, candidate_emb)
        candidates.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        top_candidates = candidates[:top_n]
        for cand in top_candidates:
            logging.info(f"サブ候補: ID={cand.get('id')}, name={cand.get('categoryname')}, similarity={cand.get('similarity'):.4f}")
        return top_candidates

    def get_top_child_candidates(self, predicted_child: str, user_embedding: np.ndarray, sub_candidate: dict, top_n: int = 5) -> list:
        """
        指定したサブカテゴリー候補の対象ファイル（./embeddings/{id}-{sanitized_name}.jsonl）から
        ユーザー入力の埋め込みと候補の埋め込みの類似度により、上位 top_n 件の子候補を返す。
        なお、候補の categoryname と predicted_child が完全一致する場合、類似度は 1.0 とする。
        """
        best_sub_id = sub_candidate.get("id")
        best_sub_name = sub_candidate.get("categoryname")
        sanitized_sub_name = Utils.sanitize_filename(best_sub_name)
        target_filepath = Path(f"./embeddings/{best_sub_id}-{sanitized_sub_name}.jsonl")
        child_candidates = FileLoader.load_jsonl(target_filepath)
        if not child_candidates:
            logging.error(f"{target_filepath} の読み込みに失敗しました。")
            return []

        # まず、完全一致候補（candidate の categoryname と predicted_child が完全一致するもの）を探す
        exact_matches = [record for record in child_candidates if record.get("categoryname", "").strip() == predicted_child.strip()]
        if exact_matches:
            for record in exact_matches:
                record["child_similarity"] = 1.0
            exact_matches.sort(key=lambda x: x.get("child_similarity", 0.0), reverse=True)
            logging.info("子候補の完全一致が見つかりました。")
            for child in exact_matches:
                logging.info(f"子完全一致候補: ID={child.get('id')}, name={child.get('categoryname')}, child_similarity=1.0000")
            return exact_matches[:top_n]

        # 完全一致がなければ embedding による類似度計算
        for record in child_candidates:
            rec_emb = record.get("embedding")
            if rec_emb is None:
                record["child_similarity"] = 0.0
            else:
                rec_emb = np.array(rec_emb, dtype="float32")
                record["child_similarity"] = Utils.cosine_similarity(user_embedding, rec_emb)
        child_candidates.sort(key=lambda x: x.get("child_similarity", 0.0), reverse=True)
        top_children = child_candidates[:top_n]
        for child in top_children:
            logging.info(f"子候補: ID={child.get('id')}, name={child.get('categoryname')}, child_similarity={child.get('child_similarity'):.4f}")
        return top_children

    def remove_duplicates(self, candidates: list) -> list:
        """
        (id, categoryname, description) の組み合わせで重複を除外してリストを返す。
        """
        unique = {}
        for cand in candidates:
            key = (cand.get("id"), cand.get("categoryname"), cand.get("description"))
            if key not in unique:
                unique[key] = cand
        return list(unique.values())

    def process(self, user_input: str) -> list:
        """
        ユーザー入力に対して、LLM によるカテゴリー推測、サブ候補（上位３件）取得、
        各サブ候補ごとに対象の子候補（上位５件）を選出し、最終的に得られた全候補の中から
        embedding 類似度に基づき上位３件（重複除外）を返す。
        """
        llm_result = self.llm_client.get_categories(user_input)
        if not llm_result or not isinstance(llm_result, list):
            logging.error("LLMの返答が正しい JSON 配列形式ではありません。")
            return []

        # ここでは最初のメインカテゴリー内の最初のサブカテゴリーを採用
        main_cat = llm_result[0]
        sub_categories = main_cat.get("sub-category", [])
        if not sub_categories:
            logging.error("サブカテゴリーが取得できませんでした。")
            return []
        sub_cat_info = sub_categories[0]
        sub_cat_name = sub_cat_info.get("categoryname", "").strip()
        sub_cat_desc = sub_cat_info.get("description", "").strip()
        if not sub_cat_name or not sub_cat_desc:
            logging.error("サブカテゴリーの名称または説明が空です。")
            return []
        logging.info(f"予測サブカテゴリー: {sub_cat_name}")
        logging.info(f"サブカテゴリー説明: {sub_cat_desc}")

        top_sub_candidates = self.get_top_sub_candidates(sub_cat_name, sub_cat_desc, top_n=3)
        if not top_sub_candidates:
            logging.error("サブカテゴリー候補が見つかりませんでした。")
            return []

        user_input_emb = self.embed_client.get_embedding(user_input)
        if user_input_emb is None:
            logging.error("ユーザー入力の埋め込みに失敗しました。")
            return []
        user_input_emb = np.array(user_input_emb, dtype="float32")

        aggregated_children = []
        for sub_candidate in top_sub_candidates:
            # ※ ここでは、子候補の検索に予測されたサブカテゴリー名を使用する
            children = self.get_top_child_candidates(sub_cat_name, user_input_emb, sub_candidate, top_n=5)
            aggregated_children.extend(children)

        # 重複候補の除外
        unique_children = self.remove_duplicates(aggregated_children)

        if not unique_children:
            logging.error("子カテゴリー候補が一件も見つかりませんでした。")
            return []

        unique_children.sort(key=lambda x: x.get("child_similarity", 0.0), reverse=True)
        top_final = unique_children[:3]
        logging.info("最終候補:")
        for candidate in top_final:
            logging.info(f"ID={candidate.get('id')}, name={candidate.get('categoryname')}, similarity={candidate.get('child_similarity'):.4f}")
        return top_final


# ===== メイン処理 =====
def main():
    #user_input = "私は大学時代に量子論を勉強し、デコヒーレンスについて独自に研究していました。"
    user_input = "現在開催されている大阪万博に行ってみようと思います。"
    logging.info(f"ユーザー入力: {user_input}")

    llm_client = LLMClient()
    embed_client = EmbedClient()
    sub_embed_filepath = Path("embeddings/sub-category-embedding.jsonl")

    searcher = CategorySearcher(llm_client, embed_client, sub_embed_filepath)
    final_results = searcher.process(user_input)

    if final_results:
        print("最終マッチ候補 (上位 3 件):")
        for res in final_results:
            print(json.dumps({
                "id": res.get("id"),
                "categoryname": res.get("categoryname"),
                "description": res.get("description"),
                "similarity": res.get("child_similarity")
            }, ensure_ascii=False, indent=2))
    else:
        print("候補が見つかりませんでした。")


if __name__ == "__main__":
    main()

