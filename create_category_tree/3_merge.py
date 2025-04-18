import json

# categories.jsonの読み込み
with open("./categories.json", "r", encoding="utf-8") as f:
    categories_data = json.load(f)

# category_tree.jsonの読み込み
with open("./category_tree.json", "r", encoding="utf-8") as f:
    tree_data = json.load(f)

def dict_to_tree(d):
    """
    category_tree.json 内の辞書構造を再帰的に
    [{"categoryname": <キー>, "sub-category": [...]}] の形式へ変換する。
    """
    result = []
    for key, val in d.items():
        # key がサブカテゴリー名、val がさらに入れ子になった辞書
        children = dict_to_tree(val) if isinstance(val, dict) and val else []
        node = {
            "categoryname": key,
            "sub-category": children
        }
        result.append(node)
    return result

merged = []

# categories.json の最上位カテゴリーを走査
for top_cat, top_cat_val in categories_data.items():
    top_cat_obj = {
        "categoryname": top_cat,       # メインカテゴリー名
        "sub-category": []
    }
    # 各メインカテゴリーに紐づく subcategories を処理
    for sub in top_cat_val.get("subcategories", []):
        sub_cat_name = sub["category"]
        # category_tree.jsonのトップレベルに同名キーがあれば取得
        sub_cat_tree = tree_data.get(sub_cat_name, {})
        sub_cat_obj = {
            "categoryname": sub_cat_name,
            "sub-category": dict_to_tree(sub_cat_tree) if isinstance(sub_cat_tree, dict) else []
        }
        top_cat_obj["sub-category"].append(sub_cat_obj)
    
    merged.append(top_cat_obj)

# 結果をファイルに書き出す（フルのJSON構造）
with open("./merged.json", "w", encoding="utf-8") as out_f:
    json.dump(merged, out_f, ensure_ascii=False, indent=2)

# 実行後、同じフォルダに merged.json が生成されます。
# 完成したJSON全体を確認したい場合は、必要に応じて外部エディタなどで開いてください。

