import mysql.connector
from collections import defaultdict
import json

# MySQL 接続設定
config = {
    'host': 'localhost',
    'user': 'wiki',
    'password': 'wiki',
    'database': 'wikidb',
    'port': 3306,
    'charset': 'utf8'
}

root_categories = [
  # 社会・公共
    "政治","行政","政策",
    "法律","司法","倫理",
    "経済","金融","産業",
    "国","外交","安全保障",
    "労働",
    "社会","人権","福祉","ジェンダー",
    "防災","危機管理",
    "思想","宗教","哲学",
  # 自然・科学・技術
    "自然","環境","生態学","気候変動",
    "宇宙","天文学",
    "数学","物理学","化学","生物学",
    "医療","健康","生命科学",
    "農業","飲食","エネルギー",
    "情報技術",
    "工学","建築","デザイン","製造",
    "情報学","統計","解析",
  # 文化・芸術・表現
    "文学","言語","語学",
    "歴史","考古学","文化人類学",
    "芸術",
    "ファッション","デザイン","建築",
    "食文化",
    "娯楽","サブカルチャー",
    "マスメディア","ジャーナリズム","パブリック・リレーションズ",
    "スポーツ","舞台芸術","ダンス","武道",
  # 生活・実践・ライフスキル
    "生活","家事","育児","介護",
    "住宅","DIY","インテリア",
    "健康","美容","フィットネス",
    "教育","学習","資格","能力開発",
    "趣味","レジャー","旅行","野外活動",
    "恋愛","結婚","家族","人間関係",
    "資産運用","ビジネススキル",
    "プロジェクトマネジメント","問題解決","ビジネススキル",
  # 心理・精神・内面世界
    "心理学","カウンセリング","メンタルヘルス",
    "哲学","倫理","死生観",
    "分析心理学","幸せ","クオリティ・オブ・ライフ",
    "瞑想","マインドフルネス","動機づけ",
    "コミュニケーション","人間関係"
]


def connect_db():
    return mysql.connector.connect(**config)

def category_exists(cursor, category_name):
    cursor.execute("""
        SELECT page_id FROM page
        WHERE page_namespace = 14 AND page_title = %s
    """, (category_name.replace(' ', '_'),))
    return cursor.fetchone() is not None

def main():
    conn = connect_db()
    cursor = conn.cursor()

    print("=== カテゴリ存在チェック ===")
    for cat in root_categories:
        exists = category_exists(cursor, cat)
        mark = "✅" if exists else "❌"
        print(f"{mark} {cat}")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()

