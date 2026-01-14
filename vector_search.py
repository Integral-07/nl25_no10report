import math
from collections import defaultdict
from typing import Dict, List, Tuple, Set

class VectorSpaceModel:

    def __init__(self):
        # 文書ごとの単語頻度: {文書ID: {単語: 出現回数}}
        self.doc_freq: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # 単語が出現する文書数: {単語: 文書数}
        self.word_doc_count: Dict[str, int] = defaultdict(int)
        # 総文書数
        self.total_docs: int = 0
        # 全単語集合
        self.vocabulary: Set[str] = set()

    def load_freq_file(self, filepath: str):

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # "文書番号 単語 出現回数" 形式の行をパース
                parts = line.split()
                if len(parts) >= 3:
                    doc_id = int(parts[0])
                    word = parts[1]
                    count = int(parts[2])

                    self.doc_freq[doc_id][word] = count
                    self.vocabulary.add(word)

    def build_index(self):

        self.total_docs = len(self.doc_freq)

        # 各単語が出現する文書数をカウント
        for doc_id, words in self.doc_freq.items():
            for word in words.keys():
                self.word_doc_count[word] += 1

        print(f"総文書数: {self.total_docs}")
        print(f"語彙サイズ: {len(self.vocabulary)}")

    def calculate_idf(self, word: str) -> float:

        if word not in self.word_doc_count:
            return 0.0
        return math.log(self.total_docs / self.word_doc_count[word])

    def get_tfidf_vector(self, doc_id: int) -> Dict[str, float]:

        tfidf_vector = {}
        for word, tf in self.doc_freq[doc_id].items():
            idf = self.calculate_idf(word)
            tfidf_vector[word] = tf * idf
        return tfidf_vector

    def load_queries(self, filepath: str) -> Dict[int, List[str]]:

        queries = {}
        with open(filepath, 'r', encoding='utf-8') as f:
            for query_id, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                words = line.split()
                queries[query_id] = words

        return queries

    def get_query_vector(self, query_words: List[str]) -> Dict[str, int]:
        """
        検索質問の単語出現回数ベクトルを作成
        """
        query_vector = defaultdict(int)
        for word in query_words:
            query_vector[word] += 1
        return dict(query_vector)

    def cosine_similarity(self, query_vec: Dict[str, int], doc_tfidf: Dict[str, float]) -> float:
        """
        余弦尺度による類似度計算
        """
        # 内積計算
        dot_product = 0.0
        for word, query_freq in query_vec.items():
            if word in doc_tfidf:
                dot_product += query_freq * doc_tfidf[word]

        # ベクトルの大きさ計算
        query_norm = math.sqrt(sum(v**2 for v in query_vec.values()))
        doc_norm = math.sqrt(sum(v**2 for v in doc_tfidf.values()))

        if query_norm == 0 or doc_norm == 0:
            return 0.0

        return dot_product / (query_norm * doc_norm)

    def jaccard_coefficient(self, query_vec: Dict[str, int], doc_tfidf: Dict[str, float]) -> float:
        """
        Jaccard係数による類似度計算
        """
        query_words = set(query_vec.keys())
        doc_words = set(doc_tfidf.keys())

        intersection = len(query_words & doc_words)
        union = len(query_words | doc_words)

        if union == 0:
            return 0.0

        return intersection / union

    def search(self, query_words: List[str], method: str = 'cosine') -> List[Tuple[int, float]]:
 
        query_vec = self.get_query_vector(query_words)
        results = []

        for doc_id in self.doc_freq.keys():
            doc_tfidf = self.get_tfidf_vector(doc_id)

            if method == 'cosine':
                score = self.cosine_similarity(query_vec, doc_tfidf)
            elif method == 'jaccard':
                score = self.jaccard_coefficient(query_vec, doc_tfidf)
            else:
                raise ValueError(f"Unknown method: {method}")

            results.append((doc_id, score))

        # スコアの降順でソート
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def main():
    """メイン処理"""
    print("="*60)
    print("ベクトル空間モデルによる文書検索システム")
    print("="*60)
    print()

    # ベクトル空間モデルの初期化
    vsm = VectorSpaceModel()

    # データの読み込み
    print("データを読み込み中...")
    vsm.load_freq_file('mai.freq')      # 毎日新聞 (記事1-9)
    vsm.load_freq_file('nikkei.freq')   # 日経新聞 (記事10)

    # インデックスの構築
    vsm.build_index()
    print()

    # 検索質問の読み込み
    queries = vsm.load_queries('query.freq')
    print(f"検索質問数: {len(queries)}")
    print()

    # 各検索質問に対して検索を実行
    for query_id, query_words in sorted(queries.items()):
        if not query_words:
            continue

        print("="*60)
        print(f"検索質問 {query_id}: {' '.join(query_words[:10])}{'...' if len(query_words) > 10 else ''}")
        print("="*60)

        # 余弦尺度による検索
        print("\n【余弦尺度 (Cosine Similarity)】")
        cosine_results = vsm.search(query_words, method='cosine')

        print(f"{'順位':<4} {'文書ID':<8} {'類似度スコア':<15}")
        print("-" * 30)
        for rank, (doc_id, score) in enumerate(cosine_results, 1):
            print(f"{rank:<4} {doc_id:<8} {score:<15.6f}")

        # Jaccard係数による検索
        print("\n【Jaccard係数 (Jaccard Coefficient)】")
        jaccard_results = vsm.search(query_words, method='jaccard')

        print(f"{'順位':<4} {'文書ID':<8} {'類似度スコア':<15}")
        print("-" * 30)
        for rank, (doc_id, score) in enumerate(jaccard_results, 1):
            print(f"{rank:<4} {doc_id:<8} {score:<15.6f}")

        print()


if __name__ == '__main__':
    main()
