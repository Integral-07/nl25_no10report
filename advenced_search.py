import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from vector_search import VectorSpaceModel


class ClusteredSearchModel(VectorSpaceModel):

    def __init__(self, idf_threshold: float = 1.0, top_k_terms: int = None):

        super().__init__()
        self.idf_threshold = idf_threshold
        self.top_k_terms = top_k_terms

        # 索引語選択用
        self.content_words: Set[str] = set()  # 高IDF値の内容語
        self.information_gain: Dict[str, float] = {}  # 各単語の情報利得
        self.selected_terms: Set[str] = set()  # 選択された索引語

        # クラスタリング用
        self.n_clusters: int = 0
        self.cluster_centers: List[Dict[str, float]] = []  # クラスタ中心
        self.doc_clusters: Dict[int, int] = {}  # 文書→クラスタID
        self.cluster_docs: Dict[int, List[int]] = defaultdict(list)  # クラスタID→文書リスト

    def build_index(self):
        """インデックス構築（拡張版）"""
        super().build_index()

        print("\n【工夫(a): 内容語の抽出】")
        self._extract_content_words()

        print("\n【工夫(b): 情報利得による索引語選択】")
        self._calculate_information_gain()
        self._select_index_terms()

    def _extract_content_words(self):

        for word in self.vocabulary:
            idf = self.calculate_idf(word)
            if idf >= self.idf_threshold:
                self.content_words.add(word)

        print(f"  全語彙数: {len(self.vocabulary)}")
        print(f"  内容語数 (IDF >= {self.idf_threshold}): {len(self.content_words)}")
        print(f"  除外された一般語: {len(self.vocabulary) - len(self.content_words)}")

    def _calculate_information_gain(self):

        # 各単語について情報利得を計算
        for word in self.content_words:
            # 単語を含む文書数と含まない文書数
            docs_with_word = self.word_doc_count[word]
            docs_without_word = self.total_docs - docs_with_word

            if docs_with_word == 0 or docs_without_word == 0:
                # 全文書に出現 or 全く出現しない場合は情報利得0
                self.information_gain[word] = 0.0
                continue

            # 確率の計算
            p_with = docs_with_word / self.total_docs
            p_without = docs_without_word / self.total_docs

            # エントロピーの計算
            entropy_with = -p_with * math.log2(p_with) if p_with > 0 else 0
            entropy_without = -p_without * math.log2(p_without) if p_without > 0 else 0

            # 情報利得の近似値
            ig = abs(0.5 - p_with)  # 0.5からの距離

            self.information_gain[word] = ig

        # 情報利得の統計を表示
        ig_values = list(self.information_gain.values())
        if ig_values:
            print(f"  情報利得の平均: {np.mean(ig_values):.4f}")
            print(f"  情報利得の最大: {np.max(ig_values):.4f}")
            print(f"  情報利得の最小: {np.min(ig_values):.4f}")

    def _select_index_terms(self):

        # 情報利得でソート
        sorted_terms = sorted(
            self.information_gain.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if self.top_k_terms is None:
            # 全ての内容語を使用
            self.selected_terms = self.content_words
            print(f"  選択された索引語数: {len(self.selected_terms)} (全内容語)")
        else:
            # 上位K個を選択
            self.selected_terms = set(
                word for word, _ in sorted_terms[:self.top_k_terms]
            )
            print(f"  選択された索引語数: {len(self.selected_terms)} (上位{self.top_k_terms}個)")

            # 上位5個の例を表示
            print("\n  【情報利得が高い上位5単語】")
            for i, (word, ig) in enumerate(sorted_terms[:5], 1):
                idf = self.calculate_idf(word)
                print(f"    {i}. {word}: IG={ig:.4f}, IDF={idf:.4f}")

    def get_filtered_tfidf_vector(self, doc_id: int) -> Dict[str, float]:

        full_vector = self.get_tfidf_vector(doc_id)
        return {
            word: tfidf
            for word, tfidf in full_vector.items()
            if word in self.selected_terms
        }

    def cluster_documents(self, n_clusters: int = 3, max_iterations: int = 10):

        print(f"\n【工夫(c): k-meansクラスタリング前処理】")
        print(f"  クラスタ数: {n_clusters}")

        self.n_clusters = n_clusters
        doc_ids = list(self.doc_freq.keys())

        # 各文書のTF*IDFベクトルを取得
        doc_vectors = {
            doc_id: self.get_filtered_tfidf_vector(doc_id)
            for doc_id in doc_ids
        }

        # 初期クラスタ中心をランダムに選択
        np.random.seed(42)
        initial_centers = np.random.choice(doc_ids, n_clusters, replace=False)
        self.cluster_centers = [
            doc_vectors[doc_id].copy() for doc_id in initial_centers
        ]

        # k-meansアルゴリズム
        for iteration in range(max_iterations):
            # 各文書を最も近いクラスタに割り当て
            old_clusters = self.doc_clusters.copy()
            self.doc_clusters.clear()

            for doc_id in doc_ids:
                doc_vec = doc_vectors[doc_id]

                # 各クラスタ中心との類似度を計算
                similarities = []
                for center in self.cluster_centers:
                    sim = self._cosine_similarity_dict(doc_vec, center)
                    similarities.append(sim)

                # 最も類似度の高いクラスタに割り当て
                cluster_id = np.argmax(similarities)
                self.doc_clusters[doc_id] = cluster_id

            # クラスタ中心を更新
            self.cluster_centers = []
            for cluster_id in range(n_clusters):
                cluster_doc_ids = [
                    doc_id for doc_id, cid in self.doc_clusters.items()
                    if cid == cluster_id
                ]

                if not cluster_doc_ids:
                    # 空のクラスタの場合はランダムな文書を中心にする
                    random_doc = np.random.choice(doc_ids)
                    center = doc_vectors[random_doc].copy()
                else:
                    # クラスタ内の文書の平均ベクトルを計算
                    center = self._calculate_centroid(
                        [doc_vectors[did] for did in cluster_doc_ids]
                    )

                self.cluster_centers.append(center)

            # 収束判定
            if old_clusters == self.doc_clusters:
                print(f"  収束しました (イテレーション {iteration + 1})")
                break

        # クラスタごとの文書リストを作成
        self.cluster_docs.clear()
        for doc_id, cluster_id in self.doc_clusters.items():
            self.cluster_docs[cluster_id].append(doc_id)

        # クラスタリング結果の統計を表示
        print("\n  【クラスタリング結果】")
        for cluster_id in range(n_clusters):
            docs_in_cluster = len(self.cluster_docs[cluster_id])
            print(f"    クラスタ {cluster_id}: {docs_in_cluster}文書")
            print(f"      文書ID: {sorted(self.cluster_docs[cluster_id])}")

    def _cosine_similarity_dict(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """2つの辞書形式ベクトル間の余弦類似度を計算"""
        dot_product = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in set(vec1) | set(vec2))
        norm1 = math.sqrt(sum(v**2 for v in vec1.values()))
        norm2 = math.sqrt(sum(v**2 for v in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _calculate_centroid(self, vectors: List[Dict[str, float]]) -> Dict[str, float]:
        """複数のベクトルの重心を計算"""
        centroid = defaultdict(float)
        n = len(vectors)

        if n == 0:
            return {}

        for vec in vectors:
            for word, value in vec.items():
                centroid[word] += value / n

        return dict(centroid)

    def search_with_clustering(self, query_words: List[str],
                              top_n_clusters: int = None) -> List[Tuple[int, float]]:

        # クエリベクトルを作成（内容語のみ）
        query_vec = self.get_query_vector(query_words)
        query_vec_filtered = {
            word: freq for word, freq in query_vec.items()
            if word in self.selected_terms
        }

        # クエリとTF*IDF重み付けしたクエリベクトルを作成
        query_tfidf = {
            word: freq * self.calculate_idf(word)
            for word, freq in query_vec_filtered.items()
        }

        # 各クラスタとの類似度を計算
        cluster_similarities = []
        for cluster_id, center in enumerate(self.cluster_centers):
            sim = self._cosine_similarity_dict(query_tfidf, center)
            cluster_similarities.append((cluster_id, sim))

        # 類似度でソート
        cluster_similarities.sort(key=lambda x: x[1], reverse=True)

        # 検索対象クラスタを選択
        if top_n_clusters is None:
            selected_clusters = [cid for cid, _ in cluster_similarities]
        else:
            selected_clusters = [cid for cid, _ in cluster_similarities[:top_n_clusters]]

        print(f"\n  検索対象クラスタ: {selected_clusters}")

        # 選択されたクラスタ内の文書のみを検索
        results = []
        searched_docs = 0

        for cluster_id in selected_clusters:
            for doc_id in self.cluster_docs[cluster_id]:
                doc_tfidf = self.get_filtered_tfidf_vector(doc_id)

                # 余弦類似度で評価
                score = self.cosine_similarity(query_vec_filtered, doc_tfidf)
                results.append((doc_id, score))
                searched_docs += 1

        print(f"  検索した文書数: {searched_docs}/{self.total_docs}")

        # スコアの降順でソート
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def search_with_ig_terms(self, query_words: List[str]) -> List[Tuple[int, float]]:

        # クエリベクトルを作成
        query_vec = self.get_query_vector(query_words)
        query_vec_filtered = {
            word: freq for word, freq in query_vec.items()
            if word in self.selected_terms
        }

        results = []

        for doc_id in self.doc_freq.keys():
            doc_tfidf = self.get_filtered_tfidf_vector(doc_id)
            score = self.cosine_similarity(query_vec_filtered, doc_tfidf)
            results.append((doc_id, score))

        # スコアの降順でソート
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def main():

    # モデルの初期化
    # idf_threshold: 内容語とみなすIDFの最小値
    # top_k_terms: 情報利得で選択する上位K個（Noneの場合は全て使用）
    model = ClusteredSearchModel(idf_threshold=1.0, top_k_terms=50)

    # データの読み込み
    print("データを読み込み中...")
    model.load_freq_file('mai.freq')
    model.load_freq_file('nikkei.freq')

    # インデックスの構築
    model.build_index()

    # クラスタリングの実行
    model.cluster_documents(n_clusters=3, max_iterations=10)
    print()

    # 検索質問の読み込み
    queries = model.load_queries('query.freq')
    print(f"\n検索質問数: {len(queries)}")
    print()

    # 各検索質問に対して検索を実行
    for query_id, query_words in sorted(queries.items()):
        if not query_words:
            continue

        print("="*70)
        print(f"検索質問 {query_id}: {' '.join(query_words[:10])}{'...' if len(query_words) > 10 else ''}")
        print("="*70)

        # 情報利得選択語のみを使用した検索
        print("\n【情報利得選択語のみを使用】")
        ig_results = model.search_with_ig_terms(query_words)

        print(f"{'順位':<4} {'文書ID':<8} {'類似度スコア':<15}")
        print("-" * 30)
        for rank, (doc_id, score) in enumerate(ig_results, 1):
            cluster_id = model.doc_clusters.get(doc_id, -1)
            print(f"{rank:<4} {doc_id:<8} {score:<15.6f}  [クラスタ{cluster_id}]")

        # クラスタリングを利用した検索（上位2クラスタのみ）
        print("\n【クラスタリング利用検索（上位2クラスタ）】")
        cluster_results = model.search_with_clustering(query_words, top_n_clusters=2)

        print(f"\n{'順位':<4} {'文書ID':<8} {'類似度スコア':<15}")
        print("-" * 30)
        for rank, (doc_id, score) in enumerate(cluster_results, 1):
            cluster_id = model.doc_clusters.get(doc_id, -1)
            print(f"{rank:<4} {doc_id:<8} {score:<15.6f}  [クラスタ{cluster_id}]")

        # 全クラスタを検索（比較用）
        print("\n【クラスタリング利用検索（全クラスタ）】")
        all_cluster_results = model.search_with_clustering(query_words, top_n_clusters=None)

        print(f"\n{'順位':<4} {'文書ID':<8} {'類似度スコア':<15}")
        print("-" * 30)
        for rank, (doc_id, score) in enumerate(all_cluster_results, 1):
            cluster_id = model.doc_clusters.get(doc_id, -1)
            print(f"{rank:<4} {doc_id:<8} {score:<15.6f}  [クラスタ{cluster_id}]")

if __name__ == '__main__':
    main()
