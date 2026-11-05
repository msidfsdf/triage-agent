"""知识库检索：jieba 分词 + BM25 打分。

为什么先用 BM25，不用向量检索：
向量检索要 embedding 接口，而 DeepSeek 没有 embedding API，得多申请一家服务商。
230 条知识用关键词检索已经够用，而且它是个更干净的基线——以后换向量检索时，
"提升了多少"这个数字是能直接对比出来的，而不是凭感觉说变好了。
"""

import json
import math
import pathlib

import jieba

DATA_PATH = pathlib.Path(__file__).parent / "data" / "faq.json"

# 读一次就够，230 条数据在内存里毫无压力，不值得为它引入缓存逻辑
CHUNKS = json.loads(DATA_PATH.read_text(encoding="utf-8"))


def tokenize(text):
    """中文分词。

    过滤掉单字：单个汉字（的、了、是）几乎不携带信息，留在索引里只会让所有文档
    都命中同一个词，把真正有区分度的词淹掉。
    """
    words = jieba.lcut(text.lower())
    return [w for w in words if len(w) > 1 and w.strip()]


def build_index():
    """建一份倒排统计：每个词出现在多少篇文档里，用来算 IDF。

    标题重复算三遍：标题是这一条知识的主题，比正文里的词更能代表它讲什么。
    """
    doc_terms = []
    for chunk in CHUNKS:
        terms = tokenize(chunk["title"] * 3 + " " + chunk["text"])
        doc_terms.append(terms)

    doc_freq = {}
    for terms in doc_terms:
        for word in set(terms):
            doc_freq[word] = doc_freq.get(word, 0) + 1

    doc_len = [len(terms) for terms in doc_terms]
    return {
        "doc_terms": doc_terms,
        "doc_freq": doc_freq,
        "doc_len": doc_len,
        "avg_len": sum(doc_len) / len(doc_len) if doc_len else 0,
        "total_docs": len(CHUNKS),
    }


INDEX = build_index()

# BM25 的两个经验参数，1.5/0.75 是通用默认值，这里没有调过，也不该在没评测的情况下调
K1 = 1.5
B = 0.75


def bm25_score(query_terms, doc_index):
    """算一条查询对一篇文档的 BM25 分数。

    IDF 用 log((N - df + 0.5) / (df + 0.5)) 这个标准形式，出现得越少的词权重越高——
    这正是"FastGPT 报错 413"里 "413" 比 "怎么" 更重要的原因。
    """
    terms = INDEX["doc_terms"][doc_index]
    length = INDEX["doc_len"][doc_index]
    avg_len = INDEX["avg_len"] or 1

    score = 0.0
    for word in query_terms:
        tf = terms.count(word)
        if tf == 0:
            continue
        df = INDEX["doc_freq"].get(word, 0)
        idf = math.log((INDEX["total_docs"] - df + 0.5) / (df + 0.5) + 1)
        # 长度归一化：长文档天然更容易命中词，要压一下，否则长片段永远排在前面
        norm = tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / avg_len))
        score += idf * norm
    return score


def search(query, top_k=3):
    """返回最相关的几条知识，附带分数。

    分数要一起返回：前端能显示"这条是 12.4 分、那条是 3.1 分"，
    一眼就能看出"匹配得很勉强"，这是后面做转人工阈值的依据。
    """
    query = (query or "").strip()
    if not query:
        return []

    query_terms = tokenize(query)
    scored = []
    for i, chunk in enumerate(CHUNKS):
        score = bm25_score(query_terms, i)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda pair: -pair[0])

    hits = []
    for score, chunk in scored[:top_k]:
        hits.append({
            "id": chunk["id"],
            "title": chunk["title"],
            "section": chunk["section"],
            "text": chunk["text"],
            "source": chunk["source"],
            "score": round(score, 2),
        })
    return hits


if __name__ == "__main__":
    # 直接跑这个文件可以手动试检索效果，不用起服务
    for question in ["上传文件报 413 怎么办", "应用发布了但聊天没更新", "怎么接入自己的模型"]:
        print(f"\n=== {question} ===")
        for hit in search(question):
            print(f"  {hit['score']:>6.2f}  {hit['id']}")
