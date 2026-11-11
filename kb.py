"""知识库检索：关键词检索（BM25）+ 向量检索（Embedding）多路召回，RRF 融合。

两路各管一块：
- BM25 认字面。版本号、报错码、参数名（413 / v4.17.3 / chunkSettingMode）只能靠它。
- 向量认语义。用户口语换说法（"把话说完整" vs "指代消解"）字面对不上，靠它补。

为什么是"混合"而不是"替换"：两路的盲区不重叠，一个漏的另一个补。
合并用 RRF，只用排名不用分数——BM25 分无界（0~200）、余弦在 -1~1，直接相加没意义。
"""

import json
import math
import pathlib

import jieba

DATA_PATH = pathlib.Path(__file__).parent / "data" / "faq.json"
VEC_PATH = pathlib.Path(__file__).parent / "data" / "faq_vectors.json"

# embedding 模型：本地路径（D 盘，避免占 C 盘）。换模型改这里 + 删掉旧的 faq_vectors.json 重算。
MODEL_PATH = pathlib.Path("D:/hf-cache/models/bge-m3")

# bge 系列检索时 query 要加这个前缀、文档不加。不加前缀召回会明显掉，这是模型官方要求。
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

# 读一次就够，415 条数据在内存里毫无压力，不值得为它引入缓存逻辑
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


def _get_model():
    """模块级单例。模型加载要几秒，绝不能在每次检索时重载。"""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(str(MODEL_PATH))
    return _MODEL


_MODEL = None


def _embed(texts):
    """把一批文本（文档侧）压成向量。文档不加前缀，只在建缓存时调用一次。"""
    return _get_model().encode(texts, normalize_embeddings=True).tolist()


def _embed_query(text):
    """把一条 query 压成向量。query 加前缀，这是 bge 模型的检索要求。"""
    return _get_model().encode([QUERY_PREFIX + text], normalize_embeddings=True)[0].tolist()


def _load_vectors():
    """读向量缓存；没有就现算并存盘。缓存的理由和 faq.json 一样：可复现，不必每次算。

    放到首次使用时才加载（惰性），不在 import 时算：这样没装 sentence-transformers
    的环境里 kb.py 照样能 import、BM25 照常工作，向量只影响 search 这一层。
    """
    global VECTORS
    if VECTORS is not None:
        return VECTORS
    if VEC_PATH.exists():
        cached = json.loads(VEC_PATH.read_text(encoding="utf-8"))
        # 维度跟当前模型对得上才用；换模型后旧缓存维度不同，直接重算，否则点积会崩
        if cached and len(cached[0]) == len(_embed_query(CHUNKS[0]["title"])):
            VECTORS = cached
            return VECTORS
    texts = [c["title"] + "\n" + c["text"] for c in CHUNKS]
    VECTORS = _embed(texts)
    VEC_PATH.write_text(json.dumps(VECTORS, ensure_ascii=False), encoding="utf-8")
    return VECTORS


VECTORS = None


def _cos(query_vec, doc_vec):
    """余弦相似度。向量已归一化，退化成点积。"""
    return sum(a * b for a, b in zip(query_vec, doc_vec))


def bm25_rank(query, n=10):
    """关键词检索一路，返回前 n 个文档下标。"""
    terms = tokenize(query)
    scored = [(bm25_score(terms, i), i) for i in range(len(CHUNKS))]
    scored = [(s, i) for s, i in scored if s > 0]
    scored.sort(key=lambda pair: -pair[0])
    return [i for _, i in scored[:n]]


def vector_rank(query, n=10):
    """向量检索一路，返回前 n 个文档下标。

    没装 sentence-transformers 时返回空列表，RRF 自然只走 BM25——这是优雅降级，
    不是报错。判断要不要上向量，看的就是这一路加进来之后召回涨多少。
    """
    #向量化都是直接调用向量计算模型的api的
    #向量化查询（提示词）
    try:
        q = _embed_query(query)
    except Exception:
        return []
    #向量化文档（文档侧）
    vecs = _load_vectors()
    #计算余弦相似度（算法是自己写的）
    scored = [(_cos(q, v), i) for i, v in enumerate(vecs)]
    #排序返回
    scored.sort(key=lambda pair: -pair[0])
    return [i for _, i in scored[:n]]


def search(query, top_k=5):
    """多路召回 + RRF 融合，返回最相关的几条知识。

    分数含义从 BM25（0~200 无界）变成 RRF（0~0.033），量纲不同，
    所以"相似度 12.4"这类比较不再成立——前端显示为"相关度"。
    """
    query = (query or "").strip()
    if not query:
        return []

    #利用RRF算法融合BM25和向量检索结果（合并多路召回结果）
    #RRF = 对每一路检索结果，把每篇文档的「排名」换算成 1/(60+rank)，同一篇文档在多路里出现就累加，最后按累加总分降序取前 top_k。
    fused = {}
    for ranker in (bm25_rank, vector_rank):  # 第 1 层：换一路（BM25 还是向量）
        for rank, idx in enumerate(ranker(query), 1):  # 第 2 层：遍历这一路返回的每一个文档 enumerate 的作用：一边遍历列表，一边给你「每个元素是第几个」这个位置号。
            fused[idx] = fused.get(idx, 0) + 1 / (60 + rank) #fused.get(idx, 0)是取旧值，1 / (60 + rank)算新值

    top = sorted(fused.items(), key=lambda pair: -pair[1])[:top_k]

    hits = []
    for idx, score in top:
        chunk = CHUNKS[idx]
        hits.append({
            "id": chunk["id"],
            "title": chunk["title"],
            "section": chunk["section"],
            "text": chunk["text"],
            "source": chunk["source"],
            "score": round(score, 4),
        })
    return hits


if __name__ == "__main__":
    # 直接跑这个文件可以手动试检索效果，不用起服务
    for question in ["上传文件报 413 怎么办", "应用发布了但聊天没更新", "怎么接入自己的模型"]:
        print(f"\n=== {question} ===")
        for hit in search(question):
            print(f"  {hit['score']:>6.2f}  {hit['id']}")
