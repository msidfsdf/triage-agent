"""数据摄入：把 FastGPT 的官方文档切成知识库，把历史 issue 收成测试工单。

为什么抓仓库里的 mdx 而不是文档网站：
文档站是 Next.js 客户端渲染，抓下来还得解析 HTML；仓库里的 mdx 是原文，切分干净，
而且能直接对应到 GitHub 上的行号，以后引用来源时可核实。
"""

import json
import re
import pathlib

import httpx

REPO = "labring/FastGPT"
BRANCH = "main"
API = "https://api.github.com"
RAW = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"

# 整个 content 目录都要，只排除两类：
# - openapi：二次开发的接口文档，客服工单里不会问
# - upgrading：版本更新日志。实测它是检索质量的头号杀手——这种文档列了几十项改动，
#   几乎能命中任何关键词（"配置""报错""应用"都有），靠一堆泛词刷到最高分，
#   把真正的答案挤下去。日志不是答案，不该进知识库。
DOC_DIRS = ["document/content"]
EXCLUDE_DIRS = ["document/content/openapi", "document/content/self-host/upgrading"]

DATA_DIR = pathlib.Path(__file__).parent / "data"


def list_doc_files(client):
    """一次拿到整个仓库的文件树，比逐层列目录省配额（1 次请求 vs 几十次）。"""
    resp = client.get(f"{API}/repos/{REPO}/git/trees/{BRANCH}?recursive=1")
    if resp.status_code != 200:
        raise RuntimeError(f"取文件树失败: {resp.status_code} {resp.text[:200]}")

    paths = []
    for item in resp.json()["tree"]:
        path = item["path"]
        if item["type"] != "blob":
            continue
        if not path.endswith(".mdx") or path.endswith(".en.mdx"):
            continue
        if not any(path.startswith(d) for d in DOC_DIRS):
            continue
        if any(path.startswith(d) for d in EXCLUDE_DIRS):
            continue
        paths.append(path)
    return sorted(paths)


def fetch_raw(client, path):
    """raw 域名不占 API 配额，所以正文都从这里取。

    实测这个域名偶尔会直接断开连接（RemoteProtocolError），所以失败重试一次，
    还失败就跳过这一篇——少一篇文档不影响知识库，但整批任务不能因此中断。
    """
    for attempt in range(2):
        try:
            resp = client.get(RAW + path)
            if resp.status_code == 200:
                return resp.text
        except httpx.HTTPError:
            if attempt == 1:
                print(f"   [跳过] {path} 连接失败")
    return ""


def clean_mdx(text):
    """去掉 mdx 的 import/export、JSX 标签、图片和代码块标记，只留可读正文。"""
    text = re.sub(r"^import .*$", "", text, flags=re.M)
    text = re.sub(r"^export .*$", "", text, flags=re.M)
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def split_chunks(raw_text, source_path):
    """按二级标题切块。

    为什么按 ## 切：文档里每个小节就是一个独立主题，正好对应"一条知识"。
    一条 FAQ 一条 chunk，检索时返回的就是能直接引用的片段，不用再二次截断。
    """
    text = clean_mdx(raw_text)

    # 第一个二级标题之前的内容通常只是标题和一句话简介，跳过
    parts = re.split(r"^##\s+", text, flags=re.M)
    file_name = source_path.split("/")[-1].replace(".mdx", "")

    chunks = []
    for part in parts[1:]:
        lines = part.strip().split("\n")
        if not lines:
            continue
        heading = lines[0].strip().lstrip("#").strip()
        body = "\n".join(lines[1:]).strip()
        body = re.sub(r"\n{2,}", "\n", body)

        # 太短的片段多半是占位或标题下的空行，留着只会污染检索结果
        if len(body) < 80:
            continue

        chunks.append({
            "id": f"{file_name}#{heading}",
            "title": heading,
            "section": file_name,
            "text": body[:1200],
            "source": f"https://github.com/{REPO}/blob/{BRANCH}/{source_path}",
        })
    return chunks


def fetch_tickets(client, count=100):
    """用 search API 抓已关闭的 issue。

    为什么用 search 而不是列表接口：列表接口会把 PR 混进来，FastGPT 的关闭记录里
    九成是 PR，翻页要几十次才凑够 100 条真 issue。search 加 is:issue 一次就干净了。
    """
    resp = client.get(
        f"{API}/search/issues",
        params={
            "q": f"repo:{REPO} is:issue is:closed",
            "per_page": count,
            "sort": "updated",
            "order": "desc",
        },
    )
    if resp.status_code != 200:
        raise RuntimeError(f"取工单失败: {resp.status_code} {resp.text[:200]}")

    tickets = []
    for item in resp.json()["items"]:
        tickets.append({
            "number": item["number"],
            "title": item["title"],
            "body": (item.get("body") or "")[:2000],
            "labels": [label["name"] for label in item.get("labels", [])],
            "state_reason": item.get("state_reason"),
            "comments": item.get("comments", 0),
            "created_at": item.get("created_at"),
        })
    return tickets


def main():
    DATA_DIR.mkdir(exist_ok=True)
    with httpx.Client(timeout=30, follow_redirects=True,
                      headers={"User-Agent": "triage-agent", "Accept": "application/vnd.github+json"}) as client:
        print("1) 取文档文件列表...")
        files = list_doc_files(client)
        print(f"   共 {len(files)} 个 mdx 文件")

        print("2) 抓正文并切块...")
        chunks = []
        for path in files:
            raw = fetch_raw(client, path)
            if not raw:
                continue
            chunks.extend(split_chunks(raw, path))
        print(f"   切出 {len(chunks)} 条知识")

        print("3) 抓历史工单...")
        tickets = fetch_tickets(client)
        print(f"   共 {len(tickets)} 条")

    (DATA_DIR / "faq.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    (DATA_DIR / "tickets.json").write_text(
        json.dumps(tickets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {DATA_DIR}")


if __name__ == "__main__":
    main()
