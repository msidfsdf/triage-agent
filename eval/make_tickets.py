"""从官方文档反推咨询型工单，生成新的 tickets.json（评测基准）。

为什么这么干：GitHub issue 天然是 bug 为主（87% 缺陷报告），官方文档全是「怎么用」。
拿 bug 工单去测「怎么用」的知识库，自动解决率永远只有 11%，这杆秤测不出功能好坏。

所以这里反过来：以官方文档为源，反推「用户会怎么问」，造一批分布合理的工单：
- 咨询类（可答）：约 60 条，每条绑定一条文档（answer_id），可核查
- 信息不足（反问）：约 20 条，只给半句话，模型该走 ask_more
- 缺陷/需求（转人工）：约 20 条，问文档里永远没有的东西，模型该走 escalate

这不是「捏造成绩」：每类工单该走哪条路是构造时就定好的，评测量的是
「模型有没有按依据选对路」，而不是「能不能答」。答案有 answer_id 可追溯。

用法：python eval/make_tickets.py
"""

import json
import pathlib
import random
import sys

from dotenv import load_dotenv

BASE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))
sys.stdout.reconfigure(encoding="utf-8")
load_dotenv(BASE.parent / ".env")

import triage

FAQ = BASE.parent / "data" / "faq.json"
OUT = BASE.parent / "data" / "tickets.json"

# 能反推成「用户会问怎么配/能不能/是什么」的 section
CONSULT_SECTIONS = {
    'chat', 'voiceInput', 'mcp_server', 'mcp_tools', 'siliconCloud', 'feishu',
    'dingtalk', 'wecom', 'collection_tags', 'collection_permission',
    'collection_tags_legacy', 'dataset', 'third_dataset', 'api_dataset',
    'dataset_engine', 'dataset_search', 'fileInput', 'htmlRendering', 'link',
    'customDomain', 'custom_feedback', 'question_classify', 'welcomeText',
    'chat_input_guide', 'sso', 'teamMode', 'team_roles_permissions',
    'team-installation', 'invitation_link', 'app-integration', 'quick-start',
    'model-presets', 'ollama', 'xinference', 'minimax', 'tokenDance', 'oxr',
    'object-storage', 'docker_minio_to_rustfs', 'sandbox-v2', 'opensandbox',
    'sealosdevbox', 'loop', 'loop_run', 'parallel_run', 'tfswitch',
    'system-tool-development', 'content_extract', 'variable_update',
    'coreferenceResolution', 'form_input', 'text_editor', 'rules', 'debug',
    'evaluation', 'methods', 'assisted-generation',
    'knowledge_base_search_merge', 'tool', 'upload_system_tool',
}

N_CONSULT = 180
N_ASK_MORE = 60
N_ESCALATE = 60

GEN_PROMPT = """你是客服工单的撰写者。给你一条产品文档，请你写一句「用户向客服提问」时最可能说的话。

要求：
1. 用真实用户的口吻，口语化，第一人称。
2. 不要照抄文档里的专有名词、产品功能名、参数名、按钮名、步骤标题。
   用户提问时通常只知道「他想做什么」，不知道系统里的功能叫什么。
3. 一句到两句话，20~80 字。
4. 只输出这句话本身，不要引号、不要解释。

文档内容：
{text}"""


def gen_consult(item):
    """基于一条文档，反推一条咨询工单。"""
    resp = triage.CLIENT.chat.completions.create(
        model=triage.MODEL,
        messages=[
            {"role": "system", "content": GEN_PROMPT.format(text=item["text"][:800])},
        ],
        temperature=0.8,
        extra_body={"thinking": {"type": "disabled"}},
    )
    text = (resp.choices[0].message.content or "").strip().strip('"').strip("“”")
    return {
        "number": item["id"],
        "title": text,
        "body": "",
        "labels": ["咨询"],
        "answer_id": item["id"],
    }


# 信息不足：只给半句话，模型该反问
ASK_MORE_SEEDS = [
    ("报错了", ""),
    ("上传失败", ""),
    ("一直转圈", ""),
    ("连不上", ""),
    ("模型没反应", ""),
    ("配置不对", ""),
    ("图片发不出去", ""),
    ("卡住了", ""),
    ("显示异常", ""),
    ("怎么弄都不行", ""),
    ("文件传不上去", ""),
    ("接口调不通", ""),
    ("应用打不开", ""),
    ("数据不对", ""),
    ("没生效", ""),
    ("崩溃了", ""),
    ("登录不了", ""),
    ("语音不行", ""),
    ("权限报错", ""),
    ("重试好几次了", ""),
    ("打不开了", ""),
    ("传不上去了", ""),
    ("识别不出来", ""),
    ("没反应了", ""),
    ("加载不出来", ""),
    ("又报错了", ""),
    ("用不了了", ""),
    ("连不上了", ""),
    ("搜不到", ""),
    ("保存不了", ""),
    ("导不出来", ""),
    ("删不掉", ""),
    ("改不了", ""),
    ("加不进去", ""),
    ("同步不了", ""),
    ("收不到消息", ""),
    ("发不出去", ""),
    ("调用失败", ""),
    ("连接超时", ""),
    ("服务挂了", ""),
    ("打不开这个功能", ""),
    ("这个按钮没反应", ""),
    ("一直加载中", ""),
    ("空白页", ""),
    ("乱码了", ""),
    ("数值不对", ""),
    ("权限不够", ""),
    ("没有权限", ""),
    ("版本不对", ""),
    ("环境有问题", ""),
    ("部署有问题", ""),
    ("数据丢失了", ""),
    ("记录不见了", ""),
    ("配置丢了", ""),
    ("token 失效", ""),
    ("接口 404", ""),
    ("接口 500", ""),
    ("模型报错", ""),
    ("解析失败", ""),
    ("上传超时", ""),
]

# 缺陷/需求：文档里永远没有答案，模型该转人工
ESCALATE_SEEDS = [
    ("分享出去的页面在 iOS 上打开白屏，安卓正常，浏览器控制台没有报错", "bug"),
    ("把知识库数据集合里的字段批量导出成 Excel，但导出来乱码", "bug"),
    ("希望支持把对话记录导出成 PDF 带水印", "feature"),
    ("两个工作流嵌套后子流程里的变量传不到父流程", "bug"),
    ("开 20 个并发请求时服务内存涨到 8G 然后 OOM 重启", "bug"),
    ("希望模型支持输出带图表的富文本，而不仅是 markdown", "feature"),
    ("调 /api/v1/chat 接口偶尔返回 504，重试又正常，无规律", "bug"),
    ("希望支持按时间段自动清理历史对话", "feature"),
    ("MCP 服务连接后工具列表偶尔加载不全，刷新才出来", "bug"),
    ("想给机器人加一个自定义的音色，不支持我们公司买的 TTS", "feature"),
    ("升级到最新版后旧的数据集全部打不开，一直 loading", "bug"),
    ("希望支持多人同时编辑同一个工作流并合并冲突", "feature"),
    ("上传 200MB 的视频文件解析到一半就中断，没有任何提示", "bug"),
    ("希望对话里能直接插入可交互的投票组件", "feature"),
    ("知识库索引进度一直卡在 99%，重启也没用", "bug"),
    ("希望支持定时任务在指定时间自动发消息", "feature"),
    ("钉钉渠道的消息偶尔重复推送两次", "bug"),
    ("希望把工作流导出成图片分享给同事", "feature"),
    ("调用工具节点时 dataId 和返回的不一致，查不到记录", "bug"),
    ("希望支持按用户画像推荐不同开场白", "feature"),
    ("工作流执行到第三步就卡死，控制台没有任何日志输出", "bug"),
    ("希望支持导出对话的完整 token 用量统计报表", "feature"),
    ("知识库删除后索引还残留，导致新传的同名文件搜不到", "bug"),
    ("希望支持子账号之间的对话数据隔离", "feature"),
    ("模型回复偶尔会夹带上一轮对话的内容，串上下文了", "bug"),
    ("希望支持自定义模型超时时间", "feature"),
    ("批量导入 CSV 时中文表头变成乱码", "bug"),
    ("希望支持接入 Salesforce 客户系统，自动同步客户标签到知识库", "feature"),
    ("iOS 端语音输入录到一半自动停止", "bug"),
    ("希望支持接入我们公司内部的 OA 审批系统，把工单审批结果回传", "feature"),
    ("向量检索偶发返回空结果，重试又有", "bug"),
    ("希望支持对接用友财务系统，导出对账单", "feature"),
    ("配置了代理后模型请求全部超时", "bug"),
    ("希望支持自动生成工单摘要", "feature"),
    ("文档解析 500 页 PDF 时内存溢出", "bug"),
    ("希望支持企业微信机器人多客服接待", "feature"),
    ("图片识别对模糊图片直接报错而不是降级", "bug"),
    ("希望支持对话内容敏感词过滤", "feature"),
    ("工作流复制后变量引用全断了", "bug"),
    ("希望支持按语言自动切换回复", "feature"),
    ("知识库权限修改后不立即生效，要等半小时", "bug"),
    ("希望支持 API 调用限流", "feature"),
    ("分享链接被搜索引擎抓到了，隐私泄露", "bug"),
    ("希望支持团队成员操作审计日志", "feature"),
    ("模型偶发返回 JSON 解析失败的错误", "bug"),
    ("希望支持自定义知识库分块规则", "feature"),
    ("多轮对话超过 20 轮后模型开始遗忘前面内容", "bug"),
    ("希望支持语音回复转文字", "feature"),
    ("上传音频文件后解析一直是 0%", "bug"),
    ("希望支持第三方模型自动负载均衡", "feature"),
    ("定时清理任务把正在使用的对话也删了", "bug"),
    ("希望支持工作流里调用外部 HTTP 接口", "feature"),
    ("知识库切换向量库后旧数据全部检索不到", "bug"),
    ("希望支持对话导出成 Markdown", "feature"),
    ("并发修改同一个应用配置会互相覆盖", "bug"),
    ("希望支持模型回答带引用来源", "feature"),
    ("公众号渠道收不到用户发来的图片", "bug"),
    ("希望支持自定义客服欢迎语模板", "feature"),
    ("升级后 API Key 全部失效", "bug"),
    ("希望支持接入金蝶 ERP 系统的物料数据", "feature"),
]


def main():
    faq = json.loads(FAQ.read_text(encoding="utf-8"))
    cands = [x for x in faq if x["section"] in CONSULT_SECTIONS and len(x["text"]) >= 40]
    random.Random(20261108).shuffle(cands)

    tickets = []
    num = 0

    print(f"1) 生成 {N_CONSULT} 条咨询工单（可答）...")
    for i, item in enumerate(cands[:N_CONSULT], 1):
        t = gen_consult(item)
        num += 1
        t["number"] = num
        tickets.append(t)
        print(f"   [{i}/{N_CONSULT}] #{num} {t['title'][:50]}")

    print(f"\n2) 生成 {N_ASK_MORE} 条信息不足工单（反问）...")
    for title, _ in ASK_MORE_SEEDS:
        num += 1
        tickets.append({"number": num, "title": title, "body": "",
                        "labels": ["信息不足"], "answer_id": None})

    print(f"\n3) 生成 {N_ESCALATE} 条缺陷/需求工单（转人工）...")
    for title, label in ESCALATE_SEEDS:
        num += 1
        tickets.append({"number": num, "title": title, "body": "",
                        "labels": [label], "answer_id": None})

    OUT.write_text(json.dumps(tickets, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUT}，共 {len(tickets)} 条")


if __name__ == "__main__":
    main()
