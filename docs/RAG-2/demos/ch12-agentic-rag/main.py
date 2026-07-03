#!/usr/bin/env python3
"""
ch12-agentic-rag: ReAct agent with tool use for RAG.
Agent reasons, acts using tools, and observes results in a loop.
Self-contained, stdlib only.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


KB: Dict[str, str] = {
    "恒瑞医药": "恒瑞医药是一家中国创新药企业，成立于1970年，总部位于江苏连云港。主要产品包括卡瑞利珠单抗（PD-1抑制剂）、吡咯替尼（HER2靶向药）、奥希替尼片（EGFR-TKI）、注射用紫杉醇等抗肿瘤药物。",
    "卡瑞利珠单抗": "卡瑞利珠单抗（艾瑞卡）是恒瑞医药自主研发的PD-1抑制剂。已获批用于霍奇金淋巴瘤、肝癌、非小细胞肺癌等多种癌症的治疗。",
    "吡咯替尼": "吡咯替尼（艾瑞妮）是恒瑞医药开发的HER2靶向药物。用于HER2阳性乳腺癌的治疗，可延长患者无进展生存期。",
    "奥希替尼片": "奥希替尼（泰瑞沙）是第三代EGFR-TKI靶向药。用于EGFR突变阳性非小细胞肺癌的一线及二线治疗。",
    "注射用紫杉醇": "紫杉醇是一种抗微管类化疗药，恒瑞医药生产白蛋白结合型紫杉醇。用于乳腺癌、非小细胞肺癌和胰腺癌的治疗。",
}


def retrieve_documents(query: str) -> str:
    results = []
    for key, value in KB.items():
        if set(query) & set(key):
            results.append(f"[{key}]: {value}")
    if not results:
        for key, value in KB.items():
            if any(t in value for t in query if len(t) > 1):
                results.append(f"[{key}]: {value}")
    return "\n".join(results[:3]) if results else "未找到相关信息。"


def calculate(expression: str) -> str:
    safe_chars = set("0123456789+-*/.()% ")
    if not all(c in safe_chars for c in expression):
        return "错误：表达式包含不允许的字符。"
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


def search_web(query: str) -> str:
    results_map = {
        "恒瑞医药": "恒瑞医药是中国领先的创新药企业，2024年研发投入超过60亿元。",
        "泰瑞沙": "泰瑞沙（奥希替尼）是第三代EGFR-TKI，用于非小细胞肺癌治疗。",
        "乳腺癌": "乳腺癌是女性最常见的恶性肿瘤之一，靶向治疗和免疫治疗是重要治疗手段。",
    }
    for key, result in results_map.items():
        if key in query:
            return f"[搜索结果] {result}"
    return f"[搜索结果] 关于'{query}'的搜索结果: 暂未找到精确匹配。"


class ReActAgent:
    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}
        self.max_steps = 5

    def think(self, question: str, context: str) -> str:
        q_terms = set(question)
        c_terms = set(context)
        if len(q_terms & c_terms) >= len(q_terms) * 0.5 and len(context) > 50:
            return "ANSWER"
        if any(w in question for w in ["计算", "多少", "+", "-", "*", "/"]):
            return "calculate"
        if any(w in question for w in ["生产", "产品", "药品", "药物"]):
            return "retrieve_documents"
        return "search_web"

    def act(self, thought: str, question: str) -> str:
        if thought == "ANSWER":
            return ""
        tool = self.tools.get(thought)
        return tool.func(question) if tool else f"未知工具: {thought}"

    def run(self, question: str) -> str:
        print(f"\n  用户问题: {question}")
        print(f"  {'=' * 50}")
        context = ""
        for step in range(1, self.max_steps + 1):
            print(f"\n  步骤 {step}/{self.max_steps}:")
            thought = self.think(question, context)
            print(f"    思考: {thought}")
            if thought == "ANSWER":
                lines = context.split("\n")
                key_lines = [l for l in lines if any(t in l for t in question)]
                answer = "\n".join(key_lines[:2]) if key_lines else context[:200]
                print(f"    回答: {answer}")
                return answer
            observation = self.act(thought, question)
            print(f"    行动 ({thought}):")
            for line in observation.split("\n"):
                print(f"      {line}")
            context += "\n" + observation if context else observation
        print(f"\n  达到最大步骤数，基于现有信息回答:")
        answer = context[:200] if context else "无法确定答案。"
        print(f"    回答: {answer}")
        return answer


def main():
    print("=" * 60)
    print("ch12-agentic-rag: 基于 ReAct 的 Agentic RAG")
    print("=" * 60)

    tools = [
        Tool("retrieve_documents", "从知识库检索文档信息", retrieve_documents),
        Tool("calculate", "执行数学计算", calculate),
        Tool("search_web", "模拟网络搜索", search_web),
    ]
    print("\n可用工具:")
    for t in tools:
        print(f"  [{t.name}] {t.description}")

    agent = ReActAgent(tools)
    print("\n" + "=" * 60)
    print("ReAct 推理过程")
    print("=" * 60)

    agent.run("恒瑞医药生产哪些药品？")

    print("\n" + "=" * 60)
    print("ReAct Agent 工作流程总结")
    print("=" * 60)
    print("  1. 思考 (Think): 分析问题，决定下一步行动")
    print("  2. 行动 (Act):  调用工具获取信息")
    print("  3. 观察 (Observe): 整合工具返回结果")
    print("  4. 循环直到可回答或达到最大步数")
    print("=" * 60)


if __name__ == "__main__":
    main()
