import json
import re
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# 定义严谨的系统提示词
QUERY_PROMPT = """你是一个 RAG 检索专家。请分析用户问题，并将其拆解为最适合向量数据库搜索的指令。
你的任务：
1. **多查询拆分**：将复杂问题拆分为 2-3 个独立的子查询，涵盖不同侧重点。
2. **中英对齐**：对于技术词汇（如 Agent, RAG, Weights），生成对应的英文查询词以提高召回率。
3. **动态阈值**：
   - 严谨/官方/学术问题：min_audit_score = 85
   - 通用/技术方案：min_audit_score = 75
   - 宽泛/闲聊：min_audit_score = 60

必须返回如下格式的 JSON，严禁输出任何其他解释性文字：
{{
    "thought": "在此简述你的思考过程，包括对问题类型的判断和拆解逻辑",
    "sub_queries": ["查询词1", "查询词2"],
    "min_audit_score": 75
}}

用户问题：{user_input}"""


def translate_user_query(user_input: str, llm) -> dict:
    # 1. 填充 Prompt
    prompt = PromptTemplate.from_template(QUERY_PROMPT)
    parser = JsonOutputParser()

    chain = prompt | llm | parser
    try:
        plan = chain.invoke({"user_input": user_input})

        # 在控制台打印出思考过程，方便调试
        if "thought" in plan:
            print(f"--- LLM CoT 推理过程 ---\n{plan['thought']}\n-----------------------")

        # 严谨性检查
        if not isinstance(plan.get("sub_queries"), list):
            plan["sub_queries"] = [user_input]

        if "min_audit_score" not in plan:
            plan["min_audit_score"] = 75

        return plan

    except Exception as e:
        print(f"[QueryTranslate Error] 链路执行失败: {e}")
        return {
            "thought": "解析失败，触发兜底",
            "sub_queries": [user_input],
            "min_audit_score": 75
        }

