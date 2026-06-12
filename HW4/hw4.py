import streamlit as st
import ollama
import pandas as pd
import os
from difflib import SequenceMatcher

# =========================
# Prompt 自動最佳化 GPT
# 檔名建議：app.py
# 執行：streamlit run app.py
# =========================

MODEL = "llama3.2"
HISTORY_FILE = "prompt_history.csv"

st.set_page_config(page_title="Prompt 自動最佳化 GPT", page_icon="🤖")
st.title("Prompt 自動最佳化 GPT")

# -------------------------
# 基本函式
# -------------------------

def ask_ollama(messages):
    """呼叫 Ollama 本地模型"""
    try:
        response = ollama.chat(
            model=MODEL,
            messages=messages
        )
        return response["message"]["content"]
    except Exception as e:
        return f"錯誤：無法呼叫 Ollama，請確認 Ollama 已啟動，且已安裝模型 {MODEL}\n\n詳細錯誤：{e}"


def similarity(a, b):
    """用文字相似度作為簡單評分"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def improve_prompt(old_prompt, question, answer, target_answer, score):
    """讓 AI 自己改寫 Prompt"""
    improve_messages = [
        {
            "role": "system",
            "content": "你是一位 Prompt Engineering 專家，請改寫 system prompt，讓 AI 回答更準確。"
        },
        {
            "role": "user",
            "content": f"""
目前 Prompt：
{old_prompt}

使用者問題：
{question}

標準答案：
{target_answer}

AI 回答：
{answer}

目前分數：
{score}

請改寫成更好的 system prompt。

要求：
1. 使用繁體中文
2. 讓回答更精準
3. 不要過度冗長
4. 只輸出改寫後的 prompt，不要解釋
"""
        }
    ]

    return ask_ollama(improve_messages)


def save_history(prompt, question, answer, target_answer, score):
    """儲存實驗紀錄"""
    data = {
        "prompt": prompt,
        "question": question,
        "answer": answer,
        "target_answer": target_answer,
        "score": score
    }

    df = pd.DataFrame([data])

    df.to_csv(
        HISTORY_FILE,
        mode="a",
        index=False,
        header=not os.path.exists(HISTORY_FILE),
        encoding="utf-8-sig"
    )


# -------------------------
# Session State
# -------------------------

if "best_prompt" not in st.session_state:
    st.session_state.best_prompt = (
        "你是一個使用繁體中文回答的 AI 助手。"
        "請用清楚、正確、簡潔的方式回答問題。"
    )

if "best_score" not in st.session_state:
    st.session_state.best_score = 0.0

if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------
# 側邊欄設定
# -------------------------

st.sidebar.header("Prompt 最佳化設定")

st.session_state.best_prompt = st.sidebar.text_area(
    "目前最佳 Prompt",
    st.session_state.best_prompt,
    height=180
)

target_answer = st.sidebar.text_input(
    "標準答案 / 期望答案",
    ""
)

opt_rounds = st.sidebar.slider(
    "自動最佳化回合數",
    min_value=1,
    max_value=5,
    value=3
)

st.sidebar.write("目前最佳分數：", round(st.session_state.best_score, 3))

if st.sidebar.button("清除對話"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("重設最佳 Prompt"):
    st.session_state.best_prompt = (
        "你是一個使用繁體中文回答的 AI 助手。"
        "請用清楚、正確、簡潔的方式回答問題。"
    )
    st.session_state.best_score = 0.0
    st.rerun()

if os.path.exists(HISTORY_FILE):
    df = pd.read_csv(HISTORY_FILE)
    st.sidebar.write("實驗次數：", len(df))

    with st.sidebar.expander("查看歷史分數"):
        st.dataframe(df[["question", "score"]].tail(10))


# -------------------------
# 顯示歷史對話
# -------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -------------------------
# 使用者輸入
# -------------------------

user_input = st.chat_input("請輸入你的問題...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    current_prompt = st.session_state.best_prompt
    best_answer = ""
    best_prompt = current_prompt
    best_score = -1.0

    with st.chat_message("assistant"):
        if not target_answer.strip():
            st.warning("尚未輸入標準答案，系統仍會回答，但分數會以 0 計算。")

        st.write("正在進行 Prompt 自動最佳化...")

        for i in range(opt_rounds):
            messages = [
                {
                    "role": "system",
                    "content": current_prompt
                },
                {
                    "role": "user",
                    "content": user_input
                }
            ]

            answer = ask_ollama(messages)

            if target_answer.strip():
                score = similarity(answer, target_answer)
            else:
                score = 0.0

            st.write(f"第 {i + 1} 回合分數：{score:.3f}")

            save_history(
                current_prompt,
                user_input,
                answer,
                target_answer,
                score
            )

            if score > best_score:
                best_score = score
                best_answer = answer
                best_prompt = current_prompt

            current_prompt = improve_prompt(
                current_prompt,
                user_input,
                answer,
                target_answer,
                score
            )

        st.session_state.best_prompt = best_prompt

        if best_score > st.session_state.best_score:
            st.session_state.best_score = best_score

        st.write("### 最佳回答")
        st.write(best_answer)

        st.write("### 最佳 Prompt")
        st.code(best_prompt)

    st.session_state.messages.append({
        "role": "assistant",
        "content": best_answer
    })
