from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import random

# 訓練文本
text = """
今天天氣很好，我想出去散步。
今天下雨了，我想待在家裡。
我喜歡學習人工智慧。
我喜歡寫程式。
人工智慧可以幫助我們學習。
今天我想學習語言模型。
"""

text = text.replace("\n", "").replace(" ", "")

# 建立資料集：前兩個字 -> 下一個字
X = []
y = []

for i in range(len(text) - 2):
    prev_two = text[i:i+2]
    next_char = text[i+2]

    X.append(prev_two)
    y.append(next_char)

print("訓練資料：")
for i in range(10):
    print(X[i], "=>", y[i])

# 將文字特徵轉成向量
vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 2))
X_vec = vectorizer.fit_transform(X)

# 使用 Naive Bayes 分類器
model = MultinomialNB()
model.fit(X_vec, y)


def predict_next(prev_two):
    x_vec = vectorizer.transform([prev_two])
    return model.predict(x_vec)[0]


def generate(start, length=30):
    result = start

    for _ in range(length):
        prev_two = result[-2:]
        next_char = predict_next(prev_two)
        result += next_char

    return result

while True:
    text = input("請輸入前兩個字：")

    if len(text) != 2:
        print("請輸入兩個字")
        continue

    print("預測下一個字：", predict_next(text))

# 測試預測
print("\n預測下一個字：")
print("今天 =>", predict_next("今天"))
print("我喜 =>", predict_next("我喜"))
print("人工 =>", predict_next("人工"))

# 生成句子
print("\n生成結果：")
print(generate("今天", 30))
print(generate("我喜", 30))
print(generate("人工", 30))
