from nn import Value, Adam, linear, cross_entropy
import random

# XOR 資料集
# 輸入 x1, x2，輸出類別 0 或 1
data = [
    ([0.0, 0.0], 0),
    ([0.0, 1.0], 1),
    ([1.0, 0.0], 1),
    ([1.0, 1.0], 0),
]

# 建立參數
# 2 個輸入 → 4 個隱藏神經元 → 2 個輸出類別
w1 = [[Value(random.uniform(-1, 1)) for _ in range(2)] for _ in range(4)]
b1 = [Value(0.0) for _ in range(4)]

w2 = [[Value(random.uniform(-1, 1)) for _ in range(4)] for _ in range(2)]
b2 = [Value(0.0) for _ in range(2)]

params = []
for row in w1:
    params += row
params += b1

for row in w2:
    params += row
params += b2

optimizer = Adam(params, lr=0.05)


def model(x):
    # 把輸入轉成 Value
    x = [Value(v) for v in x]

    # 第一層：linear + ReLU
    h = linear(x, w1)
    h = [h[i] + b1[i] for i in range(4)]
    h = [v.relu() for v in h]

    # 第二層：輸出 logits
    logits = linear(h, w2)
    logits = [logits[i] + b2[i] for i in range(2)]

    return logits


# 訓練
for epoch in range(1000):
    total_loss = 0

    for x, target in data:
        logits = model(x)
        loss = cross_entropy(logits, target)

        loss.backward()
        optimizer.step()

        total_loss += loss.data

    if epoch % 100 == 0:
        print(f"epoch {epoch}, loss = {total_loss:.4f}")


# 測試
print("\n測試結果：")
for x, target in data:
    logits = model(x)

    values = [v.data for v in logits]
    pred = values.index(max(values))

    print(f"輸入 {x}，預測 = {pred}，正確答案 = {target}")
