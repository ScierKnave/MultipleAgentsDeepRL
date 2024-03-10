import numpy as np 

train = [
1499,
5254,
851,
662,
1876,
6552,
]
train_sum = sum(train)
prob_train = np.array(train) / train_sum
train_pred = np.random.choice([0, 1, 2, 3, 4, 5], 1000000, p=prob_train)
train_true = np.random.choice([0, 1, 2, 3, 4, 5], 1000000, p=prob_train)
train_exp = np.equal(train_pred, train_true)

train_exp = []
for i in range(20000):
    train_true = np.random.choice([0, 1, 2, 3, 4, 5], 1000, p=prob_train)
    train_pred = np.random.choice([0, 1, 2, 3, 4, 5], 1000, p=prob_train)
    train_exp.append(np.mean(np.equal(train_pred, train_true)))
train_exp = np.array(train_exp)

val = [
362,
1314,
198,
174,
466,
1660,
]
val_sum = sum(val)
prob_val = np.array(val) / val_sum
val_true = np.random.choice([0, 1, 2, 3, 4, 5], 1000000, p=prob_val)
val_pred = np.random.choice([0, 1, 2, 3, 4, 5], 1000000, p=prob_train)
val_exp = np.equal(val_pred, val_true)

val_exp = []
for i in range(20000):
    val_true = np.random.choice([0, 1, 2, 3, 4, 5], 1000, p=prob_val)
    val_pred = np.random.choice([0, 1, 2, 3, 4, 5], 1000, p=prob_train)
    val_exp.append(np.mean(np.equal(val_pred, val_true)))
val_exp = np.array(val_exp)


test = [
649,
1988,
452,
370,
726,
2399,
]
test_sum = sum(test)
prob_test = np.array(test) / test_sum

test_exp = []
for i in range(20000):
    test_true = np.random.choice([0, 1, 2, 3, 4, 5], 1000, p=prob_test)
    test_pred = np.random.choice([0, 1, 2, 3, 4, 5], 1000, p=prob_train)
    test_exp.append(np.mean(np.equal(test_pred, test_true)))
test_exp = np.array(test_exp)

print("train", np.mean(train_exp), np.std(train_exp))
print("val", np.mean(val_exp), np.std(val_exp))
print("test", np.mean(test_exp), np.std(test_exp))