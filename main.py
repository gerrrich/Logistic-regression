import numpy as np
import random
import datetime as t

time = t.datetime.now()


def survived(beta: np.ndarray, index):
    temp = -beta.dot(people[index])

    if temp >= 0:
        return False
    else:
        return True


def l(beta: np.ndarray, indexes):
    s = 0.

    for x in indexes:
        temp = -beta.dot(people[x])

        if result[x] == 1:
            if temp >= 8:
                s -= temp
            elif temp <= -8:
                s += 0
            else:
                s += np.log(1 / (1 + np.e ** temp))
        else:
            if temp >= 8:
                s += 0
            elif temp <= -8:
                s += temp
            else:
                s += np.log(1 / (1 + np.e ** temp))

    return s


def delta_l(beta: np.ndarray, indexes):
    s = np.array([0, 0, 0, 0, 0, 0])

    for x in indexes:
        temp = -beta.dot(people[x])

        if temp >= 8:
            if result[x] == 1:
                s = s + people[x]
        elif temp <= 8:
            if result[x] == 0:
                s = s - people[x]
        else:
            s = s + (result[x] - 1 / (1 + np.e ** temp)) * people[x]
    return s


with open('titanic.csv') as file:
    ti = file.readlines()

people = []
result = []

for i in ti:
    human = i[:-1].split(',')
    result.append(int(human[0]))
    human = np.array([human[1], human[2], human[3], human[4], human[5], human[6]], dtype=float)
    people.append(human)

max0 = max(people, key=lambda x: x[0])[0]
max1 = max(people, key=lambda x: x[1])[1]
max2 = max(people, key=lambda x: x[2])[2]
max3 = max(people, key=lambda x: x[3])[3]
max4 = max(people, key=lambda x: x[4])[4]
max5 = max(people, key=lambda x: x[5])[5]

min0 = min(people, key=lambda x: x[0])[0]
min1 = min(people, key=lambda x: x[1])[1]
min2 = min(people, key=lambda x: x[2])[2]
min3 = min(people, key=lambda x: x[3])[3]
min4 = min(people, key=lambda x: x[4])[4]
min5 = min(people, key=lambda x: x[5])[5]

for i in range(len(people)):
    people[i] = np.array([(people[i][0] - min0) / (max0 - min0),
                          (people[i][1] - min1) / (max1 - min1),
                          (people[i][2] - min2) / (max2 - min2),
                          (people[i][3] - min3) / (max3 - min3),
                          (people[i][4] - min4) / (max4 - min4),
                          (people[i][5] - min5) / (max5 - min5)])

train_indexes = list(range(887))
test_indexes = []

while len(test_indexes) != 887 // 20:
    r = int(random.randint(0, 886 - len(test_indexes)))
    test_indexes.append(train_indexes.pop(r))

train_indexes = train_indexes * 2

per = 0
xxx = 0
yyy = 0
p = 0
mi = 1000
left = 0
right = 2
best_beta = 0
super_beta = 0
for a1 in range(left, right):
    for a2 in range(left, right):
        for a3 in range(left, right):
            for a4 in range(left, right):
                for a5 in range(left, right):
                    yyy = 0
                    for a6 in range(left, right):
                        p += 1
                        if per < round(p * 100 / (right - left) ** 6):
                            per = round(p * 100 / (right - left) ** 6)
                            print(per, '%')

                        if yyy == 4:
                            continue
                        beta = np.array([a1 / 10, a2 / 10, a3 / 10, a4 / 10, a5 / 10, a6 / 10])
                        n = 0.0000001
                        q = 1000
                        xxx = 0
                        for i in range(400):
                            if xxx == 4:
                                continue
                            beta = beta + n * delta_l(beta, train_indexes)
                            errors = 0
                            for j in train_indexes:
                                if survived(beta, j) != result[j]:
                                    errors += 1
                            if errors < q:
                                q = errors
                                best_beta = beta
                            else:
                                xxx += 1
                        if mi > q:
                            mi = q
                            super_beta = best_beta
                        else:
                            yyy += 1

# print('Вектор θ =', super_beta)
# print('L(θ) =', l(super_beta, train_indexes))

errors = 0
for j in test_indexes:
    if survived(super_beta, j) != result[j]:
        errors += 1
print('Точность на проверочных данных:', round((44 - errors)*100 / 44),'%')

errors = 0
for j in train_indexes:
    if survived(super_beta, j) != result[j]:
        errors += 1
print('Точность на обучающих данных:', round((len(train_indexes) - errors)*100 / len(train_indexes)),'%')

time = t.datetime.now() - time
print('Время работы программы:', time.seconds+round(time.microseconds/10000)/100, 'сек')
