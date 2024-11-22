import math

c = 10000
for i in range(3, c):
    for j in range(i+1, c):
        for k in range(j+1, c):
            if (2**k + 2**j + 2**i) % 2024 == 0:
                print(i, k,j)
                break




