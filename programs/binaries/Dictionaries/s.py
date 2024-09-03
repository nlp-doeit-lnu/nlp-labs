import random

dict1 = open("dict1.txt", "r").read().split("\n")

max = len(dict1)
dict2 = []
dict3 = []
for i in range(max):
    g = random.uniform(0, 1)
    
    if g <= 0.2:
        dict2.append(dict1[i])
    elif g > 0.2:
        dict3.append(dict1[i])
   
f1 = open("dict2.txt", "w")
f2 = open("dict3.txt", "w")

for i in dict2:
    f1.write(i + "\n")
for i in dict3:
    f2.write(i + "\n")

f1.close()
f2.close()