import os


directory = 'rec/'

files = os.listdir(directory)
x, y = [], []
for file in files:
    a, b, _ = file.split('_')
    a = int(a)
    b = float(b)
    x.append([a, b])

x.sort()
print(x)
