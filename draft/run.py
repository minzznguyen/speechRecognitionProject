import sys

name = sys.argv[1]
with open("hi/"+name, 'r') as f:
    for x in f:
        print(f)