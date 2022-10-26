import numpy as np

lines = []
file = open("gen2.txt", "a")
for i in range(100):
    line = "night_train_all {:010d}\n".format(i)
    file.write(line)
     