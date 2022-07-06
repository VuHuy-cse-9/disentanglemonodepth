#folders = ["2011_09_26","2011_09_28", "2011_09_29"]
#folders = ["2011_09_28", "2011_09_29"]
#folders = ["2011_09_26"]
folders = ["2011_09_28"]

#Val
lines = None
with open('eigen_zhou_val.txt') as f:
    lines = f.readlines()
    

folder_lines = []
for line in lines:
    for folder in folders:
        if folder in line:
            folder_lines.append(line)
        
with open('val_custom.txt', 'a') as f:
    for line in folder_lines:
        f.write(line)
        
#Train
lines = None
with open('eigen_zhou_train.txt') as f:
    lines = f.readlines()
    

folder_lines = []
for line in lines:
    for folder in folders:
        if folder in line:
            folder_lines.append(line)
        
with open('train_custom.txt', 'a') as f:
    for line in folder_lines:
        f.write(line)
        
#Test
lines = None
with open('eigen_zhou_test.txt') as f:
    lines = f.readlines()
    

folder_lines = []
for line in lines:
    for folder in folders:
        if folder in line:
            folder_lines.append(line)
        
with open('test_custom.txt', 'a') as f:
    for line in folder_lines:
        f.write(line)
        
