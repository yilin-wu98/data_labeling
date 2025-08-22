## read files in ~/Downlooads/ folder that starts with "code"
import os
import json
code_files = [f for f in os.listdir("/home/yilinw/Downloads/") if f.startswith("description")]

print(code_files)

## read each files and their format is like this : {"0": "description_0.txt", "1": "description_1.txt", "2": "description_2.txt", "3": "description_3.txt", "4": "description_4.txt", "5": "description_5.txt", "6": "description_6.txt", "7": "description_7.txt", "8": "description_8.txt", "9": "description_9.txt"}

for file in code_files:
    with open(os.path.join("/home/yilinw/Downloads/", file), "r") as f:
        data = json.load(f)
    print(data)
## at the 

