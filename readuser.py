import json

with open("./signeduser_table.json", "r") as f:
    data = json.load(f)
    # print(data[1]['Phone_Num'], data[1]['profileimage'])
    print(data[0][1])