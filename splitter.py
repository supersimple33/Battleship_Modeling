import json

arr = []
print("opening")
with open("dataS2.json", "r") as f:
    print("loading")
    arr = json.load(f)
    print("Data Loaded. Rows: " + str(len(arr)))
    f.close()
    print("Main Closed")
L = len(arr)//2
B = arr[:L]
C = arr[L:]
print("split, " + str(len(B)) + " " + str(len(C)))

with open("dataS21.json", "w") as f:
    json.dump(B, f, ensure_ascii=False, indent=4)
    print("Done B")
    f.close()
    print("B Closed")

with open("dataS22.json", "w") as f:
    json.dump(C, f, ensure_ascii=False, indent=4)
    print("Done C")
    f.close()
    print("C Closed")