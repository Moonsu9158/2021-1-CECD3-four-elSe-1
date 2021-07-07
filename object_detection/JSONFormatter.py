import json
import uuid

def jsonFormatter(box,label,img_name):
    b = box.astype(int)
    detected = {
        "objectID": str(uuid.uuid4()),
        "location":{
            "xmin":b[1],
            "ymin":b[3],
            "xmax":b[0],
            "ymax":b[2]
        },
        "tag": label,
        "IMG_URL" : img_name
    }

    filename = output_path + "{}_path: ({}).jpg".format(labels_to_names_seq[label]+str(labels_to_num[label]),imagePath_str)
    with open(filename,"w") as outFile:
        json.dump(detected, outFile, indent=4)

    print("hello")

box = [1,2,3,4]
label = "testLabel"
img_name = "testImg"

jsonFormatter(box,label,img_name)