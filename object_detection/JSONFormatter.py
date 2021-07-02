import json
import uuid

def jsonFormatter(b,label,img_name,croppedImg,filename):

    detected = {
        "objectID": str(uuid.uuid4()),
        "location":{
            "xmin":b[1].item(),
            "ymin":b[3].item(),
            "xmax":b[0].item(),
            "ymax":b[2].item()
        },
        "tag": str(label),
        "croppedImage":croppedImg,
        "IMG_URL" : img_name
    }
    
    

    with open(filename,"w") as outFile:
        json.dump(detected, outFile, indent=4)


