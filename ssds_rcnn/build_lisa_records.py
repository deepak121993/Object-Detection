#import packages

from config import lisa_config as config
from pyImageSearch.utils.tfAnnotation import TFAnnotation
from sklearn.model_selection import train_test_split
from PIL import Image
import tensorflow as tf
import os
import cv2



def main(_):
    #open the config file :
    f = open(config.CLASSES_FILE,"w")

    #loop over classes
    for k,v in config.CLASSES.items():
        #construct the class info and write file:
        item=("item {\n"
            "\tid: "+str(v)+" \n"
            "\tname:" + k + "'\n"
                "}\n")
        f.write(item)

    f.close()
    D={}
    rows = open(config.ANNOT_PATH).read().strip().split("\n")
    for row in rows[1:]:
        #print("row ",row)
        row = row.split(",")[0].split(";")
        (imagePath,label,startX,startY,endX,endY,_)=row
        (startX, startY) = (float(startX), float(startY))
        (endX, endY) = (float(endX), float(endY))

        #if label is not for our use 
        if label not in config.CLASSES:
            continue
        
        p= os.path.sep.join([config.BASE_PATH,imagePath])
        b = D.get(p,[])
        b.append((label,(startX,startY,endX,endY)))
        D[p] = b

    print("length ",len(list(D)))
    (trainKeys,testKeys) = train_test_split(list(D.keys()),test_size=config.TEST_SIZE, random_state=42)
    # initialize the data split files
    datasets = [
        ("train", trainKeys, config.TRAIN_RECORD),
        ("test", testKeys, config.TEST_RECORD)
    ]

    for (dtype,keys,outputPath) in datasets:
        
        writer = tf.python_io.TFRecordWriter(outputPath)
        total=0

        #load over the curent keys
        for k in keys:
            try:
                try:
                    encoded = tf.gfile.GFile(k,"rb").read()
                    encoded= bytes(encoded)
                except:
                    continue

                #load the  image again disk
                pilImage = Image.open(k)
                (w,h) = pilImage.size[:2] 

                # parse the filename and encoding from the input path
                filename = k.split(os.path.sep)[-1]
                encoding = filename[filename.rfind(".") + 1:]

                # initialize the annotation object used to store
                # information regarding the bounding box + labels
                tfAnnot = TFAnnotation()
                tfAnnot.image = encoded
                tfAnnot.encoding = encoding
                tfAnnot.filename = filename
                tfAnnot.width = w
                tfAnnot.height = h


                # loop over the bounding boxes + labels associated with
                # the image
                for (label, (startX, startY, endX, endY)) in D[k]:
                # TensorFlow assumes all bounding boxes are in the
                # range [0, 1] so we need to scale them

                    
                    xMin = startX / w
                    xMax = endX / w
                    yMin = startY / h
                    yMax = endY / h

                    ##bounding box image 
                    image = cv2.imread(k)
                    startX = int(xMin*w)
                    startY = int(yMin*h)
                    endX = int(xMax*w)
                    endY = int(yMax*h)

                    ##draw a rectangle 
                    cv2.rectangle(image,(startX,startY),(endX,endY),(0,255,0),2)
                    cv2.imshow("image ",image)
                    cv2.waitKey(0)


                    # update the bounding boxes + labels lists
                    tfAnnot.xMins.append(xMin)
                    tfAnnot.xMaxs.append(xMax)
                    tfAnnot.yMins.append(yMin)
                    tfAnnot.yMaxs.append(yMax)
                    tfAnnot.textLabels.append(label.encode("utf8"))
                    tfAnnot.classes.append(config.CLASSES[label])
                    tfAnnot.difficult.append(0)

                    total += 1
            except Exception as e:
                print("warning image not found ",str(e))


            features = tf.train.Features(feature=tfAnnot.build())
            example = tf.train.Example(features=features)

            writer.write(example.SerializeToString())

            
        #close the writer and print diagnostics
        writer.close()
        print("[INFO] {} examples saved for ’{}’".format(total,dtype))

    
    


if __name__ == "__main__":
    tf.app.run()