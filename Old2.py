#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import numpy as np
from flask import request
import json
import time
import requests as rq
import settings
import helper
from pathlib import Path
import PIL
from PIL import Image
import numpy as np
import settings
import helper
import pybase64
from logging import FileHandler,WARNING
from flask import Flask, send_from_directory, render_template
import os

app = Flask(__name__, static_folder='static')

@app.route('/files/<path:path>')
def send_file(path):
    return send_from_directory('static', path)

file_handler = FileHandler('errorlog.txt')
file_handler.setLevel(WARNING)

@app.route('/ImageScan',methods = ['POST'])
def handle_request():
    confidence = 0.4
    decoded={'input':request.json['input']}
    print(decoded)
    print(decoded['input'])
    try:
        decoded_data=pybase64.b64decode((decoded['input']))
        #write the decoded data back to original format in  file
        img_file = open('image.jpeg', 'wb')
        img_file.write(decoded_data)
        img_file.close()
        model_path = Path(settings.DETECTION_MODEL)
        model = helper.load_model(model_path)
        res = model.predict('image.jpeg',conf=confidence,hide_conf=True)
        print('Done')
        try:
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:, :, ::-1]
            image = Image.fromarray(res_plotted)
            # Save the image as a JPEG file
            filename = "output.jpeg"
            output_directory = 'static//'
            output_path = output_directory + filename
            image.save(output_path)
            #os.path.join(os.getcwd(), filename)
            #with open("output.jpeg", "rb") as img_file:
                #my_string_op = pybase64.b64encode(img_file.read())
                #op_path = my_string_op.decode('utf-8')
            names = model.names

            cla = []

            for c in res[0].boxes.cls:
                cla.append(names[int(c)])

            arrxy=res[0].boxes.xyxy
            coordinates = np.array(arrxy)


            x_coords = (coordinates[:, 0] + coordinates[:, 2]) / 2

            y_coords = (coordinates[:, 1] + coordinates[:, 3]) / 2

            midpoints = np.column_stack((x_coords, y_coords))

            rounded_n_sorted_arr = np.round(midpoints[midpoints[:, 1].argsort()]).astype(int)

            count=1
            objects=0
            group_sizes = []

            Obj = []

            for i,j in zip(range(1, len(rounded_n_sorted_arr)),cla):

                if(rounded_n_sorted_arr[i][1] - rounded_n_sorted_arr[i-1][1] > 130 ):
                    group_sizes.append(objects + 1)
                    count += 1
                    objects = 0
                    Obj.append(j)

                else:
                    objects += 1

            group_sizes.append(objects + 1)

            rc = []

            for i,j in zip(list(midpoints),cla):
                k = list(i)
                k.append(j)
                gh = k
                rc.append(gh)

            sorted_list = sorted(rc, key=lambda x: x[1])

            big = []
            for i in group_sizes:
                count=0
                li = []
                al = []
                for j in sorted_list:
                    count = count+1
                    li.append(j[2])
                    al.append(j)
                    if(count==i):
                        break
                for k in al:
                    sorted_list.remove(k)
                big.append(li)

            data = big
            # Initialize a dictionary to store class counts for each list
            list_class_counts = []

            # Iterate through each list in the data
            for sublist in data:
                # Create a dictionary to store class counts for the current list
                class_counts = {}

                # Iterate through each element in the sublist
                for item in sublist:
                    # Check if the class is already in the dictionary, if not, initialize it with 1
                    if item not in class_counts:
                        class_counts[item] = 1
                    else:
                        # If the class is already in the dictionary, increment the count
                        class_counts[item] += 1

                # Append the class counts for the current list to the list_class_counts
                list_class_counts.append(class_counts)

            Detection_Dic = {}

            for i, class_counts,j in zip(range(1,len(list_class_counts)+1), list_class_counts,group_sizes):
                print(f"Shelf {i}:")
                Detect_Dic = {}
                for class_name, count in class_counts.items():  
                    Detect_Dic_ = {}
                    print(f"         Brand: {class_name}, Count: {count}, Percentage: {round((count/j)*100,2)}")
                    Detect_Dic_['Count'] = count
                    Detect_Dic_["Percentage"] = round((count/j)*100,2)
                    Detect_Dic[class_name] = Detect_Dic_
                Detection_Dic[f"Shelf {i}"] = Detect_Dic
            Detection_Di = Detection_Dic
            Li = []
            errorCode = 1
            errorMsg = 'Success'
            imagePath = 'http://10.1.1.34:8000/'+ output_path
            Li.append(Detection_Di)
            data_set = {"errorCode": errorCode, "errorMsg": errorMsg, 'ImagePath':imagePath, 'DetectionResult': Li}
            json_dump = json.dumps(data_set)
            print('Done2')
        except:
            img_file = open('image.jpeg', 'wb')
            img_file.write(decoded_data)
            filename = "output.jpeg"
            output_directory = 'static//'
            output_path = output_directory + filename
            img_file.save(output_path)
            errorCode = 1
            errorMsg = 'Success'
            imagePath = 'http://10.1.1.34:8000/'+ output_path
            Li = []
            data_set = {"errorCode": errorCode, "errorMsg": errorMsg, 'ImagePath':imagePath, 'DetectionResult': Li}
            json_dump = json.dumps(data_set)
            print('Done3')
    except:
        errorCode = 0
        errorMsg = 'Failed'
        imagePath = ''
        Li = []
        data_set = {"errorCode": errorCode, "errorMsg": errorMsg, 'ImagePath':imagePath, 'DetectionResult':Li}
        json_dump = json.dumps(data_set)
        print('Done4')
        pass     

    return json_dump


if __name__ == '__main__':
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run(host='10.1.1.34', port=8000)



# In[ ]:




