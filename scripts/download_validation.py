import os
import urllib.request as dowReq
import zipfile

val = input("Enter BDD or COCO :")
if(val == "COCO"):
    url = "https://cloud.hipert.unimore.it/s/LNxBDk4wzqXPL8c/download"
    lib = "..\demo\COCO_val2017"
    lib_zip = "COCO_val2017.zip"
elif(val == "BDD"):
    url = "https://cloud.hipert.unimore.it/s/bikqk3FzCq2tg4D/download"
    lib = "..\demo\BDD100k_val"
    lib_zip = "BDD100k_val.zip"

dowReq.urlretrieve(url,lib_zip)

with zipfile.ZipFile(lib_zip,'r') as zip_ref:
    zip_ref.extractall(lib)

labelFolder = lib + "\labels"
imageFolder = lib + "\images"

file1 = open(".\\..\\demo\\all_labels.txt","a")
path1 = os.path.realpath(labelFolder)
for file in os.listdir(labelFolder):
    valTemp = path1 + "\\" + file
    valTemp = valTemp + '\n'
    file1.write(valTemp)
file1.close()

file2 = open(".\\..\\demo\\all_images.txt","a")
path2 = os.path.realpath(imageFolder)
for file in os.listdir(imageFolder):
    pathtemp = path2 + "\\" + file
    pathtemp = pathtemp + '\n'
    file2.write(pathtemp)
file2.close()

print("Completed")
