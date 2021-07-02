from xml.etree import ElementTree
import numpy as np
import argparse
import imutils
import cv2
import os


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image filename")
args = vars(ap.parse_args())
name = args["image"].split(".")[0]

image = cv2.imread(args["image"])
stomata = []


## Cut and save the bounding boxes
clone = image.copy()
with open(name+'.xml', 'rt') as f:
    tree = ElementTree.parse(f)

path = os.getcwd()
try:
    os.mkdir(path+"/dataset")
except:
    pass
try:
    os.mkdir(path+"/dataset/stomata")
except:
    pass
try:
    os.mkdir(path+"/dataset/stomata/"+name)
except OSError:
    print ("Creation of the directory "+"/dataset/stomata/"+name+" failed or already exists")
os.chdir(path+"/dataset/stomata/"+name)

n = 0
for node in tree.findall('.//bndbox'):
    xmin = int(node.find('xmin').text)-3
    if xmin < 0:
        xmin = 0

    xmax = int(node.find('xmax').text)+3
    if xmax > image.shape[1]:
        xmax = image.shape[1]

    ymin = int(node.find('ymin').text)-3
    if ymin < 0:
        ymin = 0

    ymax = int(node.find('ymax').text)+3
    if ymax > image.shape[0]:
        ymax = image.shape[0]

    stomata.append(image[ymin:ymax,xmin:xmax])
    cv2.imwrite(name+"_"+str(n)+".png", image[ymin:ymax,xmin:xmax])
    n = n+1


## Search and select contours. Create label images
try:
    os.mkdir(path+"/dataset/ellipse_inv/")
except:
    pass
try:
    os.mkdir(path+"/dataset/ellipse/")
except:
    pass
try:
    os.mkdir(path+"/dataset/k_means/")
except:
    pass
try:
    os.mkdir(path+"/dataset/hull_inv/")
except:
    pass
try:
    os.mkdir(path+"/dataset/hull_inv_erode/")
except:
    pass
try:
    os.mkdir(path+"/dataset/labels/")
except:
    pass
try:
    os.mkdir(path+"/dataset/ellipse_inv/"+name)
except OSError:
    print ("Creation of the directory "+"/dataset/ellipse_inv/"+name+" failed or already exists")
try:
    os.mkdir(path+"/dataset/ellipse/"+name)
except OSError:
    print ("Creation of the directory "+"/dataset/ellipse/"+name+" failed or already exists")
try:
    os.mkdir(path+"/dataset/k_means/"+name)
except OSError:
    print ("Creation of the directory "+"/dataset/k_means/"+name+" failed or already exists")
try:
    os.mkdir(path+"/dataset/hull_inv/"+name)
except OSError:
    print ("Creation of the directory "+"/dataset/hull_inv/"+name+" failed or already exists")
try:
    os.mkdir(path+"/dataset/hull_inv_erode/"+name)
except OSError:
    print ("Creation of the directory "+"/dataset/hull_inv_erode/"+name+" failed or already exists")
try:
    os.mkdir(path+"/dataset/labels/"+name)
except OSError:
    print ("Creation of the directory "+"/dataset/labels/"+name+" failed or already exists")
os.chdir(path+"/dataset/labels/"+name)

xmlRoot = ElementTree.Element('image')
xmlName = ElementTree.SubElement(xmlRoot, 'fileName')
xmlName.text = args["image"]
xmlLabels = ElementTree.SubElement(xmlRoot, 'labels')
xmlLabels.set('size', str(len(stomata)))

n = 0
for stoma in stomata:
    gray = cv2.equalizeHist(cv2.cvtColor(stoma.copy(), cv2.COLOR_BGR2GRAY))

    ## Ellipses
    dilated = cv2.dilate(gray, None, iterations=1)
    thresh = cv2.adaptiveThreshold(dilated, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1)
    thresh_inv = cv2.adaptiveThreshold(dilated, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 1)

    blue = stoma.copy()
    cnts_inv = cv2.findContours(thresh_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_inv = imutils.grab_contours(cnts_inv)
    if (len(cnts_inv)>0):
        c_inv = cnts_inv[0]
        for cnt in cnts_inv:
            if (cv2.contourArea(c_inv) < cv2.contourArea(cnt)):
                c_inv = cnt
        try:
            ellipse_inv = cv2.fitEllipse(c_inv)
            cv2.ellipse(blue, ellipse_inv, (255, 0, 0), 1)
        except:
            pass

    green = stoma.copy()
    cnts = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if (len(cnts)>0):
        c = cnts[0]
        for cnt in cnts:
            if (cv2.contourArea(c) < cv2.contourArea(cnt)):
                c = cnt
        try:
            ellipse = cv2.fitEllipse(c)
            cv2.ellipse(green, ellipse, (0, 255, 0), 1)
        except:
            pass

    #K-Means
    yellow = stoma.copy()
    vectorized = stoma.reshape((-1,3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(vectorized,3,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]

    kerode = cv2.erode(res.reshape((stoma.shape)), None, iterations=2)
    kgray = cv2.cvtColor(kerode, cv2.COLOR_BGR2GRAY)
    min = 255
    for color in center:
        value = 0.299*color[0]+0.587*color[0]+0.114*color[0]
        if (value<min):
            min = value
    T,kThresh = cv2.threshold(kgray, min-1, 255, cv2.THRESH_BINARY_INV)

    cnts_means = cv2.findContours(kThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_means = imutils.grab_contours(cnts_means)
    if (len(cnts_means)>0):
        k_means = cnts_means[0]
        for cnt in cnts_means:
            if (cv2.contourArea(k_means) < cv2.contourArea(cnt)):
                k_means = cnt
        try:
            k_means = cv2.convexHull(k_means)
            cv2.drawContours(yellow, [k_means], 0, (0,255,255), 1)
        except:
            pass

    ## Convex hulls
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 1)
    thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 1)
    thresh_inv_erode = cv2.erode(thresh_inv, None, iterations=1)

    red = stoma.copy()
    cnts_inv = cv2.findContours(thresh_inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_inv = imutils.grab_contours(cnts_inv)
    if (len(cnts_inv)>0):
        c_inv = cnts_inv[0]
        for cnt in cnts_inv:
            if (cv2.contourArea(c_inv) < cv2.contourArea(cnt)):
                c_inv = cnt
        try:
            hull_inv = cv2.convexHull(c_inv)
            cv2.drawContours(red, [hull_inv], 0, (0,0,255), 1)
        except:
            pass

    pink = stoma.copy()
    cnts_inv_erode = cv2.findContours(thresh_inv_erode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts_inv_erode = imutils.grab_contours(cnts_inv_erode)
    if (len(cnts_inv_erode)>0):
        c_inv_erode = cnts_inv_erode[0]
        for cnt in cnts_inv_erode:
            if (cv2.contourArea(c_inv_erode) < cv2.contourArea(cnt)):
                c_inv_erode = cnt
        try:
            hull_inv_erode = cv2.convexHull(c_inv_erode)
            cv2.drawContours(pink, [hull_inv_erode], 0, (255,0,255), 1)
        except:
            pass

    row1 = cv2.hconcat([stoma, red, pink])
    row2 = cv2.hconcat([blue, green, yellow])
    table = cv2.vconcat([row1, row2])
    cv2.imshow("Choose contour for "+str(n), cv2.resize(table, (int(table.shape[1]*3), int(table.shape[0]*3)), interpolation = cv2.INTER_AREA))

    ## Select label
    key = cv2.waitKey(0)
    while (key < 49 or key > 54):
        key = cv2.waitKey(0)

    cv2.destroyWindow("Choose contour for "+str(n))

    xmlLabel = ElementTree.SubElement(xmlLabels, 'label')
    xmlLabel.set('num', str(n))

    w=stoma.shape[1]
    h=stoma.shape[0]

    blank_blue = np.zeros((h,w,1), np.uint8)
    try:
        cv2.ellipse(blank_blue, ellipse_inv, (255, 255, 255), -1)
    except:
        pass
    cv2.imwrite(os.path.join(path+"/dataset/ellipse_inv/"+name, name+"_"+str(n)+".png"), blank_blue)

    blank_green = np.zeros((h,w,1), np.uint8)
    try:
        cv2.ellipse(blank_green, ellipse, (255, 255, 255), -1)
    except:
        pass
    cv2.imwrite(os.path.join(path+"/dataset/ellipse/"+name, name+"_"+str(n)+".png"), blank_green)

    blank_yellow = np.zeros((h,w,1), np.uint8)
    try:
        cv2.drawContours(blank_yellow, [k_means], 0, (255,255,255), -1)
    except:
        pass
    cv2.imwrite(os.path.join(path+"/dataset/k_means/"+name, name+"_"+str(n)+".png"), blank_yellow)

    blank_red = np.zeros((h,w,1), np.uint8)
    try:
        cv2.drawContours(blank_red, [hull_inv], 0, (255,255,255), -1)
    except:
        pass
    cv2.imwrite(os.path.join(path+"/dataset/hull_inv/"+name, name+"_"+str(n)+".png"), blank_red)

    blank_pink = np.zeros((h,w,1), np.uint8)
    try:
        cv2.drawContours(blank_pink, [hull_inv_erode], 0, (255,255,255), -1)
    except:
        pass
    cv2.imwrite(os.path.join(path+"/dataset/hull_inv_erode/"+name, name+"_"+str(n)+".png"), blank_pink)

    select = np.zeros((h,w,1), np.uint8)
    if(key==49): ## Blue
        select = blank_blue
        xmlLabel.text = "ellipse_inv"
    if(key==50): ## Green
        select = blank_green
        xmlLabel.text = "ellipse"
    if(key==51): ## Yellow
        select = blank_yellow
        xmlLabel.text = "k_means"
    if(key==52): ## Manual
        xmlLabel.text = "manual"
    if(key==53): ## Red
        select = blank_red
        xmlLabel.text = "hull_inv"
    if(key==54): ## Pink
        select = blank_pink
        xmlLabel.text = "hull_inv_erode"


    cv2.imwrite(os.path.join(path+"/dataset/labels/"+name, name+"_"+str(n)+".png"), select)
    n = n+1

tree = ElementTree.ElementTree(xmlRoot)
tree.write(name+'_labels.xml')