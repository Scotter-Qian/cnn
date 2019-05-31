import numpy as np
import os
import pickle
import cv2

def search_file(path):
    leakage_number = 0
    noleakage_number = 0
    total_number = 0
    f1 = open(r"./path_sets(noleakage)", "w")
    f2 = open(r"./path_sets(leakage)", "w")
    for file in os.listdir(path):
        total_number += 1
        file_path = os.path.join(path, file)
        print(file_path)
        assert file_path.endswith(".xy")
        file = open(file_path, "rb+")
        data = pickle.load(file)
        if data["y"] == 0:
            noleakage_number += 1
            f1.write(file_path + "\n")
        else:
            leakage_number += 1
            f2.write(file_path + "\n")
    f1.close()
    f2.close()

    return "leakage_number: %s, noleakage_number: %s, total_number: %s"%(leakage_number, noleakage_number, total_number)

def data_expansion(path):
    leakage_number = 0
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        print(file_path)
        assert file_path.endswith(".xy")
        file_data = open(file_path, "rb+")
        data = pickle.load(file_data)
        if data["y"] == 1:
            leakage_number += 1
        else:
            new_file_path = os.path.join(path, "new2" + file)
            f = open(new_file_path, "wb+")
            image, label = transform(file_path)
            pickle.dump({"x": image, "y": label}, f)
            f.close()

    return "noleakage_number: %s"%(leakage_number)


def rotation180(file_path):
    f = open(file_path, "rb+")
    data = pickle.load(f)
    image = data["x"]
    label = data["y"]
    image = cv2.rotate(image, cv2.ROTATE_180)

    return image, label

def flip(file_path):
    f = open(file_path, "rb+")
    data = pickle.load(f)
    image = data["x"]
    label = data["y"]
    image = cv2.flip(src=image, flipCode=50)

    return image, label

def CLAHE(file_path):
    f = open(file_path, "rb+")
    data = pickle.load(f)
    image = data["x"]
    label = data["y"]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)

    return image, label

def histogram_normalization(file_path):
    f = open(file_path, "rb+")
    data = pickle.load(f)
    image = data["x"]
    label = data["y"]
    image_max = np.max(image)
    image_min = np.min(image)
    Omin, Omax = 0, 255
    a = float(Omax-Omin)/(image_max-image_min)
    b = Omin - a*image_min
    image = a*image + b
    image = image.astype(np.uint8)

    return image, label

def gaussian_smoothing(file_path):
    from scipy import signal
    f = open(file_path, "rb+")
    data = pickle.load(f)
    image = data["x"]
    label = data["y"]
    gaussKernel_x = cv2.getGaussianKernel(ksize=3, sigma=3)
    gaussKernel_x = np.transpose(gaussKernel_x)
    gaussBlur_x = signal.convolve2d(image, gaussKernel_x, mode="same", boundary="symm", fillvalue=0)
    gaussKernel_y = cv2.getGaussianKernel(ksize=3, sigma=3)
    gaussBlur_xy = signal.convolve2d(gaussBlur_x, gaussKernel_y, mode="same", boundary="symm", fillvalue=0)
    image = np.round(gaussBlur_xy)
    image = image.astype(np.uint8)

    return image, label

def getClosenessWeight(sigma_g, H, W):
    import math
    r,c = np.mgrid[0:H:1, 0:W:1]
    r = (r - (H-1)/2).astype(np.uint8)
    c = (c - (W-1)/2).astype(np.uint8)
    closeWeight = np.exp(-0.5*(np.power(r, 2) + np.power(c, 2))/math.pow(sigma_g, 2))

    return closeWeight
def bfltGray(I, H, W, sigma_g, sigma_d):
    import math
    closenessWeight = getClosenessWeight(sigma_g, H, W)
    cH = int((H-1)/2)
    cW = int((W-1)/2)
    rows, cols = I.shape
    bfltGrayImage = np.zeros(I.shape, np.float32)
    for r in range(rows):
        for c in range(cols):
            pixel = I[r][c]
            rTop = 0 if r-cH<0 else r-cH
            rBottom = rows-1 if r+cH>rows-1 else r+cH
            cLeft = 0 if c-cW<0 else c-cW
            cRight = cols-1 if c+cW>cols-1 else c+cW
            region = I[rTop:rBottom+1, cLeft:cRight+1]
            similarityWeightTemp = np.exp(-0.5*np.power(region-pixel, 2.0)/math.pow(sigma_d, 2))
            closenessWeightTemp = closenessWeight[rTop-r+cH:rBottom-r+cH+1, cLeft-c+cW:cRight-c+cW+1]
            weightTemp = similarityWeightTemp*closenessWeightTemp
            weightTemp = weightTemp/np.sum(weightTemp)
            bfltGrayImage[r][c] = np.sum(region*weightTemp)
    return bfltGrayImage.astype(np.uint8)
def bilateral_filtering(file_path):
    f = open(file_path, "rb+")
    data = pickle.load(f)
    image = data["x"]
    label = data["y"]
    image = bfltGray(image, 33, 33, 19, 0.2)

    return image, label

def errosion(file_path):
    f = open(file_path, "rb+")
    data = pickle.load(f)
    image = data["x"]
    label = data["y"]
    s = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    image = cv2.erode(image, s)

    return image, label

def transform(file_path):
    transform_key = np.random.randint(0,7)
    print(transform_key)
    transform_dict = dict(zip(("0123456"),[rotation180(file_path),
                                           flip(file_path),
                                           CLAHE(file_path),
                                           histogram_normalization(file_path),
                                           gaussian_smoothing(file_path),
                                           bilateral_filtering(file_path),
                                           errosion(file_path)]))

    image, label = transform_dict["%d"%transform_key]

    return image, label


if __name__=="__main__":
    path = r"D:\CNN\data\test"
    print(data_expansion(path))