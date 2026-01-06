import numpy as np
import cv2 as cv
import io
from matplotlib import pyplot as plt

def _plt2np(fig:plt.Figure, dpi:int=180) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv.imdecode(img_arr, 1)

    return img

def plot_data_points(
        input:np.ndarray,
        target:np.ndarray,
        prediction:np.ndarray = None,
        title:str="Data Plot",
        xlabel:str="x",
        ylabel:str="y"
    ):
    s = 6.
    fig = plt.figure(figsize=(6, 4))
    plt.title(title)
    plt.minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='lightgray')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray', alpha=0.5)
    plt.scatter(input, target, c='red', alpha=0.5, label='Ground Truth', s=s)
    if prediction is not None:
        plt.scatter(input, prediction, c='blue', alpha=0.5, label='Prediction', s=s)
    
    xmin, xmax, ymin, ymax = plt.axis()
    plt.hlines(0, xmin=xmin, xmax=xmax, colors='black', linestyles='-', linewidth=0.5, alpha=0.5)
    plt.vlines(0, ymin=ymin, ymax=ymax, colors='black', linestyles='-', linewidth=0.5, alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    plt.legend()
    plt.show()

    #img = _plt2np(fig, dpi=180)
    plt.close(fig)

    #return img

def plot_series(
    data:np.ndarray,
    title:str="Data Series",
    xlabel:str="x",
    ylabel:str="y",  
    ):
    fig = plt.figure(figsize=(6, 4))
    plt.plot(data, c='blue', alpha=0.5)
    plt.title(title)
    plt.minorticks_on()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')
    plt.show()

    #img = _plt2np(fig, dpi=180)
    plt.close(fig)

    #return img