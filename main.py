import cv2

from linedetector import LineDetector
from overlay import Overlay

if __name__ == "__main__":
    filepath = "G:/Lynntech/Projects/Export Controlled/ARM-1912II E/4929000_POCIBA/300 Cartridge/4-Testing/2023-04-12 New Data for Sean 8loop 2-3-4milmin/Basler_acA640-750um__22808485__20230412_100910210_8loop_15and6um_1-50_2milmin.png"

    # Read in image from user-supplied filepath
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Generate the Plot from the image
    LD = LineDetector(filename="Main Test", img=img)