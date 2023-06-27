"""
"""
import logging
import os
import sys
from tkinter import filedialog
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


class LineDetector:
    """
    """
    def __init__(self, dest: str = "./outputs", fname: str = "graph_name", img = None):
        fmt_main: str = "%(asctime)s | %(levelname)s |\tLineDetector:\t%(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")

        outpath = dest
        filename = fname

        if img is None:
            # Get Filepath from user
            inpath: str = filedialog.askopenfilename(
                title="Select Input Overlay Image",
                filetypes=[
                    ("Images", "png"),
                    ("Images", "jpg"),
                ]
            )
            if inpath == "":
                logging.error("No Image File Specified! Exiting...")
                sys.exit(1)
            logging.info("Processing Image: %s", inpath)

            # Read in image from user-supplied filepath
            img = cv2.imread(inpath, cv2.IMREAD_GRAYSCALE)

            # Extract the Image name from the path
            filename: str = os.path.basename(inpath).split(".")[0]

        if outpath == "":
            # Get Output Destination from user
            outpath: str = filedialog.askdirectory( title="Select Output Directory" )
            if outpath == "":
                logging.error("No Image File Specified! Exiting...")
                sys.exit(1)
            logging.info("Processing Image: %s", outpath)

        # This will contain a list of peaks and valleys
        self.cluster_bounds: list[Any] = []
        self.cluster_marks: list[Any] = []
        self.wall_marks: list[Any] = []

        # Blur, then Compress the image into a 1D array of values, then scale to emphasize differences
        blur = cv2.GaussianBlur(img, (15, 5), 0)        # TODO: experiment with different values of blur
        avg_img = self.compress_columns(blur)
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # logging.info("Wall Marks: " + str(self.wall_marks))
        # logging.info("Cluster Marks: " + str(self.cluster_marks))
        # logging.info("Cluster Bounds: " + str(self.cluster_bounds))
        for i, val, climb in self.cluster_marks:
            plt.plot(i, val, "vg")
            color[:,i] = (0,255,0)
        for i, val, climb in self.cluster_bounds:
            plt.plot(i, val, "Xb")
            color[:,i] = (255,0,0)
        for i, val, climb in self.wall_marks:
            plt.plot(i, val, "Xr")
            color[:,i] = (0,0,255)

        cv2.imwrite(f"{outpath}/{filename}_colored.png", color)

        # Plot the arrays to show the user what the distribution is
        tick_int: int = len(avg_img) // 12
        columns = range(len(avg_img))
        column_label = range(len(avg_img), tick_int)
        values = range(0, 257, 16)
        plt.plot(columns, avg_img, label="Average")

        # Format the Plot and either show or save to PNG
        plt.xlabel("Pixel Columns")
        plt.xticks(ticks=column_label, labels=column_label)
        plt.ylabel("Value")
        plt.yticks(ticks=values, labels=values)
        plt.title(os.path.basename(filename))
        plt.legend(
            loc="upper left",
            fancybox=True,
            shadow=True,
            ncol=1,
        )
        # plt.show()
        plt.savefig(f"{outpath}/{filename}_graph.png")

        # Shut down OpenCV Windows and Pyplot
        # cv2.waitKey()
        # haze = np.average(avg_img)
        # logging.info("Haze: %f", haze)
        plt.close()
        cv2.destroyAllWindows()

    def compress_columns(self, img):
        """ Compress an image to a 1D array of values
        
            This will generate two 1D arrays from the following values
                - the average of all values in each column
                - the minumum of all values in each column
        """
        avg_col = np.zeros(len(img[0]), np.uint8)
        for x in range(len(avg_col)):
            avg_col[x] = np.average(img[:,x])

        # Rescale the images, trim to new start/stop, and convert to 1D
        avg_col = self.linear_reframe(avg_col)

        return avg_col


    def linear_reframe(self, img):
        """ Using a 1D array, it will readjust the value scale to make differences more noticable
        
            This will be done by finding the min and max values, subtracting the min, and
            multiplying by the different between the two scaled to 255.
        """
        # Acquire the Min and Max
        b_p = np.min(img)
        w_p = np.max(img)

        # Cut off any values above the white point, subtract by the black point, then multiply to scale from 0-255
        scale = (w_p-b_p)/255
        rec_img = np.clip(img, b_p, w_p) - b_p
        rec_img = (rec_img / scale).astype(np.uint8)

        self.mountain_climber(rec_img)

        return rec_img


    def mountain_climber(self, avg_img, minClimb: int = 0):
        """
        """
        def getDirection(a: int, b: int) -> int:
            dx: int = b - a
            if dx == 0:
                return 0
            elif dx > 0:
                return -1
            return 1

        # init trackers
        last_valley: int = int(avg_img[0])
        last_peak: int = int(avg_img[0])
        last_val: int = int(avg_img[0])
        last_dir: int = 0

        # get climbing (each element should contain [index, height, climb])
        cluster_bounds: list[Any] = []     # Should outline the cluster of cells
        cluster_marks: list[Any] = []      # Should line up on center of cluster
        wall_marks: list[Any] = []         # Bounding walls of channel (not cells)
        for a in range(1, len(avg_img)-1):
            # get current values
            cur_val: int = int(avg_img[a])
            cur_dir: int = getDirection(last_val, cur_val)
            next_val: int = int(avg_img[a+1])
            next_dir: int = getDirection(cur_val, next_val)

            
            if cur_val > 127:           # Filter out left wall
                if next_val < 127:      # Right Wall
                    # logging.info("Found right wall at: %d", a)
                    wall_marks.append([a, cur_val, 127])

                # if cur_dir == 0:
                #     if (last_dir <= 0 and cur_dir == 1) or (last_dir == -1 and next_dir >= 0):
                #         cluster_marks.append([a, cur_val, 0])
                # if not equal, check gradient
                if cur_dir != 0:
                    if cur_dir != last_dir:
                        # change in gradient, record peak or valley
                        if last_dir > 0:
                            # peak
                            # logging.info("Found peak at: %d", a)
                            last_peak = cur_val
                            climb = last_peak - last_valley
                            climb = round(climb, 2)
                            cluster_marks.append([a, cur_val, climb])
                        else:
                            # valley
                            # logging.info("Found valley at: %d", a)
                            last_valley = cur_val
                            climb = last_valley - last_peak
                            climb = round(climb, 2)
                            cluster_bounds.append([a, cur_val, climb])
                    last_dir = cur_dir

                # Prepare for next iteration
                last_val = cur_val
            elif next_val > 127:
                # logging.info("Found left wall at: %d", a)
                wall_marks.append([a, cur_val, 127])

        # filter out very small climbs
        # filtered_pv = []
        # for dot in cluster_marks:
        #     if abs(dot[2]) > minClimb:
        #         filtered_pv.append(dot)

        self.cluster_bounds = cluster_bounds
        self.cluster_marks = cluster_marks
        self.wall_marks = wall_marks


if __name__ == "__main__":
    LineDetector()
