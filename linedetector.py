"""
"""
import logging
import os
import sys
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_MODE: int = 0            # TODO: Output mode identifies if we are using control system
TRIM_FILTER: bool = False        # Do we want to trim the image outside of focus?
THRESH_FILTER: bool = False      # Do we want to highlight values above a threshold?

class LineDetector:
    """
    """
    def __init__(self, filename: str="sample", img = None):
        fmt_main: str = "%(asctime)s | %(levelname)s |\tLineDetector:\t%(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")

        if img is None:
            # Get Filepath from user
            filepath: str = filedialog.askopenfilename(
                title="Select Overlay Image",
                filetypes=[
                    ("Images", "png"),
                    ("Images", "jpg"),
                ]
            )
            if filepath == "":
                logging.error("No Image File Specified! Exiting...")
                sys.exit(1)
            logging.info("Processing Image: %s", filepath)

            # Read in image from user-supplied filepath
            img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            # Extract the Image name from the path
            filename: str = os.path.basename(filepath).split(".")[0]

        self.start: int = 0#350
        self.stop: int = len(img[0])#-12

        # This will contain a list of peaks and valleys
        self.cluster_list = []

        # Blur, then Compress the image into a 1D array of values, then scale to emphasize differences
        blur = cv2.GaussianBlur(img, (15, 5), 0)
        avg_img = self.compress_columns(blur)
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        print(self.cluster_list)
        mk = np.zeros((len(avg_img)), np.uint8)
        for i, val, climb in self.cluster_list:
            plt.plot(i, val, "og")
            if val < 127:
                color[:,i] = (0,0,255)
            elif climb > 0:
                color[:,i] = (255,0,0)
            else:
                color[:,i] = (0,255,0)
        cv2.imshow("marked", color)
        cv2.imwrite(f"{filename}_colored.png", color)

        # Plot the arrays to show the user what the distribution is
        tick_int: int = (self.stop-self.start) // 12
        columns = range(self.start, self.stop)
        column_label = range(self.start, self.stop, tick_int)
        values = range(0, 256, 16)
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
        plt.savefig(f"{filename}_graph.png")

        # Shut down OpenCV Windows and Pyplot
        # cv2.waitKey()
        plt.close()
        cv2.destroyAllWindows()

    def compress_columns(self, img):
        """ Compress an image to a 1D array of values
        
            This will generate two 1D arrays from the following values
                - the average of all values in each column
                - the minumum of all values in each column
        """
        if TRIM_FILTER:
            self.start, self.stop = self.select_zone(img)

        avg_col = np.zeros((self.stop-self.start), np.uint8)
        for x in range(len(avg_col)):
            avg_col[x] = np.average(img[:,x])

        # Rescale the images, trim to new start/stop, and convert to 1D
        avg_col = self.linear_reframe(avg_col)

        return avg_col

    def select_zone(self, img) -> tuple[int, int]:
        """
        """
        first_bar = True
        num_cols = len(img[0])
        start: int = 350
        stop: int = num_cols-10

        # Display usage instructions in terminal
        logging.info("How to use Trimmer:")
        logging.info("\t- 'esc' - skips the current video")
        logging.info("\t- 'enter' - accepts the current settings")
        logging.info("\t- 'left' - moves back 10 columns")
        logging.info("\t- 'right' - moves forward 10 columns")
        logging.info("\t- '1' - Select the Starting column")
        logging.info("\t- '2' - Select the Stopping column")

        # Loop until the user confirms the threshold value from the previews
        while True:
            # Draw lines on image
            color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            color[:,start] = (0,255,0)
            color[:,stop] = (0,0,255)

            # Show preview
            cv2.imshow("markers", color)

            # Use arrow keys to adjust threshold, up/down are fine tuning, left/right are bigger
            Key = cv2.waitKeyEx()
            if Key == 2424832:          # Left arrow, move bar left
                if first_bar:
                    start -= 10
                elif stop > start+10:
                    stop -= 10
            elif Key == 2555904:        # Right arrow, move bar right
                if not first_bar:
                    stop += 10
                elif start < stop-10:
                    start += 10
            elif Key == 49:        # 1 Key
                first_bar: bool = True
                logging.info("Now changing the Starting point...")
            elif Key == 50:        # 2 Key
                first_bar: bool = False
                logging.info("Now changing the Stopping point...")
            elif Key == 13:             # Enter Key, accept current settings
                return start, stop
            elif Key == 27:             # Escape Key
                logging.info("Skipping video...")
                return 0, num_cols-10
            else:
                logging.warning("Invalid Key: %d", Key)

            # Enforce bounds and debug
            if stop >= num_cols:
                stop = num_cols-1
            if start < 0:
                start = 0

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

        # If we want to manually filter out certain values, this will prompt us
        if THRESH_FILTER:
            rec_img = self.highlight(rec_img)

        self.cluster_list = self.mountain_climber(rec_img)

        return rec_img

    def highlight(self, img):
        """ Attempts to flatten out peaks and trim noise

            FIXME: This should be treated per peak, not on the image as a whole
        """
        # Display usage instructions in terminal
        logging.info("How to use Highlighter:")
        logging.info("\t- 'esc' - skips the current video")
        logging.info("\t- 'enter' - accepts the current settings")
        logging.info("\t- 'up' - increases the threshold")
        logging.info("\t- 'down' - decreases the threshold")

        # TODO: make thresh automatic
        thresh: int = 255

        # Loop until the user confirms the threshold value from the previews
        while True:
            mkd_img = np.copy(img)

            # Cut noise to floor
            # for x in range(len(mkd_img)):
            #     if mkd_img[x] > thresh:
            #         mkd_img[x] = 0
            #     else:
            #         mkd_img[x] = 255


            # Convert the 1D array into a 2D image and display it
            img_2d = np.stack([img for _ in range(100)], axis=0)
            mkd_img_2d = np.stack([mkd_img for _ in range(100)], axis=0)
            cv2.imshow("original", img_2d)
            cv2.imshow("averaged", mkd_img_2d)

            # Use arrow keys to adjust threshold, up/down are fine tuning, left/right are bigger
            Key = cv2.waitKeyEx()
            if Key == 2621440:          # Down arrow
                if thresh > 0:      # Enforce min bounds
                    thresh -= 1
                logging.info("Threshold = %d", thresh)
            elif Key == 2490368:        # Up arrow
                if thresh < 255:    # Enforce max bounds
                    thresh += 1
                logging.info("Threshold = %d", thresh)
            elif Key == 13:             # Enter key, accept current settings
                logging.info("Threshold value set to %d...", thresh)
                return mkd_img
            elif Key == 27:
                logging.info("Skipping Image Highlight...")
                return img
            elif Key == -1:
                pass
            else:
                logging.warning("Invalid Key: %d", Key)

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

        # recent_Valley: int = avg_img[0]
        # recent_Peak: int = avg_img[0]
        # recent_Direction: int = 0
        # lastValley: int = avg_img[0]

        # wall_thresh: int = 128

        # wall_loc1: int = -1
        # wall_loc2: int = -1

        # for col, val in enumerate(avg_img):
        #     recent

        #     if val >= wall_thresh:
        #         if 

        # StackOverflow     # init trackers
        last_valley: int = int(avg_img[0])
        last_peak: int = int(avg_img[0])
        last_val: int = int(avg_img[0])
        last_dir: int = getDirection(int(avg_img[0]), int(avg_img[1]))

        # get climbing
        peak_valley = [] # index, height, climb (positive for peaks, negative for valleys)
        for a in range(1, len(avg_img)):
            # get current direction
            sign = getDirection(last_val, int(avg_img[a]))
            last_val = int(avg_img[a])

            # if not equal, check gradient
            if sign != 0:
                if sign != last_dir:
                    # change in gradient, record peak or valley
                    # peak
                    if last_dir > 0:
                        last_peak = int(avg_img[a])
                        climb = last_peak - last_valley
                        climb = round(climb, 2)
                        peak_valley.append([a, int(avg_img[a]), climb])
                    else:
                        # valley
                        last_valley = int(avg_img[a])
                        climb = last_valley - last_peak
                        climb = round(climb, 2)
                        peak_valley.append([a, int(avg_img[a]), climb])

                    # change direction
                    last_dir = sign

        # filter out very small climbs
        filtered_pv = []
        for dot in peak_valley:
            if abs(dot[2]) > minClimb:
                filtered_pv.append(dot)
        return filtered_pv


if __name__ == "__main__":
    LineDetector()
