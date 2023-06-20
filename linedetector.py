"""
"""
import logging
import os
import sys
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_MODE: int = 0
TRIM_FILTER: bool = True        # Do we want to trim the image outside of focus?
THRESH_FILTER: bool = True      # Do we want to highlight values above a threshold?

class LineDetector:
    """
    """
    def __init__(self, filepath = "", img = None):
        fmt_main = "%(asctime)s | %(levelname)s |\tLineDetector:\t%(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")

        if img == None:
            # Get Filepath from user
            filepath = filedialog.askopenfilename(
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

            self.start: int = 0
            self.stop: int = len(img[0])

        # Extract the Image name from the path
        filename: str = os.path.basename(filepath).split(".")[0]

        # Compress the image into a 1D array of values, then scale to emphasize differences
        avg_img, min_img = self.compress_columns(img)

        # Plot the arrays to show the user what the distribution is
        tick_int: int = (self.stop-self.start) // 12
        columns = range(self.start, self.stop)
        column_label = range(self.start, self.stop, tick_int)
        values = range(0, 256, 16)
        plt.plot(columns, avg_img, label="Average")
        plt.plot(columns, min_img, label="Minimum")

        # Format the Plot and either show or save to PNG
        plot_name = f"graph_{filename}"
        plt.xlabel("Pixel Columns")
        plt.xticks(ticks=column_label, labels=column_label)
        plt.ylabel("Value")
        plt.yticks(ticks=values, labels=values)
        plt.title(plot_name)
        plt.legend(
            loc="upper left",
            fancybox=True,
            shadow=True,
            ncol=1,
        )
        # plt.show()
        plt.savefig(f"{plot_name}.png")

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
        r, c = img.shape
        avg_col = np.zeros((r, c), np.uint8)
        min_col = np.zeros((r, c), np.uint8)

        for x in range(c):
            avg_col[:,x] = np.average(img[:,x])
            min_col[:,x] = np.min(img[:,x])

        if TRIM_FILTER:
            start, stop = self.select_zone(avg_col)
            self.start = start
            self.stop = stop
            avg_col = avg_col[0,start:stop]
            min_col = min_col[0,start:stop]
        else:
            avg_col = avg_col[0,:]
            min_col = min_col[0,:]

        avg_col = self.linear_reframe(avg_col)
        min_col = self.linear_reframe(min_col)

        return avg_col, min_col

    def select_zone(self, col_img) -> tuple[int, int]:
        """
        """
        # color = cv2.cvtColor(col_img, cv2.COLOR_GRAY2BGR)
        first_bar = True
        num_cols = len(col_img[0])
        start: int = 300
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
            color = cv2.cvtColor(col_img, cv2.COLOR_GRAY2BGR)
            color[:,start] = (0,255,0)
            color[:,stop] = (0,0,255)

            # Show preview
            cv2.imshow("columned", color)

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
                first_bar = True
            elif Key == 50:        # 2 Key
                first_bar = False
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
        if THRESH_FILTER:
            img = self.highlight(img)

        b_p = np.min(img)
        w_p = np.max(img)

        scale = (w_p-b_p)/255
        rec_img = np.clip(img, b_p, w_p) - b_p
        rec_img = (rec_img / scale).astype(np.uint8)

        return rec_img

    def highlight(self, img):
        """
        """
        thresh: int = 255

        # Display usage instructions in terminal
        logging.info("How to use Highlighter:")
        logging.info("\t- 'esc' - skips the current video")
        logging.info("\t- 'enter' - accepts the current settings")
        logging.info("\t- 'up' - increases the threshold")
        logging.info("\t- 'down' - decreases the threshold")

        # Loop until the user confirms the threshold value from the previews
        while True:
            # Draw lines on image
            # color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # mkd_img = np.minimum(img, thresh)
            mkd_img = np.copy(img)

            # for x in range(c):
            #     if color[0,x] > thresh:
            #         color[:,x] = (0,255,0)

            for x in range(len(mkd_img)):
                if mkd_img[x] > thresh:
                    mkd_img[x] = 0


            # Show preview
            h = 30
            mkd_img_2d = np.repeat(mkd_img, h).reshape((h, len(mkd_img)))
            cv2.imshow("scaled", mkd_img_2d)

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
                # img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                return mkd_img
            elif Key == 27:
                logging.info("Skipping Image Highlight...")
                return img
            else:
                logging.warning("Invalid Key: %d", Key)

    def mountain_climber(self):
        """
        """

if __name__ == "__main__":
    LineDetector()
