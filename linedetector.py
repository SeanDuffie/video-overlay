"""
"""

from tkinter import filedialog

import os
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np

# logging.info("Index = %d/%d\t|\tThreshold = %d", index, len(self.frame_arr)-1, self.thresh)
# logging.info("How to use:")
# logging.info("\t- 'esc' - skips the current video")
# logging.info("\t- 'enter' - accepts the current settings")
# logging.info("\t- 'space' - sets the current frame as the starting point")
# logging.info("\t- 'backspace' - sets the current frame as the ending point")
# logging.info("\t- 'left' - moves back one frame")
# logging.info("\t- 'right' - moves forward one frame")
# logging.info("\t- 'up' - increases the threshold")
# logging.info("\t- 'down' - decreases the threshold")


ZONE1 = True
ZONE2 = True

class LineDetector:
    """
    """
    def __init__(self):
        """
        """
        self.filepath, self.img = self.read_file()

        
        self.num_cols = len(self.img[0])

        self.start = 0
        self.stop = self.num_cols-1

        self.col_im = self.compress_columns()
        self.rec_im = self.linear_reframe()

        self.filename = os.path.basename(self.filepath)

        # x, y = enumerate(col_im[0])
        plt.plot(self.col_im[0,self.start:self.stop])
        plt.plot(self.rec_im[0,self.start:self.stop])
        plt.show()
        plt.savefig(f"graph_{self.filename}")

        # cv2.waitKey()
        cv2.destroyAllWindows()

    # def click_event(self, event, x, y, flags, params):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         print(f"({x},{y}) -> {self.col_im[y,x]}")


    def read_file(self):
        filename = filedialog.askopenfilename()
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        cv2.imshow("original", im)

        return filename, im

    def compress_columns(self):
        """ Compress an image to a 1xwidth array
        
            This will take the average of each value in a column, and generate a new image that is
            only one pixel tall with that
        """
        r, c = self.img.shape
        avg_col = np.zeros((r, c), np.uint8)

        for x in range(c):
            avg_col[:,x] = np.average(255-self.img[:,x])

        if ZONE1:
            self.select_zone(avg_col)
        else:
            cv2.imshow("columned", avg_col)

        return avg_col

    def select_zone(self, col_img):
        """
        """
        # color = cv2.cvtColor(col_img, cv2.COLOR_GRAY2BGR)
        first_bar = True

        num_cols = len(col_img[0])

        self.start = 0
        self.stop = num_cols-1

        # Loop until the user confirms the threshold value from the previews
        while True:
            # Draw lines on image
            color = cv2.cvtColor(col_img, cv2.COLOR_GRAY2BGR)
            color[:,self.start] = (0,255,0)
            color[:,self.stop] = (255,0,0)

            # Show preview
            cv2.imshow("columned", color)
            # cv2.setMouseCallback('columned', self.click_event)

            # Use arrow keys to adjust threshold, up/down are fine tuning, left/right are bigger
            Key = cv2.waitKeyEx()
            if Key == 2424832:          # Left arrow, move bar left
                if first_bar:
                    self.start -= 1
                elif self.stop > self.start+1:
                    self.stop -= 1
                logging.info("New range: (%d-%d)", self.start, self.stop)
            elif Key == 2621440:        # Down arrow
                first_bar = True
            elif Key == 2490368:        # Up arrow
                first_bar = False
            elif Key == 2555904:        # Right arrow, move bar right
                if not first_bar:
                    self.stop += 1
                elif self.start < self.stop-1:
                    self.start += 1
                logging.info("New range: (%d-%d)", self.start, self.stop)
            elif Key == 13:             # Enter key, accept current settings
                self.col_im = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                break
            elif Key == 27:
                self.skip_clip = True
                logging.info("Skipping video...")
                break
            else:
                logging.warning("Invalid Key: %d", Key)

            # Enforce bounds and debug
            if self.stop >= num_cols:
                self.stop = num_cols-1
            if self.start < 0:
                self.start = 0

    def linear_reframe(self):
        """
        """
        b_p = np.min(self.col_im[0])
        w_p = np.max(self.col_im[0])

        scale = (w_p-b_p)/255
        rec_img = np.clip(self.col_im, b_p, w_p) - b_p
        rec_img = (rec_img / scale).astype(np.uint8)

        if ZONE2:
            rec_img = self.highlight(rec_img)
        else:
            cv2.imshow("scaled", rec_img)

        return rec_img

    def highlight(self, rec_img):
        """
        """
        # color = cv2.cvtColor(rec_img, cv2.COLOR_GRAY2BGR)
        r, c = self.img.shape
        first_bar = True
        thresh = 127

        # Loop until the user confirms the threshold value from the previews
        while True:
            # Draw lines on image
            color = cv2.cvtColor(rec_img, cv2.COLOR_GRAY2BGR)

            for x in range(c):
                if rec_img[0,x] > thresh:
                    color[:,x] = (0,255,0)
            # color[:,self.start] = (0,255,0)
            # color[:,self.stop] = (255,0,0)

            # Show preview
            cv2.imshow("scaled", color)
            # cv2.setMouseCallback('scaled', self.click_event)

            # Use arrow keys to adjust threshold, up/down are fine tuning, left/right are bigger
            Key = cv2.waitKeyEx()
            # if Key == 2424832:          # Left arrow, move bar left
            if Key == 2621440:        # Down arrow
                thresh -= 1
            elif Key == 2490368:        # Up arrow
                thresh += 1
            # elif Key == 2555904:        # Right arrow, move bar right
            elif Key == 13:             # Enter key, accept current settings
                rec_im = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                return rec_im
            elif Key == 27:
                self.skip_clip = True
                logging.info("Skipping video...")
                break
            else:
                logging.warning("Invalid Key: %d", Key)

            # Enforce bounds and debug
            if thresh > 255:
                thresh = 255
            if thresh < 0:
                thresh = 0

    def mountain_climber(self):
        """
        """

if __name__ == "__main__":
    LineDetector()
