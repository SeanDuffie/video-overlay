""" overlay.py
"""
import datetime
import logging
import os
import sys
from tkinter import filedialog

import cv2
import numpy as np

RECURSIVE = True
BATCH = True
ALPHA = False
INIT_THRESH = 255
START = 0
STOP = -1

class VidCompile:
    """ Compiles an input video into one overlayed image """
    def __init__(self, path="") -> None:
        fmt_main = "%(asctime)s\t| %(levelname)s\t| VidCompile:\t%(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")

        # Set path
        self.filepath = path
        self.directory = path
        self.outputpath = path
        self.start = START
        self.stop = STOP
        self.thresh = INIT_THRESH
        self.skip_clip = False
        self.contour_bounds = 250

        # Prepare the outputs directory if it doesn't exist yet
        if not os.path.exists("./outputs"):
            os.mkdir("./outputs")

        logging.debug("Reading video...")
        if RECURSIVE:
            if self.filepath == "":
                self.filepath = filedialog.askdirectory()
            if self.filepath == "":
                logging.error("No directory specified! Exiting...")
                sys.exit(1)
            logging.info("Parsing directory: %s", self.filepath)
            if self.outputpath == "":
                self.outputpath = filedialog.askdirectory()
            if self.outputpath == "":
                logging.error("No output directory specified! Exiting...")
                sys.exit(1)
            logging.info("Output directory: %s", self.outputpath)
            for root,d_names,f_names in os.walk(self.filepath):
                for f in f_names:
                    if f.endswith(".avi") or f.endswith(".mp4"):
                        self.directory = root
                        logging.info("Current Video Diretory: %s", self.directory)
                        self.filename = os.path.basename(f)
                        logging.info("Current Video: %s", self.filename)
                        self.run()
                        self.skip_clip = False

        elif BATCH:
            if self.filepath == "":
                self.filepath = filedialog.askdirectory()
            if self.filepath == "":
                logging.error("No directory specified! Exiting...")
                sys.exit(1)
            logging.info("Parsing directory: %s", self.filepath)
            for file_name in os.listdir(self.filepath):
                if file_name.endswith(".avi") or file_name.endswith(".mp4"):
                    self.filename = os.path.basename(file_name)
                    logging.info("Current Video: %s", self.filename)
                    self.run()
                    self.skip_clip = False

        else:
            if self.filepath == "":
                self.filepath = filedialog.askopenfilename()
            if self.filepath == "":
                logging.error("No file specified! Exiting...")
                sys.exit(1)
            if (not self.filepath.endswith(".avi") and not self.filepath.endswith(".mp4")):
                logging.error("File must be a video! Exiting...")
                sys.exit(1)
            self.filename = os.path.basename(self.filepath)
            self.run()

    def run(self):
        """ Main runner per video """
        # Obtain frames to go into the overlayed image
        self.frame_arr = []
        self.alpha_arr = []
        self.read_video()

        # Determine Maximum Brightness Threshold
        if "bubble" in self.filename:   # If video has a bubble, prompt user to trim
            self.choose_thresh()
        cv2.destroyAllWindows()

        # Cancel video processing
        if self.skip_clip:
            return

        # Generate initial background images
        if ALPHA:
            self.alpha_output = np.zeros(self.frame_arr[0].shape, dtype=np.float64)
            self.alpha_output.fill(0)

        self.thresh_output = np.zeros(self.frame_arr[0].shape, dtype=np.uint8)
        self.thresh_output.fill(255)

        alpha = 1/(self.stop-self.start)

        start_time = datetime.datetime.utcnow()
        # logging.info("\tOverlay started at: %s", datetime.datetime.strftime(start_time, "%Y-%m-%d %H:%M:%S"))

        # Overlay each of the selected frames onto the output image
        for i, im in enumerate(self.frame_arr):
            if i < self.start:
                continue
            if i > self.stop:
                break

            if ALPHA:
                self.alpha_overlay(im, alpha)

            self.thresh_overlay(im)
            # logging.info("Frame %d/%d overlayed...", i-self.start, self.stop-self.start)
            # cv2.imshow("output", self.thresh_output)

            # cv2.waitKey(1)

        end_time = datetime.datetime.utcnow()
        # logging.info("\tOverlay finished at: %s", datetime.datetime.strftime(end_time, "%Y-%m-%d %H:%M:%S"))
        logging.info("\tOverlay took %s seconds", str(end_time-start_time))
        logging.info("\t%f Frames per second at %dx%d (%d Frames)",
                        len(self.frame_arr)/(end_time-start_time).total_seconds(),
                        len(self.frame_arr[0]),
                        len(self.frame_arr[0][0]),
                        len(self.frame_arr))

        # Display the final results and output to file
        logging.info("Finished! Writing to file...\n")
        pth = "./outputs/"
        if RECURSIVE:
            pth = self.outputpath
            subdir = self.directory
            head, tail = os.path.split(self.filepath)
            subdir = subdir.replace(head,'')
            pth += subdir + "/"
            pth = os.path.normpath(pth) + "/"
            if not os.path.exists(pth):
                os.makedirs(pth, exist_ok=True)

        elif BATCH:
            pth += os.path.basename(self.filepath) + "/"
            if not os.path.exists(pth):
                os.mkdir(pth)

        if ALPHA:
            cv2.imwrite(f"{pth}{self.filename[0:len(self.filename)-4]}-alpha.png",
                                            (np.rint(self.alpha_output)).astype(np.uint8))

        cv2.imwrite(f"{pth}{self.filename[0:len(self.filename)-4]}.png", self.thresh_output)
        cv2.destroyAllWindows()

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"({x},{y}) -> {self.current_frame[y,x]}")

    def read_video(self, start=0, stop=-1):
        """ Read in individual frames from the video

            Inputs:
            - start: int marking the first frame of the video to include
            - stop: int marking the last frame to include
            - step: int representing the amount of frames to skip in between includes
            Outputs:
            - None

            FIXME: This is the weak link after the other speed fix, how can this run faster?
        """
        if RECURSIVE:
            cap = cv2.VideoCapture(self.directory + "/" + self.filename)
        elif BATCH:
            cap = cv2.VideoCapture(self.filepath + "/" + self.filename)
        else:
            cap = cv2.VideoCapture(self.filepath)

        # Check if camera opened successfully
        if cap.isOpened() is False:
            logging.error("Error opening video stream or file")

        # Read until video is completed or stop is reached
        start_time = datetime.datetime.utcnow()
        # logging.info("\tFile Read started at: %s", datetime.datetime.strftime(start_time, "%Y-%m-%d %H:%M:%S"))
        c = 0

        while cap.isOpened():
            ret, frame = cap.read()     # Capture frame-by-frame

            if ret is True and (c <= stop or stop == -1):

                if c >= start:       # Skip frames that are less than start
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.frame_arr.append(frame)

            else:                   # Close the video after all frames have been read
                break

            c+=1

        end_time = datetime.datetime.utcnow()
        # logging.info("\tFile Read finished at: %s", datetime.datetime.strftime(end_time, "%Y-%m-%d %H:%M:%S"))
        logging.info("\tFile Read took %s seconds", str(end_time-start_time))
        logging.info("\t%f Frames per second at %dx%d (%d Frames)",
                        len(self.frame_arr)/(end_time-start_time).total_seconds(),
                        len(self.frame_arr[0]),
                        len(self.frame_arr[0][0]),
                        len(self.frame_arr))

        # Debug for image errors
        if len(self.frame_arr) == 0:
            logging.warning("Issue Reading Video...")
        else:
            self.start = 0
            self.stop = len(self.frame_arr) - 1
            logging.debug("Video length = %d frames", len(self.frame_arr))

        cap.release()

    def choose_thresh(self) -> None:
        """ Decide on what threshold to apply on the image
            Anything above the threshold will be considered background and ignored

            Inputs:
            - img: numpy image array, usually is the first frame of the video, but can be key frame
            Outputs:
            - None

            FIXME: This is disruptive for recursive runs, here are some options:
                - Add a boolean at the top to disable this for faster runs
                - Replace the old video with the new frame array after cropping and remove "bubble"
                    - 
        """
        index = 0
        logging.info("Index = %d/%d\t|\tThreshold = %d",
                        index,
                        len(self.frame_arr)-1,
                        self.thresh)
        logging.info("How to use:")
        logging.info("\t- 'esc' - skips the current video")
        logging.info("\t- 'enter' - accepts the current settings")
        logging.info("\t- 'space' - sets the current frame as the starting point")
        logging.info("\t- 'backspace' - sets the current frame as the ending point")
        logging.info("\t- 'left' - moves back one frame")
        logging.info("\t- 'right' - moves forward one frame")
        logging.info("\t- 'up' - increases the threshold")
        logging.info("\t- 'down' - decreases the threshold")
        logging.info("\t- 'left click' - click anywhere on the image to show that pixel's value")

        # Loop until the user confirms the threshold value from the previews
        while True:
            # Is the input image grayscale already? If not, convert it
            gry = self.frame_arr[index]
            color = cv2.cvtColor(gry, cv2.COLOR_GRAY2BGR)
            self.current_frame = gry

            # Generate thresholds
            ret, edit = cv2.threshold(gry,self.thresh,255,cv2.THRESH_TOZERO_INV)
            ret, binary = cv2.threshold(gry,self.thresh,255,cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(edit, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            for ctr in contours:
                if ctr[0,0,0] > self.contour_bounds:
                    cv2.drawContours(color, ctr, 0, (0,255,0), 5)

            # Show preview
            cv2.imshow("contour", color)
            cv2.imshow("image", edit)
            cv2.setMouseCallback('image', self.click_event)
            cv2.imshow("binary", binary)

            # Use arrow keys to adjust threshold, up/down are fine tuning, left/right are bigger
            Key = cv2.waitKeyEx()
            if Key == 2424832:          # Left arrow, previous frame
                index -= 1
            elif Key == 2621440:        # Down arrow, step down brightness
                self.thresh -= 1
            elif Key == 2490368:        # Up arrow, step up brightness
                self.thresh += 1
            elif Key == 2555904:        # Right arrow, next frame
                index += 1
            elif Key == 13:             # Enter key, accept current settings
                break
            elif Key == 27:             # Escape key, skip current output
                self.skip_clip = True
                logging.info("Skipping video...")
                break
            elif Key == 97:             # A key, decrement contour
                self.contour_bounds -= 1
            elif Key == 100:            # D key, increment contour
                self.contour_bounds += 1
            elif Key == 32:             # Space, set starting point
                self.start = index
                logging.info("New range: (%d-%d)", self.start, self.stop)
            elif Key == 8:              # Backspace, set stopping point
                self.stop = index
                logging.info("New range: (%d-%d)", self.start, self.stop)
            else:                       # Report unassigned key
                logging.warning("Invalid Key: %d", Key)

            # Enforce bounds and debug
            if index > len(self.frame_arr)-1:
                index = len(self.frame_arr)-1
            elif index < 0:
                index = 0
            if self.thresh > 255:
                self.thresh = 255
            elif self.thresh < 0:
                self.thresh = 0
            logging.info("%s\t|\tIndex = %d/%d\t| Threshold = %d\t| Contours = %d",
                                self.filename,
                                index,
                                len(self.frame_arr)-1,
                                self.thresh,
                                len(contours))

    def alpha_overlay(self, im, alpha):
        """ Overlay an image onto the background using alpha channel
            This average together all the pixels in the video for each individual spot
            Right now it is for grayscale images, but can be modified for color
        """
        self.alpha_output += im * alpha

    def thresh_overlay(self, im):
        """ Overlay an image onto the background by comparing pixels
            This chooses the darker pixel for each spot of the two images
            Right now it is for grayscale images, but can be modified for color
        """
        self.thresh_output = np.minimum(self.thresh_output, im)

if __name__ == "__main__":
    ov = VidCompile()
