""" overlay.py
"""
import datetime
import logging
import multiprocessing as mp
import os
import sys
from tkinter import filedialog

import cv2
import numpy as np

RECURSIVE = True
INIT_THRESH = 255

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
        self.thresh = INIT_THRESH
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
                # self.outputpath = filedialog.askdirectory()
                self.outputpath = "./outputs"
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
        self.single_process()
        # self.multi_process()

    def single_process(self) -> None:
        self.process_video(1)

    # def multi_process(self):
    #     num_processes = mp.cpu_count()
    #     frame_jump_unit = f
    #     p = mp.Pool(self.process_video, range(num_processes))

    def process_video(self, group_number):
        """ Read in individual frames from the video

            Inputs:
            - None
            Outputs:
            - None

            FIXME: This is the weak link after the other speed fix, how can this run faster?
                - Move output here and get rid of frame_arr?
                - Multiprocessing?
        """
        start_time = datetime.datetime.utcnow()

        if RECURSIVE:
            cap = cv2.VideoCapture(self.directory + "/" + self.filename)
        else:
            cap = cv2.VideoCapture(self.filepath)

        # Check if camera opened successfully
        if cap.isOpened() is False:
            logging.error("Error opening video stream or file")

        # If successful, set parameters
        # cap.set(cv2.CAP_PROP_MODE, cv2.CAP_MODE_GRAY)

        # Read variables from video file
        width, height, fcnt, fps = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FPS))
        )
        self.start: int = 0
        self.stop: int = fcnt - 1
        logging.info("\tVideo contains %d frames at %dx%d resolution and %d fps", fcnt, width, height, fps)

        # Initialize the starting output background
        self.output = np.zeros((height, width), dtype=np.uint8)
        self.output.fill(255)

        try:
            # Read until video is completed or stop is reached
            c: int = 0              # Iterater through video frames
            edit_status: int = -1    # If the video has a bubble, this determines editing state
            while cap.isOpened():

                # Skip frames that are outside of the set bounds
                if c < fcnt and c >= 0:
                    # frameTime = 1000.0 * c / fps
                    cap.set(cv2.CAP_PROP_POS_FRAMES, c)
                    # cap.set(cv2.CAP_PROP_POS_MSEC, frameTime)
                    # cur_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                    # if c != cur_pos:
                    #     print("Current cursor:\t", cur_pos, "\t|\tNew cursor:\t", c)

                    # Do image reading
                    ret, frame = cap.read()         # Capture frame-by-frame
                    if ret is True:                     # Check if read is successful
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                        # If video has a bubble, prompt user to trim until they confirm
                        if "bubble" in self.filename:
                            if edit_status <= 0:
                                c, edit_status  = self.edit_vid(frame, fcnt, c, edit_status)
                                ret = True
                                cv2.waitKey(1)
                            if edit_status == 1:
                                if c < self.start:
                                    continue
                                if c > self.stop:
                                    break
                                self.process_frame(frame)
                                c+=1
                                # TODO: save to new video and delete old one
                                cv2.destroyAllWindows()
                            elif edit_status == 2:               # If the user elects to ignore the clip, move on
                                logging.warning("Skipping the current bubble clip...")
                                cv2.destroyAllWindows()
                                return

                        else:
                            # Do image processing
                            self.process_frame(frame)
                            c+=1
                    else:                   # Close the video after all frames have been read
                        cap.release()
                        break

        except KeyboardInterrupt:
            cap.release()
            sys.exit(1)
        
        cap.release()

        end_time = datetime.datetime.utcnow()
        run_time: float = (end_time-start_time).total_seconds()
        logging.info("\tProcessing took %f seconds", run_time)
        logging.info("\t%f Frames processed per second", fcnt/run_time)

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
        cv2.imwrite(f"{pth}{self.filename[0:len(self.filename)-4]}.png", self.output)

    def process_frame(self, im):
        """ Overlay an image onto the background by comparing pixels
            This chooses the darker pixel for each spot of the two images
            Right now it is for grayscale images, but can be modified for color
        """
        self.output = np.minimum(self.output, im)

    def click_event(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"({x},{y}) -> {self.current_frame[y,x]}")

    def edit_vid(self, gry, fcnt, index, status) -> tuple[int, int]:
        """ Decide on what threshold to apply on the image
            Anything above the threshold will be considered background and ignored

            Inputs:
            - img: numpy image array, usually is the first frame of the video, but can be key frame
            Outputs:
            - None

            FIXME: This is disruptive for recursive runs, here are some options:
                - Add a boolean at the top to disable this for faster/smoother runs
                - Replace the old video with the new frame array after cropping and remove "bubble" from name
                    - This would make bubbles a one time fix, but would modify the original file (could be good or bad)
        """
        if status == -1:                # Only print this the first time for each video edit
            # def nothing(x):
            #     pass

            # cv2.namedWindow('image', cv2.WINDOW_FULLSCREEN)
            # cv2.setMouseCallback('image', self.click_event)
            # cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)
            # cv2.createTrackbar('color_track', 'image', INIT_THRESH, 255, nothing)

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
            logging.info("Editing video with %d frames...", fcnt)

        # Converts grayscale to bgr for contours and sets self.current_frame for the click event
        # (These are not currently being used and could technically be removed)
        color = cv2.cvtColor(gry, cv2.COLOR_GRAY2BGR)
        self.current_frame = gry

        # Generate thresholds
        ret, edit = cv2.threshold(gry,self.thresh,255,cv2.THRESH_TOZERO_INV)
        ret, binary = cv2.threshold(gry,self.thresh,255,cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(edit, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        for ctr in contours:
            if ctr[0,0,0] > self.contour_bounds:
                cv2.drawContours(color, ctr, 0, (0,255,0), 5)

        # Show previews
        cv2.imshow("contour", color)
        cv2.imshow("binary", binary)
        cv2.imshow("image", edit)

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
            return index, 1
        elif Key == 27:             # Escape key, skip current output
            logging.info("Skipping video...")
            return index, 2
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
        elif Key == 102:            # F key, print filename
            logging.info("File:\t%s", self.filename)
        else:                       # Report unassigned key
            logging.warning("Invalid Key: %d", Key)

        # self.thresh = cv2.getTrackbarPos('color_track', 'image')

        # Enforce bounds and debug
        # FIXME: the fcnt-2 below prevents the user from accessing the last frame, because when
        #       they do, the process_video sees that it's the last frame and exits without saving
        #       any of the frames and writes a white image. Everything else still works, and the
        #       last frame isn't skipped in processing, just editing
        if index > fcnt-2:
            index = fcnt-2
        elif index < 0:
            index = 0
        if self.thresh > 255:
            self.thresh = 255
        elif self.thresh < 0:
            self.thresh = 0
        logging.info("Index = %d/%d\t| Threshold = %d\t| Contours = %d",
                            index,
                            fcnt-1,
                            self.thresh,
                            len(contours))

        return index, 0

    # def alpha_overlay(self, im, alpha):
    #     """ Overlay an image onto the background using alpha channel
    #         This average together all the pixels in the video for each individual spot
    #         Right now it is for grayscale images, but can be modified for color
    #     """
    #     alpha = 1/(self.stop-self.start)
    #     self.alpha_output += im * alpha

if __name__ == "__main__":
    ov = VidCompile()
