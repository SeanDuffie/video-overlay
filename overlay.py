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

RECURSIVE: bool = False  # Single file or whole directory?
MP: bool = True         # Use Multiprocessing?
# INIT_THRESH: int = 255  # Background cutoff value

class VidCompile:
    """ Compiles an input video into one overlayed image """
    def __init__(self) -> None:
        fmt_main = "%(asctime)s\t| %(levelname)s\t| VidCompile:\t%(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")

        # Set path
        self.cur_dir: str = ''              # Placeholder for the filepath string iterator
        self.cur_vid: str = ''              # Placeholder for the filename string iterator
        self.inpath: str = './sample/'               # Path of the root output directory
        self.outpath: str = './outputs/'    # Path of the root output directory

        # Prepare the outputs directory if it doesn't exist yet
        os.makedirs("./outputs", exist_ok=True)

        logging.debug("Reading video...")
        if RECURSIVE:
            # Timing for performance diagnostics
            start_time = datetime.datetime.utcnow()
            tot_frames = 0

            # Find the directory to read from
            # self.inpath = filedialog.askdirectory()
            if self.inpath == "":
                logging.error("No directory specified! Exiting...")
                sys.exit(1)
            logging.info("Parsing directory: %s", self.inpath)

            # Find the directory to write to
            # self.outpath = filedialog.askdirectory()     # Pick output location
            if self.outpath == "":
                logging.error("No output directory specified! Exiting...")
                sys.exit(1)
            logging.info("Output directory: %s", self.outpath)
            for root,d_names,f_names in os.walk(self.inpath):
                for f in f_names:
                    if f.endswith(".avi") or f.endswith(".mp4"):
                        self.cur_dir = root.replace(self.inpath, '')
                        logging.info("Current Video Directory:   %s", self.cur_dir)
                        self.cur_vid = os.path.basename(f)
                        logging.info("Current Video:\t\t   %s", self.cur_vid)
                        tot_frames += self.run()

            # Timing for the whole input set
            end_time = datetime.datetime.utcnow()
            run_time: float = (end_time-start_time).total_seconds()
            logging.info("The whole directory took %f seconds", run_time)
            logging.info("%f Total FPS", tot_frames/run_time)

        else:
            # Acquire the path of a specific file
            self.inpath = filedialog.askopenfilename(
                title="Select Video",
                filetypes=[
                    ("Videos", "mp4"),
                    ("Videos", "avi"),
                    ("Videos", "flv"),
                ]
            )
            if self.inpath == "":
                logging.error("No file specified! Exiting...")
                sys.exit(1)

            # Separate into directory and filename, then confirm it's a video
            self.inpath, self.cur_vid = os.path.split(self.inpath)
            self.run()

    def run(self) -> int:
        """ Main runner per video """
        # Open Camera and check for success
        fname: str = self.inpath + self.cur_dir + "/" + self.cur_vid
        cap = cv2.VideoCapture(fname)
        if cap.isOpened() is False:
            logging.error("Error opening video stream or file. Exiting...")
            logging.error("Filename: %s", fname)
            sys.exit(1)

        # Read variables from video file
        width, height, fcnt, fps = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            int(cap.get(cv2.CAP_PROP_FPS))
        )
        cap.release()

        start: int = 0
        stop: int = fcnt
        logging.info("\tVideo contains %d frames at (%dx%d) resolution and %d fps", fcnt, width, height, fps)

        # # If video has a bubble, prompt user to trim until they confirm
        # if "bubble" in fname:
        #     start, stop = self.edit_vid(cap, fname)

        # Timing for performance diagnostics
        start_time = datetime.datetime.utcnow()
        frame_queue = []

        # Choose between multiprocessing and single process
        if MP:
            num_processes = mp.cpu_count()              # Number of processes based on cores
            logging.info("\tLaunching Multiprocessing with %d Cores", num_processes)
            p = mp.Pool(num_processes)

            # try:
            # Only one parameter can be passed to a pool map, expand it by packing into a tuple
            params = [(fname, x, start, stop) for x in range(num_processes)]
            frame_queue = p.map(process_video, params) # blocking until finished
            # except KeyboardInterrupt:
            #     p.terminate()
            p.close()
        else:
            logging.info("\tLaunching Single Process")
            frame_queue.append(process_video([fname, 0, start, stop]))

        if frame_queue:             # Make sure that an image was actually returned
            final_output = frame_queue.pop()        # Initial Frame
            while frame_queue:
                new_frame = frame_queue.pop()       # Frames from other processes (if used)
                final_output = process_frame(final_output, new_frame)

            # Display the final results and output to file
            pth = self.outpath
            if RECURSIVE:                   # Output file structure must match source
                subdir = os.path.basename(self.inpath) + self.cur_dir
                pth = os.path.normpath(self.outpath + subdir)
                os.makedirs(pth, exist_ok=True)

            outname = f"{pth}/{self.cur_vid[0:len(self.cur_vid)-4]}.png"
            logging.info("Finished! Writing to file: %s", pth)
            cv2.imwrite(outname, final_output)
        else:
            logging.warning("Output File empty")

        end_time = datetime.datetime.utcnow()
        run_time: float = (end_time-start_time).total_seconds()
        logging.info("Processing took %f seconds", run_time)
        logging.info("%f Frames processed per second\n", fcnt/run_time)

        return fcnt

    def edit_vid(self, cap, fcnt) -> tuple[int, int, int, int]:
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
        fname: int = vid_stats[0]
        fcnt: int = vid_stats[1]

        status: int = -1                       # If the video has a bubble, this determines editing state
        while status != 1:
            if status == 2:               # If the user elects to ignore the clip, move on
                logging.info("Skipping the current bubble clip...")
                cv2.destroyAllWindows()
                return
            if status == 1:
                i = 0
                while i < fcnt:
                    if i < start:
                        i+=1
                        continue
                    if i > stop:
                        break
                    i+=1
                # TODO: save to new video and delete old one
                cv2.destroyAllWindows()

            if status == -1:                # Only do this the first time for each video edit
                # def nothing(x):
                #     pass
                # def click_event(event, x, y, flags, params):
                #     if event == cv2.EVENT_LBUTTONDOWN:
                #         logging.info(f"({x},{y}) -> {gry[y,x]}")

                # cv2.namedWindow('image', cv2.WINDOW_FULLSCREEN)
                # cv2.setMouseCallback('image', click_event)
                # cv2.setWindowProperty('image', cv2.WND_PROP_TOPMOST, 1)
                # cv2.createTrackbar('color_track', 'image', self.thresh, 255, nothing)

                logging.info("How to use:")
                logging.info("\t- 'esc' - skips the current video")
                logging.info("\t- 'enter' - accepts the current settings")
                logging.info("\t- 'space' - sets the current frame as the starting point")
                logging.info("\t- 'backspace' - sets the current frame as the ending point")
                logging.info("\t- 'left' - moves back one frame")
                logging.info("\t- 'right' - moves forward one frame")
                # logging.info("\t- 'up' - increases the threshold")
                # logging.info("\t- 'down' - decreases the threshold")
                # logging.info("\t- 'left click' - click anywhere on the image to show that pixel's value")
                logging.info("Editing video with %d frames...", fcnt)

            cv2.waitKey(1)

        # Generate thresholds
        # ret, edit = cv2.threshold(gry,self.thresh,255,cv2.THRESH_TOZERO_INV)
        # ret, binary = cv2.threshold(gry,self.thresh,255,cv2.THRESH_BINARY)

        # Show previews
        # cv2.imshow("binary", binary)
        cv2.imshow("image", gry)

        # Enforce bounds and debug

        # Use arrow keys to adjust threshold, up/down are fine tuning, left/right are bigger
        Key = cv2.waitKeyEx(1)
        if Key != -1:
            if Key == 2424832:          # Left arrow, previous frame
                if index > 0:               # Enforce min bounds
                    index -= 1
                logging.info("Index = %d/%d", index, fcnt-1)
            elif Key == 2555904:        # Right arrow, next frame
                # FIXME: the fcnt-2 below prevents the user from accessing the last frame, because when
                #       they do, the process_video sees that it's the last frame and exits without saving
                #       any of the frames and writes a white image. Everything else still works, and the
                #       last frame isn't skipped in processing, just editing
                if index < fcnt-2:     # Enforce max bounds
                    index += 1
                logging.info("Index = %d/%d", index, fcnt-1)
            elif Key == 13:             # Enter key, accept current settings
                return index, start, stop, 1
            elif Key == 27:             # Escape key, skip current output
                logging.info("Skipping video...")
                return index, start, stop, 2
            elif Key == 32:             # Space, set starting point
                start = index
                logging.info("New range: (%d-%d)", start, stop)
            elif Key == 8:              # Backspace, set stopping point
                stop = index
                logging.info("New range: (%d-%d)", start, stop)
            else:                       # Report unassigned key
                logging.warning("Invalid Key: %d", Key)

        # self.thresh = cv2.getTrackbarPos('color_track', 'image')

        return index, start, stop, 0

def process_video(params):
    """ Read in individual frames from the video

        Inputs:
        - None
        Outputs:
        - None

        FIXME: This is the weak link after the other speed fix, how can this run faster?
    """
    fname = params[0]
    group_number: int = params[1]
    start = params[2]
    stop = params[3]

    # Open Camera and check for success
    cap = cv2.VideoCapture(fname)
    if cap.isOpened() is False:
        print("\tError opening video stream or file")
        sys.exit(1)

    try:
        # Read variables from video file
        width, height, fcnt = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        
        # Set up parameters based on core usage
        if MP:
            frame_jump_unit = (stop - start) // mp.cpu_count()
            start_frame = (group_number * frame_jump_unit) + start
            c: int = start_frame     # Iterater through video frames
        else:
            frame_jump_unit = stop - start
            start_frame = start
            c = 0

        # Initialize the starting output background
        output = np.zeros((height, width), dtype=np.uint8)
        output.fill(255)

        c = start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while c < frame_jump_unit:  # Read until video is completed or stop is reached
            # Skip frames that are outside of the set bounds

            # Do image reading
            ret, frame = cap.read()         # Capture frame-by-frame
            if ret is True:                     # Check if read is successful
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Do image processing
                output = process_frame(output, frame)
                c+=1
            else:                   # Close the video after all frames have been read
                break

    except KeyboardInterrupt:
        cap.release()
        sys.exit(1)

    cap.release()
    return output

def process_frame(im, output):
    """ Overlay an image onto the background by comparing pixels
        This chooses the darker pixel for each spot of the two images
        Right now it is for grayscale images, but can be modified for color
    """
    return np.minimum(output, im)

if __name__ == "__main__":
    ov = VidCompile()
