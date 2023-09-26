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

from linedetector import LineDetector

RECURSIVE: bool = True  # Single file or whole directory?
MP: bool = False         # Use Multiprocessing?
OUTPUT_MODE: int = 0    # Where should output files go?
                        #       - 0: './outputs' with the python script
                        #       - 1: same as wherever the input videos came from
                        #       - 2: Pick location manually with a tkinter dialog
                        #       - NOTE: There is a maximum path length, if the file/directory names
                        #           are too long then it will fail to output the images without
                        #           any warning(especially mode 2, using G Drive)
INIT_THRESH: int = 255  # Background cutoff value (not used)

class Overlay:
    """ Compiles an input video into one overlayed image """
    def __init__(self) -> None:
        fmt_main = "%(asctime)s | %(levelname)s |\tOverlay:\t%(message)s"
        logging.basicConfig(format=fmt_main, level=logging.INFO,
                        datefmt="%Y-%m-%d %H:%M:%S")

        # Set path
        self.cur_dir: str = ''              # Placeholder for the filepath string iterator
        self.cur_vid: str = ''              # Placeholder for the filename string iterator
        self.inpath: str = './samples'      # Path of the root output directory
        self.outpath: str = './outputs'     # Path of the root output directory

        logging.debug("Reading video...")
        # Select the Directory or Video file to read
        if RECURSIVE:
            self.inpath = filedialog.askdirectory(title="Select Recursive Input Path")  # Pick input directory
            if self.inpath == "":
                logging.error("No directory specified! Exiting...")
                sys.exit(1)
            logging.info("Parsing directory: %s", self.inpath)
        else:
            # Acquire the path of a specific file
            self.inpath = filedialog.askopenfilename(
                title="Select Input Video",
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


        # Find the directory to write to
        if OUTPUT_MODE == 1:
            self.outpath, self.cur_dir = os.path.split(self.inpath)
        elif OUTPUT_MODE == 2:
            self.outpath = filedialog.askdirectory(title="Select Output Path")     # Pick output location
        if self.outpath == "":
            logging.error("No output directory specified! Exiting...")
            sys.exit(1)
        logging.info("Output directory: %s", self.outpath)

        if RECURSIVE:
            # Timing for performance diagnostics
            start_time = datetime.datetime.utcnow()
            tot_frames = 0

            # Perform the operations
            for root,d_names,f_names in os.walk(self.inpath):
                for f in f_names:
                    if f.endswith((".avi", ".mp4")):
                        self.cur_dir = root.replace(self.inpath, '')
                        logging.info("Current Video Directory:\t%s", self.cur_dir)
                        self.cur_vid = os.path.basename(f)
                        logging.info("Current Video:\t\t\t%s", self.cur_vid)
                        tot_frames += self.run()

            # Timing for the whole input set
            end_time = datetime.datetime.utcnow()
            run_time: float = (end_time-start_time).total_seconds()
            logging.info("The whole directory took %f seconds", run_time)
            logging.info("%f Total FPS", tot_frames/run_time)
        else:
            self.run()

    def run(self) -> int:
        """ Main runner per video """
        # Open Camera and check for success
        fname: str = os.path.normpath(self.inpath + self.cur_dir + "/" + self.cur_vid)
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

        start: int = 0
        stop: int = fcnt
        logging.info("\tVideo contains %d frames at (%dx%d) resolution and %d fps", fcnt, width, height, fps)

        # If video has a bubble, prompt user to trim until they confirm
        if "bubble" in fname:
            fname = self.edit_vid(cap, fname)
            if "fixed" not in fname:
                return 0

        cap.release()

        # Timing for performance diagnostics
        start_time = datetime.datetime.utcnow()
        frame_queue = []

        # Choose between multiprocessing and single process
        if MP:
            num_processes = mp.cpu_count()              # Number of processes based on cores
            logging.info("\tLaunching Multiprocessing with %d Cores", num_processes)
            with mp.Pool(num_processes) as p:

                # try:
                # Only one parameter can be passed to a pool map, expand it by packing into a tuple
                params = [(fname, x, start, stop) for x in range(num_processes)]
                frame_queue = p.map(process_video, params) # blocking until finished
                # except KeyboardInterrupt:
                #     p.terminate()
        else:
            logging.info("\tLaunching Single Process")
            frame_queue.append(process_video([fname, 0, start, stop]))

        if frame_queue:             # Make sure that an image was actually returned
            final_output = frame_queue.pop()        # Initial Frame
            logging.info("Showing the Multiprocessing output...")
            while frame_queue:
                new_frame = frame_queue.pop()       # Frames from other processes (if used)
                # FIXME: Cores 0-2 produce blank white images, Core 3 Produces a 4th of the image
                # logging.info("Showing the frame from a core")
                # cv2.imshow("mp", new_frame)
                # cv2.waitKey(0)
                final_output = process_frame(final_output, new_frame)

            # Display the final results and output to file
            pth = self.outpath
            if RECURSIVE:                   # Output file structure must match source
                subdir = os.path.basename(self.inpath) + self.cur_dir
                pth = os.path.normpath(self.outpath + "/" + subdir)

            # Prepare the outputs directory if it doesn't exist yet
            os.makedirs(pth, exist_ok=True)

            outname = self.cur_vid.split(".")[0]
            outpath = f"{pth}/{outname}.png"
            logging.info("Finished! Writing to file:\t%s", outpath)
            cv2.imwrite(outpath, final_output)

            if "bifurcation" not in outname:
                ld_start_time = datetime.datetime.utcnow()

                # Run the Line Detection algorithm (Comment out to speed up)
                LineDetector(dest=pth, fname=outname, img=final_output)

                ld_end_time = datetime.datetime.utcnow()
                run_time: float = (ld_end_time-ld_start_time).total_seconds()
                logging.info("Line Detect took %f seconds", run_time)
        else:
            logging.warning("Output File empty")

        end_time = datetime.datetime.utcnow()
        run_time: float = (end_time-start_time).total_seconds()
        logging.info("Processing took %f seconds", run_time)
        logging.info("%f Frames processed per second\n", fcnt/run_time)

        return fcnt

    def edit_vid(self, cap, fpath: str): #-> tuple[int, int, bool]:
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
        fcnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fname = os.path.basename(fpath)

        cv2.namedWindow(fname, cv2.WINDOW_FULLSCREEN)
        cv2.namedWindow('edit', cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(fname, cv2.WND_PROP_TOPMOST, 1)

        ret, frame = cap.read()         # Capture frame-by-frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow(fname, frame)

        logging.info("How to use:")
        logging.info("\t- 'esc' - skips the current video")
        logging.info("\t- 'enter' - accepts the current settings")
        logging.info("\t- 'space' - sets the current frame as the starting point")
        logging.info("\t- 'backspace' - sets the current frame as the ending point")
        logging.info("\t- 'left' - moves back one frame")
        logging.info("\t- 'right' - moves forward one frame")
        logging.info("\t- 'left click' - click anywhere on the image to show that pixel's value")
        logging.info("Editing video with %d frames...", fcnt)

        index: int = 0
        start: int = 0
        stop: int = fcnt
        while True:
            # Use arrow keys to move left or right
            # Use space/backspace to set start/end
            # Use enter to accept and esc to cancel
            Key = cv2.waitKeyEx(1)
            if Key != -1:
                if Key == 2424832:          # Left arrow, previous frame
                    if index > 0:               # Enforce min bounds
                        index -= 1
                    logging.info("Index = %d/%d", index, fcnt-1)

                    cap.set(cv2.CAP_PROP_POS_FRAMES, index-1)   # Update Image
                    ret, frame = cap.read()         # Capture frame-by-frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    cv2.imshow(fname, frame)

                elif Key == 2555904:        # Right arrow, next frame
                    if index < fcnt-1:     # Enforce max bounds
                        index += 1
                    logging.info("Index = %d/%d", index, fcnt-1)

                    cap.set(cv2.CAP_PROP_POS_FRAMES, index-1)   # Update Image
                    ret, frame = cap.read()             # Capture frame-by-frame
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    cv2.imshow(fname, frame)

                elif Key == 13:             # Enter key, accept current settings
                    # Reset video location
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

                    # Output the trimmed bubble video
                    width, height, fps = (
                        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                        cap.get(cv2.CAP_PROP_FPS),
                    )
                    fixed_fname: str = fpath.replace("bubble", "fixed")
                    out = cv2.VideoWriter(fixed_fname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
                    for _ in range(stop-start):
                        out.write(cap.read()[1])
                    out.release()

                    # Remove the old video and exit
                    cap.release()
                    os.remove(fpath)
                    cv2.destroyAllWindows()
                    return fixed_fname

                elif Key == 27:             # Escape key, skip current output
                    logging.info("Skipping the current bubble clip...")
                    cap.release()
                    cv2.destroyAllWindows()
                    return fname

                elif Key == 32:             # Space, set starting point
                    start = index
                    logging.info("New range: (%d-%d)", start, stop)

                elif Key == 8:              # Backspace, set stopping point
                    stop = index
                    logging.info("New range: (%d-%d)", start, stop)

                else:                       # Report unassigned key
                    logging.warning("Invalid Key: %d", Key)

def process_video(params):
    """ Read in individual frames from the video

        Inputs:
        - None
        Outputs:
        - None

        FIXME: This is the weak link after the other speed fix, how can this run faster?
    """
    fname: str = params[0]
    group_number: int = params[1]
    start: int = params[2]
    stop: int = params[3]

    # Open Camera and check for success
    cap = cv2.VideoCapture(fname)
    if cap.isOpened() is False:
        logging.error("\tError opening video stream or file")
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
            frame_jump_unit: int = (stop - start) // mp.cpu_count()
            start_frame: int = (group_number * frame_jump_unit) + start
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
                # TODO: Why are Cores 0-2 blank
                # logging.info(f"Frame={c}")
                # cv2.imshow("core", frame)
                # cv2.waitKey(0)

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
    ov = Overlay()
