""" overlay.py
"""
import datetime
import logging
import multiprocessing as mp
import os
import sys
import time
from tkinter import filedialog

import cv2
import numpy as np

from linedetector import LineDetector

RECURSIVE: bool = True  # Single file or whole directory?
MP: bool = True         # Use Multiprocessing?
OUTPUT_MODE: int = 0    # Where should output files go?
                        #       - 0: './outputs' with the python script
                        #       - 1: same as wherever the input videos came from
                        #       - 2: Pick location manually with a tkinter dialog
                        #       - NOTE: There is a maximum path length, if the file/directory names
                        #           are too long then it will fail to output the images without
                        #           any warning(especially mode 2, using G Drive)
MP_MODE: int = 2        # How to handle multithreading?
                        #       - 0: Proven method of mp.Pool
                        #       - 1: Proven method of mp.Pool
                        #       - 2: Worker processes
                        #       - 3: Live video
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
            if MP_MODE != 3:
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
                    if f.endswith(".avi") or f.endswith(".mp4"):
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
        if MP_MODE == 3:
            cap = cv2.VideoCapture(0)        # Capture video from camera
        else:
            cap = cv2.VideoCapture(fname)   # Capture video from file

        # Confirm video opened properly
        if cap.isOpened() is False:
            logging.error("Error opening video stream or file. Exiting...")
            logging.error("Filename: %s", fname)
            sys.exit(1)

        # Read variables from video file
        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        start: int = 0
        stop: int = 0
        if MP_MODE == 3:
            fcnt: int = 0
            logging.info("\tLive Video running at (%dx%d) resolution", width, height)
        else:
            fcnt, fps = (
                int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(cap.get(cv2.CAP_PROP_FPS)),
            )
            logging.info("\tVideo contains %d frames at (%dx%d) resolution and %d fps", fcnt, width, height, fps)

            # If video has a bubble, prompt user to trim until they confirm
            if "bubble" in fname:
                fname = self.edit_vid(cap, os.path.basename(fname))
                if "fixed" not in fname:
                    return 0

                # Close and reopen the video file after an edit
                cap.release()
                cap = cv2.VideoCapture(fname)   # Capture video from file
                if cap.isOpened() is False:
                    logging.error("Error opening the edited video file. Exiting...")
                    logging.error("Filename: %s", fname)
                    sys.exit(1)
                fcnt, fps = (
                    int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    int(cap.get(cv2.CAP_PROP_FPS)),
                )
                logging.info("\tEdited Video contains %d frames at (%dx%d) resolution and %d fps", fcnt, width, height, fps)

            # Timing for performance diagnostics
            start_time = datetime.datetime.utcnow()

            # Initialize and populate Shared Variables
            logging.info("Starting shared variables")
            np_arr_shape = (fcnt, height, width)
            mp_array = mp.Array("I", int(np.prod(np_arr_shape)), lock=mp.Lock())
            np_array = np.frombuffer(mp_array.get_obj(), dtype="I").reshape(np_arr_shape)
            shared_memory = (mp_array, np_array)
            out_queue = mp.Queue()

            # Choose between multiprocessing and single process
            if MP_MODE == 0:
                logging.info("\tLaunching Single Process")
                process_video(mp_array, out_queue, 0, fcnt)

            elif MP_MODE == 1:
                num_processes = mp.cpu_count()              # Number of processes based on cores

                # Set up the Processes Pool
                logging.info("\tLaunching Multiprocessing Pool with %d Cores", num_processes)
                # Pool
                with mp.Pool(num_processes) as p:
                    # Only one parameter can be passed to a pool map, expand it by packing into a tuple
                    # params = [(fname, x, start, stop) for x in range(num_processes)]
                    params = [(shared_memory, out_queue, x, fcnt) for x in range(num_processes)]

                    # Launch the map of pool processes, the out_queue returned is a list of outputs that need to be recombined
                    p.map(pool_process_video, params) # blocking until finished
            else:
                num_processes = mp.cpu_count() - 1          # Number of processes based on cores
                # Set up the Processes Workers
                logging.info("\tLaunching Multiprocessing Workers with %d Cores", num_processes)

                # Worker
                processes = [mp.Process(target=process_video, args=(shared_memory, out_queue, x, fcnt)) for x in range(num_processes)]

                # Load inqueue with values
                for x in range(fcnt):
                    logging.info("Acquire: %d", x)
                    # mp_array.acquire()
                    ret, frame = cap.read()
                    if ret:
                        np_array[x] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Launch Processes
                for proc in processes:
                    proc.start()

            out_queue.put(None)
            partial_output = out_queue.get()
            final_output = partial_output
            while partial_output is not None:           # Make sure that an image was actually returned
                final_output = process_frame(final_output, partial_output)
                partial_output = out_queue.get()       # Frames from other processes (if used)

            if final_output is not None:
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
                    # Run the Line Detection algorithm (Comment out for speed)
                    LD = LineDetector(dest=pth, fname=outname, img=final_output)
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
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        int(cap.get(cv2.CAP_PROP_FPS)),
                    )
                    fixed_fname: str = fpath.replace("bubble", "fixed")
                    out = cv2.VideoWriter(fixed_fname, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
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

def pool_process_video(params):
    process_video(params[0], params[1], params[2], params[3])

def process_video(shared_memory, out_queue, group_number, fcnt):
    """ Read in individual frames from the video

        Inputs:
        - None
        Outputs:
        - None

        FIXME: This is the weak link after the other speed fix, how can this run faster?
    """
    print("Starting process")
    mp_array, np_array = shared_memory
    # Set up parameters based on core usage
    if MP:
        frame_jump_unit: int = fcnt // mp.cpu_count()
        c: int = group_number * frame_jump_unit     # Iterater through video frames
    else:
        frame_jump_unit = len(mp_array)
        c = 0

    # Initialize the starting output background
    frame = np_array[c]
    output = np.zeros(frame.shape, dtype=np.uint8)
    output.fill(255)
    while c < frame_jump_unit:  # Read until video is completed or stop is reached
        # Do image reading
        frame = np_array[c]         # Capture frame-by-frame
        output = process_frame(output, frame)
        c += 1

    if output is not None:
        out_queue.put(output)

    while True:
        try:
            mp_array.release()
            break
        except ValueError:
            time.sleep(0.001)

def process_frame(im, output):
    """ Overlay an image onto the background by comparing pixels
        This chooses the darker pixel for each spot of the two images
        Right now it is for grayscale images, but can be modified for color
    """
    return np.minimum(output, im)

if __name__ == "__main__":
    ov = Overlay()
