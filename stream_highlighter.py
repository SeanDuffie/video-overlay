logging.info("Index = %d/%d\t|\tThreshold = %d", index, len(self.frame_arr)-1, self.thresh)
logging.info("How to use:")
logging.info("\t- 'esc' - skips the current video")
logging.info("\t- 'enter' - accepts the current settings")
logging.info("\t- 'space' - sets the current frame as the starting point")
logging.info("\t- 'backspace' - sets the current frame as the ending point")
logging.info("\t- 'left' - moves back one frame")
logging.info("\t- 'right' - moves forward one frame")
logging.info("\t- 'up' - increases the threshold")
logging.info("\t- 'down' - decreases the threshold")

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
        # print(ctr[0,0,1])
        # Key = cv2.waitKeyEx()
        if ctr[0,0,0] > 360:
            cv2.drawContours(color, ctr, 0, (0,255,0), 3)

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
    elif Key == 13:
        break
    elif Key == 27:
        self.skip_clip = True
        logging.info("Skipping video...")
        break
    elif Key == 32:
        self.start = index
        logging.info("New range: (%d-%d)", self.start, self.stop)
    elif Key == 8:
        self.stop = index
        logging.info("New range: (%d-%d)", self.start, self.stop)
    else:
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
    logging.info("Index = %d/%d\t| Threshold = %d\t| Contours = %d",
                        index, len(self.frame_arr)-1, self.thresh, len(contours))