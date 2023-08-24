
def preprocess(img, methods=[]):
    img = np.asarray(img * 255, dtype='uint8')
    org_img = img_quantize(img, 8)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # for ch in range(3):
    #     img = cv2.GaussianBlur(org_img[0], (5, 5), 0)
    #
    #     img = cv2.Canny(img, 254, 255, apertureSize=5)
    #
    #     kernelSize = 5
    #     kernel = np.ones((kernelSize, kernelSize), np.uint8)
    #
    #     iterations = 1
    #     img = 255 - cv2.dilate(img, kernel, iterations=iterations)
    #     org_img[ch] = img
    org_img = simplify_image_with_hough(org_img)

    return np.asarray(org_img / 255).astype(np.float32)


def img_quantize(img, n_colors=8):
    z = img.reshape((-1, 3))

    # convert to np.float32
    z = np.float32(z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(z, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    return res.reshape((img.shape))


def simplify_image_with_hough(image, animate=True):

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve line detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection using Canny
    canny_img = cv2.Canny(blurred, 254, 255, apertureSize=5)

    kernelSize = 5
    kernel = np.ones((kernelSize, kernelSize), np.uint8)

    iterations = 1
    dilate_imgs = 255 - cv2.dilate(canny_img, kernel, iterations=iterations)


    tested_angles = np.arange(-90,89.5,0.5)
    num_peaks = 10
    hough_thresh = 0.05
    fill_gap_val = 200
    min_length_val = 1000

    # h, t, r = hough_line(dilate_imgs[1], theta=tested_angles)
    # p, angles, dists = hough_line_peaks(h, t, r,num_peaks=num_peaks,threshold=math.ceil(hough_thresh*max(np.ndarray.flatten(h))))
    # lines_img_sim = probabilistic_hough_line(dilate_imgs[1], threshold=math.ceil(hough_thresh*max(np.ndarray.flatten(h))), line_length=min_length_val, line_gap=fill_gap_val, theta=t)

    # Apply Hough Transform to detect lines
    lines = cv2.HoughLinesP(dilate_imgs, rho=1, theta=np.pi / 180, threshold=50, minLineLength=100, maxLineGap=5)

    # Create a blank canvas to draw the lines on
    line_image = np.zeros_like(image)

    # Draw the detected lines on the blank canvas
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # You can adjust the color and line thickness

    # Combine the original image with the detected lines
    return cv2.addWeighted(image, 0, line_image, 1, 1)
