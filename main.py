import math
import cv2
import numpy as np


def check(argv):
    ## [load-灰度化]
    default_file = './line_pictures/elevator_closed.png'
    filename = argv[0] if len(argv) > 0 else default_file

    # Loads an image
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded fine
    if src is None:
        print('Error opening image!')
        print('Usage: hough_lines.py [image_name -- default ' + default_file + '] \n')
        return -1
    ## [load-灰度化]

    ## [edge_detection]
    # Edge detection
    edge = cv2.Canny(src, 50, 200, None, 3)
    ## [edge_detection]
    cv2.imshow("edge", edge)

    ## [恢复BGR]
    src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    ## [恢复BGR]

    ## [边缘膨胀]
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edge, kernel, iterations=3)
    cv2.imshow("dilation", dilation)
    ## [边缘膨胀]

    ## [hough_lines_p]
    # Probabilistic Line Transform
    linesP = cv2.HoughLinesP(dilation, 1, np.pi / 180, 50, None, 500, 1)
    ## [hough_lines_p]

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            x1 = l[0]
            y1 = l[1]
            x2 = l[2]
            y2 = l[3]

            line_length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            try:
                line_radian = math.atan(math.fabs(y1 - y2) / math.fabs(x1 - x2))
            except ZeroDivisionError:
                line_radian = math.pi / 2.0
            line_angle = 180.0 * line_radian / math.pi

            if line_angle > 75.0:
                print(line_length, line_angle)
                cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    return src


def cropped(img):
    height = img.shape[0]
    width = img.shape[1]
    cropped_img = img[height // 2 - height // 5:height // 2 + height // 5,
                  width // 2 - width // 5:width // 2 + width // 5]  # 裁剪坐标为[y0:y1, x0:x1]
    # cv2.imwrite("./line_pictures/cv_cut_thor.jpg", cropped)
    return cropped_img


def harris(img):  # 角点提取
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 10, 3, 0.04)  # 生成估价矩阵
    # result is dilated for marking the corners, not important
    dst = cv2.dilate(dst, None)
    #Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img


if __name__ == "__main__":
    # check(sys.argv[1:])
    file1 = "./line_pictures/closed2.png"
    file2 = "./line_pictures/opened.png"

    img1 = check([file1])
    img2 = check([file2])

    cv2.imshow('opened', img1)
    cv2.imshow('closed', img2)

    # cv2.imshow("opened2", cropped(img1))
    # cv2.imshow("closed2", cropped(img2))

    # img1 = cv2.imread(file1)
    # img2 = cv2.imread(file2)

    # 创建ORB特征检测器
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 暴力匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    goodMatches = []

    print(len(kp1), len(kp2), len(matches))

    # 筛选出好的描述子
    matches = sorted(matches, key=lambda x: x.distance)
    for i in range(len(matches)):
        if matches[i].distance < 0.5 * matches[-1].distance:
            goodMatches.append(matches[i])

    print("good matches", len(goodMatches))

    result = cv2.drawMatches(img1, kp1, img2, kp2, goodMatches, None)
    cv2.imshow("orb-match", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
