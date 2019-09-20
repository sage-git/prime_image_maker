import sys
import argparse
import subprocess
import itertools
import numpy as np
import cv2

def show_image(img, winname="test"):
    cv2.imshow(winname, img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def array_to_str(arr):
    # arr: array of 0-9 integers
    ret = np.array2string(arr.flatten(), max_line_width=arr.size+2, threshold=arr.size, separator="")[1:-1]
    return ret

def quantize_img_LUT(img, ncolor):
    img_p = None
    img_g = img
    if len(img.shape) == 3:
        img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_bin = img_g.max()
    min_bin = img_g.min()
    n = min(ncolor, max_bin - min_bin + 1)
    bins = np.linspace(min_bin, max_bin + 1, n + 1)
    y = np.array([bins[i - 1] for i in np.digitize(np.arange(256), bins)]).astype(int)
    img_p = np.array(cv2.LUT(img_g, y), dtype=np.uint8)
    return img_p

def quantize_img_kmeans(img, ncolor):
    h, w, c = img.shape
    Z = np.float32(img.reshape((-1, 3)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)
    ret, label, center = cv2.kmeans(Z, ncolor, None, crit, 10, cv2.KMEANS_RANDOM_CENTERS)
    return label.reshape((h, w))

def check_prime(string):
    cmdret = subprocess.run(["openssl", "prime", string], stdout=subprocess.PIPE)
    # capture_output = True: since python 3.7
    return "is prime" in cmdret.stdout.decode()

def make_image(base_prime_image, color_table):
    height, width = base_prime_image.shape
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.0
    font_thickness = 1
    ch_color_bright = (200, 200, 200)
    ch_color_dark = (55, 55, 55)
    color_thresh = 128
    (char_height, char_width), baseline = cv2.getTextSize("0", font, font_scale, font_thickness)
    margin_width = char_width//2
    margin_height = char_height//2
    panel_width = char_width + margin_width*2
    panel_height = char_height + margin_height*2
    img = np.ones((panel_height*height, panel_width*width, 3), dtype=np.uint8)*255
    for ih, line in enumerate(base_prime_image):
        y0 = ih*panel_height
        y1 = y0 + panel_height
        for iw, pixel in enumerate(line):
            pstr = "{}".format(pixel)
            x0 = iw*panel_width
            x1 = x0 + panel_width
            color = color_table[pixel]
            img[y0:y1, x0:x1, :] = np.array(color)
            if np.mean(color) > color_thresh:
                cv2.putText(img, pstr, (x0 + margin_width, y1 - margin_height), font, font_scale, ch_color_dark, font_thickness)
            else:
                cv2.putText(img, pstr, (x0 + margin_width, y1 - margin_height), font, font_scale, ch_color_bright, font_thickness)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("-W", "--width", type=int, default=37)
    parser.add_argument("-H", "--height", type=int, default=-1)
    parser.add_argument("-N", "--ncolor", type=int, default=10, choices=range(2,11))
    parser.add_argument("-q", "--quiet", action="store_true")

    args = parser.parse_args()

    image_file = args.filename
    width = args.width
    if width < 1:
        print("width should be a positive integer")
        return
    height = args.height
    if height < 1:
        height = width
    ncolor = args.ncolor
    quiet = args.quiet

    img_orig = cv2.imread(image_file)
    if not quiet:
       show_image(img_orig, "original")
    img_ico = cv2.resize(img_orig, (width, height))
    img = quantize_img_kmeans(img_ico, ncolor)
    pixels = list(set(img.flatten()))
    avgcolor = {}
    for ipix, pix in enumerate(pixels):
        avgcolor[ipix] = np.array(np.mean(img_ico[img == pix].reshape(-1, 3), axis=0), dtype=np.uint8)
    npix = len(pixels)
    if npix > 10:
        print("Too many colors")
        return
    idx_mat = np.zeros_like(img)
    for i, px in enumerate(pixels):
        idx_mat[img == px] = i
    lefttop = idx_mat[0, 0]
    rightbottom = idx_mat[-1, -1]
    last_img = np.array(idx_mat)
        
    for newpx in itertools.permutations(range(10), npix):
        if newpx[lefttop] == 0:
            continue
        if newpx[rightbottom] in (0, 2, 4, 5, 6, 8):
            continue
        for i in range(npix):
            last_img[idx_mat == i] = newpx[i]
        if not quiet:
           print("checking", newpx)
        numstring = array_to_str(last_img)
        if check_prime(numstring):
            print("found good permutation")
            print("----")
            print(numstring)
            print("----")
            for line in last_img:
                print(array_to_str(line))
            print("Avg color:")
            colortable = {}
            for iorig, inew in enumerate(newpx):
                colortable[inew] = tuple(int(c) for c in avgcolor[iorig])
            for i, c in colortable.items():
                print(i, c)
            pimg = make_image(last_img, colortable)
            show_image(pimg, "result")
            cv2.imwrite("prime_" + image_file, pimg)
            return
    print("No suitable prime number found")

if __name__ == "__main__":
    main()
