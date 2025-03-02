import cv2
import numpy as np

image = cv2.imread('coin.jpg')


#Function to count number of coins
def countCoins(contours):
    count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 5:
            count = count + 1
    return count


if image is None:
    print("Error: Could not open or find the image.")
else:
    width = 800
    height = int(image.shape[0] * (width / image.shape[1]))
    image = cv2.resize(image, (width, height))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale Input', image_gray)
    cv2.waitKey(0)
    cv2.imwrite("Output_Images_coin_detection/1_Grayscale_Input.jpg", image_gray)

    #Noise removal
    blurred_image = cv2.GaussianBlur(image_gray, (7, 7), 1)


    #Edge Detection with Sobel Filter
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    magnitude = np.uint8(np.absolute(magnitude))
    cv2.imshow('Sobel Edge Map', magnitude)
    cv2.waitKey(0)
    cv2.imwrite("Output_Images_coin_detection/2_Sobel_Edge_Map.jpg", magnitude)

    sobel_highlighted_image = cv2.add(image_gray, magnitude)
    cv2.imshow('Grayscale Input image with Sobel edges highlighted', sobel_highlighted_image)
    cv2.waitKey(0)
    cv2.imwrite("Output_Images_coin_detection/3_Grayscale_input_image_with_Sobel_edges_highlighted.jpg", sobel_highlighted_image)


    #Edge Detection with Laplacian Filter
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F)
    laplacian_edge_map = np.uint8(np.absolute(laplacian))
    cv2.imshow('Laplacian Edge Map', laplacian_edge_map)
    cv2.waitKey(0)
    cv2.imwrite("Output_Images_coin_detection/4_Laplacian_Edge_Map.jpg", laplacian_edge_map)

    laplacian_highlighted_image = cv2.add(image_gray, laplacian_edge_map)
    cv2.imshow('Input image with Laplacian edges highlighted', laplacian_highlighted_image)
    cv2.waitKey(0)
    cv2.imwrite("Output_Images_coin_detection/5_Grayscale_input_image_with_Laplacian_edges_highlighted.jpg", laplacian_highlighted_image)


    #Edge Detection with Cammy Edge Detector
    cammy_edge_map = cv2.Canny(blurred_image, 100, 200)
    cv2.imshow('Cammy Edge Map', cammy_edge_map)
    cv2.waitKey(0)
    cv2.imwrite("Output_Images_coin_detection/6_Cammy_Edge_Map.jpg", cammy_edge_map)

    cammy_highlighted_image = cv2.add(image_gray, cammy_edge_map)
    cv2.imshow('Input image with Cammy edges highlighted', cammy_highlighted_image)
    cv2.waitKey(0)
    cv2.imwrite("Output_Images_coin_detection/7_Grayscale_input_image_with_Cammy_edges_highlighted.jpg", cammy_highlighted_image)

    cv2.destroyAllWindows()


    #Region Based Segmentation
    #Cammy edge detector is not able to find full complete contours, so I used thresholding
    threshold, threshold_output = cv2.threshold(image_gray, 128, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('Image Thresholding at threshold = 128', threshold_output)
    cv2.waitKey(0)
    cv2.imwrite("Output_Images_coin_detection/8_Thresholding_Output_on_Grayscale_input_image.jpg", threshold_output)

    contours, _ = cv2.findContours(threshold_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for contour in contours:
        if cv2.contourArea(contour) > 5:
            i = i + 1
            mask = np.zeros_like(image_gray)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
            isolated_coin = cv2.bitwise_and(image_gray, image_gray, mask=mask)
            filename = str(i + 8) + "_Isolated_Coin_" + str(i) + ".jpg"
            cv2.imshow(filename, isolated_coin)
            cv2.waitKey(0)
            cv2.imwrite("Output_Images_coin_detection/" + filename, isolated_coin)

    cv2.destroyAllWindows()

    print(f"Total number of coins detected = {countCoins(contours)}")
