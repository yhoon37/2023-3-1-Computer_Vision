from PIL import Image
import math
import numpy as np

"""
Get and use the functions associated with gaussconvolve2d that you used in the last HW02.
"""
def gauss1d(sigma):
    #길이 구하기
    temp = math.ceil(sigma*6)
    length = temp if temp%2 == 1 else temp+1
    n = int(length/2)

    # [… -2, -1, 0, 1, 2, …] 를 활용해 gaussian 1d array 생성
    gaussian = np.array([np.exp(-x**2/(2*sigma**2)) for x in range(-n, n+1)])
    # normalizaition
    gauss1d_filter = gaussian/gaussian.sum()
    return gauss1d_filter

def gauss2d(sigma):
    # 1d filter 구하기
    gaussian1d = gauss1d(sigma)
    # np.outer 를 활용하여 2d filter 구하기
    gaussian2d = np.outer(gaussian1d, gaussian1d)
    # normalization
    gauss2d_filter = gaussian2d/gaussian2d.sum()
    return gauss2d_filter

def convolve2d(array, filter):
    # padding 사이즈 n 구하기
    n = int((filter.shape[0] - 1)/2)

    # numpy 함수를 이용하여 패딩 넣어주기
    padding_array = np.pad(array, ((n,n), (n,n)), "constant", constant_values = 0.)
    
    # Convolution -> Cross-correlation을 위해 filter 뒤집기
    flipped_filter = np.flip(np.flip(filter, axis=0), axis=1)
    
    # filter를 통과한 값을 저장하기 위한 array, 기존 이미지 array와 크기가 같음
    filtered_array = np.zeros((array.shape[0], array.shape[1]))
    filter_size = filter.shape[0]

    # 모든 element에 대헤
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            # Cross-correlation을 해준다  
            filtered_array[y, x] = np.sum(padding_array[y:y+filter_size, x:x+filter_size]*flipped_filter)
            
    
    return filtered_array

def gaussconvolve2d(array, sigma):
    # gaussian filter 얻기
    filter = gauss2d(sigma)
    # filter 적용
    filtered_array = convolve2d(array, filter)
    return filtered_array

def reduce_noise(img):
    """ Return the gray scale gaussian filtered image with sigma=1.6
    Args:
        img: RGB image. Numpy array of shape (H, W, 3).
    Returns:
        res: gray scale gaussian filtered image (H, W).
    """
    grayscale_img = img.convert("L")   # grayscale로 변경
    grayscale_array = np.asarray(grayscale_img)     # array 형식으로 변경

    filtered_array = gaussconvolve2d(grayscale_array, 1.6)  # filter 적용
    res = filtered_array.astype(np.float32) # np.float32 형식으로 변경

    filtered_img = Image.fromarray(res)     # array -> 이미지로 변경
    
    img.show()          # 원본 이미지
    filtered_img.show() # filtered 이미지

    return res

def sobel_filters(img):
    """ Returns gradient magnitude and direction of input img.
    Args:
        img: Grayscale image. Numpy array of shape (H, W).
    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction of gradient at each pixel in img.
            Numpy array of shape (H, W).
    Hints:
        - Use np.hypot and np.arctan2 to calculate square root and arctan
    """
    grayscale_array = np.asarray(img)   # img -> array
    X_filter = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]]) # Sobel X filter
    Y_filter = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]]) # Sobel Y filter
    
    X_gradient_array = convolve2d(grayscale_array, X_filter)    # X 축 방향 gradient
    Y_gradient_array = convolve2d(grayscale_array, Y_filter)    # Y 축 방향 gradient
    
    G = np.hypot(X_gradient_array, Y_gradient_array)        # X, Y gradient 합치기, sqrt(x1**2 + y1**2), element-wise
    theta = np.arctan2(Y_gradient_array, X_gradient_array)  # theta 값 구하기
    
    G = G * 255 / np.max(G) # gradient 값을 0~255 사이로 만들어 주기 위해

    X_gradient_img = Image.fromarray(X_gradient_array.astype('uint8'))  # 이미지로 변환
    Y_gradient_img = Image.fromarray(Y_gradient_array.astype('uint8'))  # 이미지로 변환
    G_img          = Image.fromarray(G.astype('uint8'))                 # 이미지로 변환
    theta_img      = Image.fromarray(theta.astype('uint8'))             # 이미지로 변환

    X_gradient_img.show()
    Y_gradient_img.show()
    G_img.show()
    theta_img.show()
    
    return (G, theta)

def non_max_suppression(G, theta):
    """ Performs non-maximum suppression.
    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).
    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).
    Returns:
        res: non-maxima suppressed image.
    """
    height = G.shape[0]
    width = G.shape[1]
    
    res = np.zeros((height, width))   # NMS 이미지 array
    
    # edge 영역은 제외하고 진행
    for y in range(1, height-1):
        for x in range(1, width-1):
            degree = theta[y][x] * 180/ np.pi    # radin -> degree 단위 변환
            
            case = math.floor((degree + 180 + 22.5) / 45) % 4    # case 구하기
            comp_num1 = 0
            comp_num2 = 0

            # 왼쪽 위부터 (0,0)이고 밑으로 갈 수록 y 값이 증가하고 오른쪽으로 갈수록 x 값이 증가함에 유의!!! 
            # case 1, case 3 헷갈리지 않게 조심하기
            if(case == 0):  # 좌, 우와 비교
                comp_num1 = G[y][x-1]
                comp_num2 = G[y][x+1]
            elif(case == 1):    # 오른쪽 위 대각선 방향과 비교
                comp_num1 = G[y-1][x+1]
                comp_num2 = G[y+1][x-1]
            elif(case == 2):    # 위, 아래와 비교
                comp_num1 = G[y-1][x]
                comp_num2 = G[y+1][x]
            elif(case == 3):    # 왼쪽 위 대각선 방향과 비교
                comp_num1 = G[y-1][x-1]
                comp_num2 = G[y+1][x+1]

            if((comp_num1 < G[y][x]) and (comp_num2 < G[y][x])):    # local-max 값이면
                res[y][x] = G[y][x]
            else:                                               # local-max 아닌 경우
                res[y][x] = 0
    
    return res

def double_thresholding(img):
    """ 
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: double_thresholded image.
    """    
    min = np.min(img)   # min 값
    max = np.max(img)   # max 값
    diff = max - min
    T_high = min + diff * 0.15
    T_low = min + diff * 0.03

    res = img.copy()
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            intensity = img[y][x]
            if(intensity < T_low):      # no edge, 0
                res[y][x] = 0
            elif(intensity < T_high):   # weak edge, 80
                res[y][x] = 80
            else:                       # strong edge, 255
                res[y][x] = 255

    return res

def dfs(img, res, i, j, visited=[]):
    # calling dfs on (i, j) coordinate imply that
    #   1. the (i, j) is strong edge
    #   2. the (i, j) is weak edge connected to a strong edge
    # In case 2, it meets the condition to be a strong edge
    # therefore, change the value of the (i, j) which is weak edge to 255 which is strong edge
    res[i, j] = 255

    # mark the visitation
    visited.append((i, j))

    # examine (i, j)'s 8 neighbors
    # call dfs recursively if there is a weak edge
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    """ Find weak edges connected to strong edges and link them.
    Iterate over each pixel in strong_edges and perform depth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
    Returns:
        res: hysteresised image.
    """
    width = img.shape[1]
    height = img.shape[0]

    res = np.zeros((height, width))
    visited = []    # dfs시 방문한 pixel의 위치 정보 저장

    for y in range(1, height-1):
        for x in range(1, width-1):
            visited.append((y, x))
            if(img[y, x] == 255):
                dfs(img, res, y, x, visited)
    
    return res

def main():
    RGB_img = Image.open('./iguana.bmp')

    noise_reduced_img = reduce_noise(RGB_img)
    Image.fromarray(noise_reduced_img.astype('uint8')).save('./iguana_blurred.bmp', 'BMP')
    
    g, theta = sobel_filters(noise_reduced_img)
    Image.fromarray(g.astype('uint8')).save('./iguana_sobel_gradient.bmp', 'BMP')
    Image.fromarray(theta.astype('uint8')).save('./iguana_sobel_theta.bmp', 'BMP')

    non_max_suppression_img = non_max_suppression(g, theta)
    Image.fromarray(non_max_suppression_img.astype('uint8')).save('./iguana_non_max_suppression.bmp', 'BMP')

    double_threshold_img = double_thresholding(non_max_suppression_img)
    Image.fromarray(double_threshold_img.astype('uint8')).save('./iguana_double_thresholding.bmp', 'BMP')

    hysteresis_img = hysteresis(double_threshold_img)
    Image.fromarray(hysteresis_img.astype('uint8')).save('./iguana_hysteresis.bmp', 'BMP')

main()