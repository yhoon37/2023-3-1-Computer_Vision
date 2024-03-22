from PIL import Image
import numpy as np
import math

# part 1-1
def boxfilter(n):
    # 짝수 예외 처리
    assert n%2 == 1, "dimension must be odd"
    # 값이 1/(n*n)이고 n x n 크기의 array return
    return np.full((n,n), 1/(n*n))

# part 1-2
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

# part 1-3
def gauss2d(sigma):
    # 1d filter 구하기
    gaussian1d = gauss1d(sigma)
    # np.outer 를 활용하여 2d filter 구하기
    gaussian2d = np.outer(gaussian1d, gaussian1d)
    # normalization
    gauss2d_filter = gaussian2d/gaussian2d.sum()
    return gauss2d_filter

# part 1-4 (a)
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

# part 1-4(b)
def gaussconvolve2d(array, sigma):
    # gaussian filter 얻기
    filter = gauss2d(sigma)
    # filter 적용
    filtered_array = convolve2d(array, filter)
    return filtered_array

# part 1-4(c,d)
def part1_4(img_name):
    img = Image.open(img_name)              # 이미지 불러오기
    grayscale_img = img.convert('L')        # grayscale로 변환
    img_array = np.asarray(grayscale_img)   # 이미지 -> array 로 변환

    filtered_array = gaussconvolve2d(img_array, 3)  # gaussian convolution
    filtered_array = filtered_array.astype('uint8') # float -> uint(0~255)로 바꾸기
    
    filtered_img = Image.fromarray(filtered_array)  # array -> 이미지 로 변환

    img.show()          # 원본 이미지
    filtered_img.show() # filter 적용 이미지
    
    #filtered_img.save(img_name[:-4]+"_gray_low.bpm", 'bmp')    # 저장
    return filtered_array

# part 2-1
def part2_1(img_name, sig):
    sigma = sig   # 적절한 sigma 값 설정 
    img = Image.open(img_name)   # 이미지를 불러온다

    # RGB channel을 분리한 후 filter 적용을 위해 array 형식으로 바꿔준다
    red, green, blue = map(np.asarray, img.split()) 

    # 각각의 channel에 convolution을 적용한 후, float 값을 uint(0~255) 값으로
    # 바꾸고 Image 형식으로 바꾼다
    red = Image.fromarray(gaussconvolve2d(red, sigma).astype('uint8'))
    green = Image.fromarray(gaussconvolve2d(green, sigma).astype('uint8'))
    blue = Image.fromarray(gaussconvolve2d(blue, sigma).astype('uint8'))

    # RGB 3개의 channel을 합쳐 하나의 이미지로 만든다
    low_freq_img = Image.merge('RGB', (red, green, blue))

    img.show()          # 원본 이미지
    low_freq_img.show() # RGB 사진에 filter를 적용한 이미지
    #low_freq_img.save(img_name[:-4]+"_low.bpm", 'bmp')

    low_freq_array = np.asarray(low_freq_img)   # return을 위한 low_freq_array
    
    return low_freq_array

# part 2-2
def part2_2(img_name, sigma, margin_on_off = True):
    original_img = Image.open(img_name)         # 이미지 오픈
    original_array = np.asarray(original_img)   # array 형식으로 바꾸기
    
    low_freq_array = part2_1(img_name, sigma)   # low-frequency array 얻기

    # high = original - low 인데 array의 값들이 uint8이므로 int16으로 바꾸어 음의 값도 나올 수 있게 한다
    high_freq_array = original_array.astype('int16') - low_freq_array.astype('int16')

    # 가장 낮은 intensity를 margin으로 설정한다
    # mragin을 넣어주지 않으면 음수 값이 존재하고 이를 uint8로 바꾸면 이미지가 이상해짐
    if(margin_on_off):
        margin = np.min(high_freq_array)

        high_freq_array = high_freq_array + margin  # margin 값을 넣어주어 0보다 크게 만들어 준다
    
        high_freq_array = high_freq_array.astype('uint8')   # uint 형으로 바꿔준다
        high_freq_img = Image.fromarray(high_freq_array)    # 이미지 로 바꿔준다

        high_freq_img.show()    # high-frequency 이미지
        #high_freq_img.save(img_name[:-4]+"_high.bpm", 'bmp')# 저장
        
    return high_freq_array

# part 2-3, 마진 안하고, 0~255 사이어야 하는데, 
def adjustValue(n):
    # 0보다 작은 값은 0으로 255보다 큰 값은 255로 바꾸기
    if n < 0:
        return 0
    elif n > 255:
        return 255
    else:
        return n

def part2_3(low_img, high_img, low_sigma, high_sigma):
    low_freq_array = part2_1(low_img, low_sigma)
    high_freq_array = part2_2(high_img, high_sigma, False)   # 마진 추가 하지 말기

    
    hybrid_array = low_freq_array + high_freq_array # 두 이미지 더하기

    vectorized_func = np.vectorize(adjustValue)     
    hybrid_array = vectorized_func(hybrid_array)    # 각 pixel의 intensity를 0~255 사이의 값으로 맞춰준다

    hybrid_array = hybrid_array.astype('uint8')     # Image로 바꾸기 위해 uint로 형 변환
    
    hybrid_img = Image.fromarray(hybrid_array)      # Image로 바꾸기
    hybrid_img.show()
    #hybrid_img.save("hybrid.bmp", 'bmp')    # 저장

    return hybrid_img

#part1_4('2a_mangosteen.bmp')
#part2_1('2a_mangosteen.bmp', 5)
#part2_2('2a_mangosteen.bmp', 2)
part2_3('3a_lion.bmp', '3b_tiger.bmp', 3, 2)
