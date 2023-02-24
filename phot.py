import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path = "phot_img/RSDD/Type2RSDDsdataset/Railsurfaceimages/rail_1.jpg"
# img_path = "phot_img/ciwa/MT_Blowhole/Imgs/exp1_num_327970.jpg"
# img_path = "phot_img/ciwa/MT_Crack/Imgs/exp1_num_3191.jpg"
# img_path = "phot_img/RSDD/Type2RSDDsdataset/Railsurfaceimages/rail_69.jpg"
# img_path = "phot_img/image.png"
ori_img = cv2.imread(img_path, 0)
ori_dft = cv2.dft(np.float32(ori_img), flags=cv2.DFT_COMPLEX_OUTPUT)
ori_dftShift = np.fft.fftshift(ori_dft)
ori_result = cv2.magnitude(ori_dftShift[:, :, 0], ori_dftShift[:, :, 1])

img = ori_img#cv2.GaussianBlur(ori_img,(3,3),1,1)
dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
dftShift = np.fft.fftshift(dft)
result = cv2.magnitude(dftShift[:, :, 0], dftShift[:, :, 1])

# 幅值归一化
dftShift[:,:,0]/=result
dftShift[:,:,1]/=result

# 逆傅立叶变换
ishift=np.fft.ifftshift(dftShift)
im2 = cv2.idft(ishift)
im=cv2.magnitude(im2[:,:,0],im2[:,:,1])


# 下面是生成一个大小为7x7，标准差为3的核函数
sigma = 3.0  # 高斯核标准差
size = 7  # 高斯核大小
# 生成一个二维坐标系
x, y = np.mgrid[-(size//2):(size//2)+1, -(size//2):(size//2)+1]
# 计算高斯核的值
kernel = np.exp(-(x**2 + y**2)/(2*sigma**2))
kernel = kernel / kernel.sum() 

# 对逆变换回来的图片进行高斯模糊
blur_img = cv2.filter2D(im,-1,kernel=kernel)
# 对高斯模糊降噪后的图片进行归一化，从而为了统一所有的图片的量纲一致，也就达到了马氏距离的目的
ma_img=(blur_img-np.mean(blur_img))/np.std(blur_img)

# 这里我设定马氏距离为4，和论文里面一致
dark_mask = ma_img<4
bright_mask = ma_img>=4

# 根据上面的mask生成二值图
ma_img[dark_mask]=0
ma_img[bright_mask]=255


# 这一步其实还是二值化，主要是为了转换成np.uint8,要不然opencv报错
_, fgmask = cv2.threshold(ma_img, 100, 255, cv2.THRESH_BINARY_INV)
fgmask = np.array(fgmask,dtype=np.uint8)

# 使用opencv找轮廓就行了，这里我过滤掉太小的瑕疵像素点
contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)#查找轮廓，输入的是二值图像
i = 0
for c in contours:
    perimeter = cv2.arcLength(c, True) #计算这个轮廓的周长
#     print(perimeter)
    if perimeter > 30 and perimeter < 1000: #轮廓周长大于188才画出来
        m = i
        x, y, w, h = cv2.boundingRect(c) #找到三角形的上右边界
        print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (123, 123), 2)
        # cv2.imshow(str(i), fgmask)
        # cv2.imwrite('D:\cycFeng\Data\\'+str(i)+'.jpg', fgmask)
#         print(hierarchy[0][i])
    i += 1
# 画出最终结果
plt.imshow(img,cmap='gray')
plt.show()
