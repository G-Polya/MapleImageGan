def read_data(data_dir, image_size, crop_size=None):
    """
    GAN을 학습하기 위해 데이터를 전처리하고 불러옴
    :param data_dir : str, image가 저장된 경로.
    :param image_size : tuple (width, height), 이미지를 resize할 경우 이미지 사이즈
    :param crop_size : int, 얼굴 이미지에서 배경을 제외한 얼굴만을 crop할경우 crop할 영역의 크기
    :return: X_set : np.ndarray, shape: (N, H, W, C).
    """
    img_list = [img for img in os.listdir(data_dir) if img.split(".")[-1] in IMAGE_EXTS]
    images = []
    
    for img in img_list:
        img_path = os.path.join(data_dir, img)
        im = imread(img_path)
        im = np.array(im, dtype=np.float32)
        if crop_size:
            im = center_crop(im, crop_size, crop_size)
        else:
            im = resize(im, (image_size[1], image_size[0]))
        im = im/127.5 - 1
        im = im[:,:,::-1]
        images.append(im)
        
    X_set = np.array(images, dtype=np.float32)
    
    return X_set