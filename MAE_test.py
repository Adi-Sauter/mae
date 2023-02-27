import models_mae
import torch
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


def prep_model(checkpoint_dir, arch='mae_vit_large_patch16'):
    model = getattr(models_mae, arch)()
    checkpoint = torch.load(checkpoint_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    # print(msg)
    return model


def get_img_list(img_dir, img_extension='jpg'):
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    imgs_list = []
    dir_path = 'data.nosync/frames/'
    for file_name in os.listdir(dir_path):
        print(file_name)
        #print(os.path.splitext(file_name)[1].lower())
        if os.path.splitext(file_name)[1].lower() == img_extension:
            im = Image.open(os.path.join(dir_path, file_name))
            imgs_list.append(im)
            #print(type(im))

    #for img in imgs_list:
    #    img = img.resize((224, 224))
    #    img = np.array(img) / 255.
    #    assert img.shape == (224, 224, 3)
    return imgs_list


def main():
    checkpoint_dir = './mae_main/checkpoints/mae_visualize_vit_large.pth'
    #model = prep_model(checkpoint_dir)
    img_dir = './data.nosync/frames'
    imgs = get_img_list(img_dir, 'jpg')
    print(imgs)
    #plt.imshow(imgs[0])
    #plt.show()


if __name__ == '__main__':
    main()



