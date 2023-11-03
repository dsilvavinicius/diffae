import os
import os.path as osp
from multiprocessing import Pool

import numpy as np
import PIL.Image
from tqdm import tqdm
from argparse import ArgumentParser

LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'


def image_align(src_file,
                dst_file):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    output_size = 256

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print(
            '\nCannot find source image. Please run "--wilds" before "--align".'
        )
        return
    img = PIL.Image.open(src_file)
    img = img.convert('RGB')
    center = (img.width // 2, img.height // 2)
    new_width = img.width * 0.8
    new_height = img.height * 0.8

    img = img.crop((center[0] - new_width // 2, center[1] - new_height // 2, center[0] + new_width // 2, center[1] + new_height // 2))

    # Transform.
    img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

    # Save aligned image.
    img.save(dst_file, 'PNG')


def work_landmark(raw_img_path, img_name):
    face_img_name = '%s.png' % (os.path.splitext(img_name)[0], )
    aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
    #if os.path.exists(aligned_face_path):
    #    return
    image_align(raw_img_path,
                aligned_face_path)


if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    parser = ArgumentParser()
    parser.add_argument("-i",
                        "--input_imgs_path",
                        type=str,
                        default="imgs",
                        help="input images directory path")
    parser.add_argument("-o",
                        "--output_imgs_path",
                        type=str,
                        default="imgs_crop",
                        help="output images directory path")
    parser.add_argument(
        "--no-align", "-n", action="store_true", help=""
    )

    args = parser.parse_args()

    # RAW_IMAGES_DIR = sys.argv[1]
    # ALIGNED_IMAGES_DIR = sys.argv[2]
    RAW_IMAGES_DIR = args.input_imgs_path
    ALIGNED_IMAGES_DIR = args.output_imgs_path

    if not osp.exists(ALIGNED_IMAGES_DIR): os.makedirs(ALIGNED_IMAGES_DIR)

    files = os.listdir(RAW_IMAGES_DIR)
    print(f'total img files {len(files)}')
    with tqdm(total=len(files)) as progress:

        def cb(*args):
            # print('update')
            progress.update()

        def err_cb(e):
            print('error:', e)

        with Pool(8) as pool:
            res = []
            for img_name in files:
                raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
                # print('img_name:', img_name)
                # assert i == 1, f'{i}'
                # print(i, face_landmarks)
                # face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
                # aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
                # image_align(raw_img_path, aligned_face_path, face_landmarks, output_size=256)

                work_landmark(raw_img_path, img_name)
                progress.update()

                # job = pool.apply_async(
                #     work_landmark,
                #     (raw_img_path, img_name, face_landmarks),
                #     callback=cb,
                #     error_callback=err_cb,
                # )
                # res.append(job)

            # pool.close()
            # pool.join()
    print(f"output aligned images at: {ALIGNED_IMAGES_DIR}")
