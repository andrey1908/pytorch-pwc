import argparse
from run import estimate
import numpy
import torch
import os
import os.path as osp
from tqdm import tqdm
from PIL import Image


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img-fld', '--images-folder', type=str, required=True)
    parser.add_argument('-m', '--model-name', type=str, default='default')
    parser.add_argument('-out-fld', '--out-folder', type=str, required=True)
    return parser


def run_pair(first_image_file, second_image_file, model_name):
    firstImage = Image.open(first_image_file)
    secondImage = Image.open(second_image_file)

    assert (firstImage.mode == 'RGB')
    assert (secondImage.mode == 'RGB')

    tenFirst = torch.FloatTensor(numpy.ascontiguousarray(
        numpy.array(firstImage)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))
    tenSecond = torch.FloatTensor(numpy.ascontiguousarray(
        numpy.array(secondImage)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (1.0 / 255.0)))

    tenOutput = estimate(tenFirst, tenSecond, model_name)

    return tenOutput


def run_sequence(images_folder, model_name, out_folder):
    if not osp.exists(out_folder):
        os.mkdir(out_folder)
    image_files = os.listdir(images_folder)
    image_files.sort()
    for image_file in image_files:
        print(image_file)
    for first_image_file, second_image_file in tqdm(list(zip(image_files[:-1], image_files[1:]))):
        out_flow = run_pair(osp.join(images_folder, first_image_file), osp.join(images_folder, second_image_file),
                            model_name)
        out_file = '.'.join(first_image_file.split('.')[:-1]) + '.flo'
        with open(osp.join(out_folder, out_file), 'wb') as f:
            numpy.array([80, 73, 69, 72], numpy.uint8).tofile(f)
            numpy.array([out_flow.shape[2], out_flow.shape[1]], numpy.int32).tofile(f)
            numpy.array(out_flow.numpy().transpose(1, 2, 0), numpy.float32).tofile(f)
    out_file = '.'.join(image_files[-1].split('.')[:-1]) + '.flo'
    with open(osp.join(out_folder, out_file), 'wb') as f:
        numpy.array([80, 73, 69, 72], numpy.uint8).tofile(f)
        numpy.array([out_flow.shape[2], out_flow.shape[1]], numpy.int32).tofile(f)
        numpy.array(out_flow.numpy().transpose(1, 2, 0), numpy.float32).tofile(f)


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()
    run_sequence(args.images_folder, args.model_name, args.out_folder)
