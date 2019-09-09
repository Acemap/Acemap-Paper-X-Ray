# -*- coding:utf-8 -*-

import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import lightgbm as lgb
import PIL.Image as Image
from torchvision import models
from torchvision import transforms
from torch.nn import functional as F

sys.path.append('..')
from nn_process import save_jpg
from lgb_process import get_pdf_meta

INPUT_DIR = '../input/'
OUTPUT_DIR = '../output/'
NET_WEIGHT = 0.5


def get_nn_model():
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    ckp_path = OUTPUT_DIR + 'nn_output/model_best.pth.tar'
    checkpoint = torch.load(ckp_path)
    d = checkpoint['state_dict']
    d = {k.replace('module.', ''): v for k, v in d.items()}
    model.load_state_dict(d)
    return model


def get_lgb_model():
    return lgb.Booster(model_file=OUTPUT_DIR + 'lgb_output/lgb_model.txt')


def load_img(path):
    img = Image.open(path)
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    img = trans(img)
    img = torch.unsqueeze(img, 0)
    return img


def predict(pdf_path):
    # lgb predict
    figures, tables, formulas, cnt = get_pdf_meta(pdf_path)
    meta = figures + tables + formulas + [cnt]
    gbm_score = lgb_model.predict(np.array([meta]))[0]

    # nn predict
    tmp_dir = INPUT_DIR + 'temp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    tgt_path = tmp_dir + 'image.jpg'
    print(tgt_path)
    save_jpg(pdf_path, tgt_path, tmp_dir)
    img = load_img(tgt_path)
    logit = nn_model(img)
    probs = F.softmax(logit, dim=1).data.squeeze()
    nn_score = probs[1].numpy()

    score = NET_WEIGHT * nn_score + (1 - NET_WEIGHT) * gbm_score
    return round(score*100, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf_path", type=str, help="pdf_path")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    pdf_path = args.pdf_path

    nn_model = get_nn_model()
    lgb_model = get_lgb_model()
    score = predict(pdf_path)
    print('The score of (%s) is %.1f' % (pdf_path, score))
