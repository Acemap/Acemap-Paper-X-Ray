# -*- coding:utf-8 -*-

import os
import fitz
import random
import shutil
from tqdm import tqdm
import PIL.Image as Image

INPUT_DIR = '../input/'
OUTPUT_DIR = '../output/'
TEST_SIZE = 0.2

# convert the first W*H pages of pdf to one image,
# which has H rows and W pages in each row
W = 3
H = 2


def save_jpg(src, tgt, tmp_dir):
    doc = fitz.open(src)
    # if doc.pageCount < W * H + 1: # pagenum>=W*H without considering reference page
    #     return

    for pg in range(min(doc.pageCount, W * H)):
        page = doc[pg]
        zoom = 40
        rotate = 0
        trans = fitz.Matrix(zoom / 100.0, zoom / 100.0).preRotate(rotate)

        # create raster image of page (non-transparent)
        pm = page.getPixmap(matrix=trans, alpha=False)

        # write a PNG image of the page
        pm.writePNG(tmp_dir + '%s.png' % pg)

    index = 0
    fromImage = Image.open(tmp_dir + '%s.png' % index)
    width, height = fromImage.size
    toImage = Image.new('RGB', (width * W, height * H))
    for i in range(H):
        for j in range(W):
            fname = tmp_dir + '%s.png' % index
            try:
                fromImage = Image.open(fname) if index > 0 else Image.new('RGB', (width, height), 'white')
            except Exception as e:
                fromImage = Image.new('RGB', (width, height), 'white')

            toImage.paste(fromImage, (j * width, i * height))
            index += 1

    for i in range(W * H):
        fname = tmp_dir + '%s.png' % i
        if os.path.exists(fname):
            os.remove(fname)
    toImage.save(tgt)


if __name__ == '__main__':
    print('execute nn_process.py ...')
    conf_fold = INPUT_DIR + 'raw/conference/'
    arxiv_fold = INPUT_DIR + 'raw/arxiv/'
    tmp_dir = INPUT_DIR + 'temp/'

    # make folders
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    for mode in ['train', 'test']:
        for name in ['conference', 'arxiv']:
            fold = INPUT_DIR+'/%s/%s/' % (mode, name)
            if not os.path.exists(fold):
                os.makedirs(fold)

    # convert pdf to images
    for fold, s in [[conf_fold, 'conference'], [arxiv_fold, 'arxiv']]:
        tgt_fold = INPUT_DIR+'/train/%s/' % s
        for pdf in tqdm(os.listdir(fold)):
            src = fold + pdf
            tgt = tgt_fold + pdf.replace('pdf', 'jpg')
            try:
                save_jpg(src, tgt, tmp_dir)
            except:
                pass

    # train_test_split
    arxiv_lst = os.listdir(INPUT_DIR+'train/arxiv')
    conf_lst = os.listdir(INPUT_DIR+'train/conference')
    random.shuffle(arxiv_lst)
    random.shuffle(conf_lst)
    for i in arxiv_lst[:int(len(arxiv_lst) * TEST_SIZE)]:
        shutil.move(INPUT_DIR+'train/arxiv/' + i, INPUT_DIR+'test/arxiv/' + i)
    for i in conf_lst[:int(len(conf_lst) * TEST_SIZE)]:
        shutil.move(INPUT_DIR+'train/conference/' + i, INPUT_DIR+'test/conference/' + i)