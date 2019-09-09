# -*- coding:utf-8 -*-

import re
import os
import json
import PyPDF2
import numpy as np
from multiprocessing.pool import Pool

INPUT_DIR = '../input/'
OUTPUT_DIR = '../output/'

score_d = {
    '=': 1.2,
    '(': 0.2,
    ')': 0.3,
    '[': 0.3,
    ']': 0.3,
    '−': 0.6,
    '+': 1.5,
    '-': 0.6,
    '/': 1.5,
    '^': 1.1,
    '|': 0.9,
    '': 0
}
formula_pattern = re.compile(r'=|/|−|-|^|\||\[|\]|\+|\(\d{1,2}\)|\(\w\)|\(')
exclude_pattern = re.compile(r'\+\+|http|--|==|pdf')


def cal_score(r):
    s = 0
    for t in r:
        if len(t) > 1:
            s += 1
        else:
            s += score_d[t]
    return s


def get_formula_num(text):
    cnt = 0
    l = []
    tmp = ''
    for t in text.split('\n'):
        if len(t) <= 5:
            tmp += t
        else:
            if tmp != '':
                l.append(tmp)
                tmp = ''
            l.append(t)

    for t in l:
        p = exclude_pattern.findall(t)
        if len(t) > 100 or len(p) > 0: continue
        res = formula_pattern.findall(t)
        s = cal_score(res)
        if s > 3:
            cnt += 1
    return cnt


def get_pdf_meta(path):
    try:
        pdfFileObject = open(path, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObject)
        count = pdfReader.numPages
        figures = [0 for _ in range(8)]
        tables = [0 for _ in range(8)]
        formulas = [0 for _ in range(8)]

        for i in range(count):
            if i > 7: break
            page = pdfReader.getPage(i)
            text = page.extractText()
            # figures and table
            pattern = re.compile(r'(Figure|Table|Fig\.)\s*(\d+)(:)')
            res = pattern.findall(text)
            res = set([''.join(i) for i in res])
            for item in res:
                if item.startswith('Fig'):
                    figures[i] += 1
                else:
                    tables[i] += 1

            # formula
            formulas[i] = get_formula_num(text)

        if np.array(figures).sum() == 0 and np.array(tables).sum() == 0:
            s = set()
            for i in range(count):
                if i > 7: break
                page = pdfReader.getPage(i)
                text = page.extractText()
                pattern = re.compile(r'(Figure|Table|Fig\.)\s*(\d+)(\.[A-Z]{0,1})')
                res = pattern.findall(text)
                res = set([''.join(i).replace('.', ':').split(':')[0] for i in res]) - s
                s = s | res
                for item in res:
                    if item.startswith('Fig'):
                        figures[i] += 1
                    else:
                        tables[i] += 1
        pdfFileObject.close()
        return figures, tables, formulas, count
    except Exception as e:
        print(path, e)
        pdfFileObject.close()
        return [-1], [-1], [-1], -1


if __name__ == '__main__':
    print('execute lgb_process.py ...')
    lgb_output = OUTPUT_DIR + 'lgb_output/'
    if not os.path.exists(lgb_output):
        os.makedirs(lgb_output)

    def func(path):
        figures, tables, formulas, cnt = get_pdf_meta(path)
        return path, figures, tables, formulas, cnt


    for name in ['conference', 'arxiv']:
        lst = []
        conf_fold = INPUT_DIR + 'raw/%s/' % name
        for pdf in os.listdir(conf_fold):
            lst.append(conf_fold + pdf)

        ret = Pool(20).map(func, lst)
        json.dump(ret, open(lgb_output + '%s.json' % name, 'w', encoding='utf-8'), ensure_ascii=False)
