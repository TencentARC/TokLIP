import torch
import sys
import json
from collections import defaultdict
from os.path import join, dirname, abspath


@torch.no_grad()
def itm_eval(score_matrix, txt_ids, img_ids, txt2img, img2txts):
    # match the code, col: images, row: txts 
    score_matrix = score_matrix.t()
    # [1000, 5005] ---> [5005, 1000]

    # text to images retrieval
    img2j = {i: j for j, i in enumerate(img_ids)}
    _, rank_txt = score_matrix.topk(10, dim=1)
    groud_truth_img_j = torch.LongTensor([img2j[txt2img[txt_id]]
                                            for txt_id in txt_ids],
                                         ).to(rank_txt.device
                                              ).unsqueeze(1).expand_as(rank_txt)
    rank = (rank_txt == groud_truth_img_j).nonzero()
    if rank.numel():
        ir_r1 = (rank < 1).sum().item() / len(txt_ids)
        ir_r5 = (rank < 5).sum().item() / len(txt_ids)
        ir_r10 = (rank < 10).sum().item() / len(txt_ids)
    else:
        ir_r1, ir_r5, ir_r10 = 0, 0, 0


    # images to text retrival
    txt2i = {t: i for i, t in enumerate(txt_ids)}
    _, rank_img = score_matrix.topk(10, dim=0)

    tr_r1, tr_r5, tr_r10 = 0, 0, 0
    for j, img_id in enumerate(img_ids):
        ground_truth_is = [txt2i[t] for t in img2txts[img_id]]
        ranks = [(rank_img[:, j] == i).nonzero() for i in ground_truth_is]
        rank = min([10] + [r.item() for r in ranks if r.numel()])
        if rank < 1:
            tr_r1 += 1
        if rank < 5:
            tr_r5 += 1
        if rank < 10:
            tr_r10 += 1
    tr_r1 /= len(img_ids)
    tr_r5 /= len(img_ids)
    tr_r10 /= len(img_ids)

    tr_mean = (tr_r1 + tr_r5 + tr_r10) / 3
    ir_mean = (ir_r1 + ir_r5 + ir_r10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'txt_r1': tr_r1,
                   'txt_r5': tr_r5,
                   'txt_r10': tr_r10,
                   'txt_r_mean': tr_mean,
                   'img_r1': ir_r1,
                   'img_r5': ir_r5,
                   'img_r10': ir_r10,
                   'img_r_mean': ir_mean,
                   'r_mean': r_mean}
    return eval_result


def get_caption_img_list_dict(path, root_path):
    img2caption_dict = defaultdict(list)
    coco = []
    # print(path)
    for line in open(path, 'r'):
        coco.append(json.loads(line))

    img_dict = {}
    caption_dict = {}
    img_caption_pair_list = []
    img2captions = defaultdict(list)
    caption2img = {}
    caption_cnt = 0

    for i, img in enumerate(coco):
        # print(i, img)
        img_dict[i] = join(root_path, img['filename'])
        for caption in img['caption']:
            caption_dict[caption_cnt] = caption
            img_caption_pair_list.append((i, caption_cnt))
            img2captions[i].append(caption_cnt)
            caption2img[caption_cnt] = i
            caption_cnt += 1
    caption_ids = list(range(caption_cnt))
    img_ids = list(range(len(img_dict)))
    # print(caption_cnt)
    # print(len(img_dict))
    return caption_ids, img_ids, caption2img, img2captions

def calc_metric(score_matrix, root_path):
    path = './coco_json/coco_test.json'
    caption_ids, img_ids, caption2img, img2captions = get_caption_img_list_dict(path, root_path)
    eval_log = itm_eval(score_matrix, caption_ids, img_ids, caption2img, img2captions)

    print(f'evaluation finished')
    print(
        f"=======================Results==========================\n"
        f"text Retrieval R1: {eval_log['txt_r1'] * 100: .2f}, \n"
        f"text Retrieval R5: {eval_log['txt_r5'] * 100: .2f}, \n"
        f"text Retrieval R10: {eval_log['txt_r10'] * 100: .2f}, \n"
        f"text Retrieval Mean: {eval_log['txt_r_mean'] * 100: .2f}, \n"
        f"image Retrieval R1: {eval_log['img_r1'] * 100: .2f}, \n"
        f"image Retrieval R5: {eval_log['img_r5'] * 100: .2f}, \n"
        f"image Retrieval R10: {eval_log['img_r10'] * 100: .2f}, \n"
        f"image Retrieval Mean: {eval_log['img_r_mean'] * 100: .2f}, \n"
        f"Retrieval Mean: {eval_log['r_mean'] * 100: .2f}, \n"
    )
    return eval_log