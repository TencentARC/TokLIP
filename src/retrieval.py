import torch
import os
import json
from tqdm.notebook import tqdm
from argparse import ArgumentParser
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image

from dataloader.clip_dataset_coco import ClipDatasetCOCO
from easydict import EasyDict
from calc_retrieval_metric import calc_metric

V2_DATASET_SIZE = 10000
VAL_DATASET_SIZE = 50000

def _collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    image_ids = [_['image_id'] for _ in batch]
    filenames = [_['filename'] for _ in batch]
    if type(batch[0]['image']) == list:
        images = [torch.stack([_['image'][0] for _ in batch]), torch.stack([_['image'][1] for _ in batch]), \
                  torch.stack([_['image'][2] for _ in batch]), torch.stack([_['image'][3] for _ in batch])]
              
    else:
        images = torch.stack([_['image'] for _ in batch])

    labels = torch.as_tensor([_.get('label', -1)
                              for _ in batch], dtype=torch.long)
    label_names = [_.get('label_name', None) for _ in batch]
    captions = [_.get('caption', []) for _ in batch]
    tags = [_.get('tag', []) for _ in batch]

    output = EasyDict({
        'image_ids': image_ids,
        'filenames': filenames,
        'images': images,
        'captions': captions,
        'tags': tags,
    })

    output['labels'] = labels if labels[0] is not None else None
    output['label_names'] = label_names if label_names[0] is not None else None

    return output


def retrieval(args, model, preprocess, tokenizer):

    CurrDataset = ClipDatasetCOCO

    dataset = CurrDataset(
        root_dir=args.coco_dir,
        meta_file='./coco_json/coco_test.json',
        transform=preprocess,
        read_from='fs',
        image_reader_type='pil',
    )


    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=16, collate_fn=_collate_fn)

    with torch.no_grad():
        text_feats = []
        text_embeds = []

        image_feats = []
        image_embeds = []

        all_captions = []

        for batch_idx, batch in enumerate(loader):
            input = batch['images']
            input = input.cuda()
            # label = label.squeeze().view(-1).cuda().long()
            # compute output
            if 'siglip' or 'toklip' in args.name:
                image_pred = model.encode_image(input)
            else:
                image_pred, _ = model.encode_image(input)

            image_embed = image_pred / (image_pred.norm(dim=-1, keepdim=True))
            image_feats.append(image_pred)
            image_embeds.append(image_embed)

            captions = batch['captions']

            caption_texts = []
            for caption in captions:
                caption_texts.extend(caption)


            texts = tokenizer(caption_texts)
            texts = texts.cuda()
            captions_pred = model.encode_text(texts)

            text_feats.append(captions_pred)

            text_embed = captions_pred / (captions_pred.norm(dim=-1, keepdim=True))

            text_embeds.append(text_embed)
            all_captions.append(caption_texts)

            number = batch_idx


        # print(len(text_embeds))

        text_embeds = torch.cat(text_embeds, dim=0)
        text_feats = torch.cat(text_feats, dim=0)

        image_feats = torch.cat(image_feats, dim=0)
        image_embeds = torch.cat(image_embeds, dim=0)


        # print(image_embeds.shape)
        # print(text_embeds.shape)

        # image to text
        sims_matrix = image_embeds @ text_embeds.t()
        # print("sim_matrix:", sims_matrix.shape)   

        metrics = calc_metric(sims_matrix, args.coco_dir)
        return metrics


def main():
    parser = ArgumentParser(description='Evaluating CLIP.')
    parser.add_argument('--pretrained_model_dir', type=str,
                        default='models/ViT-B-32.pt')
    parser.add_argument('--test_dataset', dest='test_dataset',
                        default='imagenet')
    parser.add_argument('--test_data_dir', dest='test_data_dir', )
    parser.add_argument('--train_data_dir', dest='train_data_dir')
    parser.add_argument('--test_output_dir', dest='test_output_dir', default='output_test')
    parser.add_argument("--prompt_ensemble", default=0, type=int,
                        help="using an emsemble of prompts")
    # late interaction
    parser.add_argument("--late_interaction", default=0, type=int,
                        help="using the late interaction as the colbert paper")
    parser.add_argument("--text_ensemble", default=1, type=int,
                        help="ensemble the text with mask tokens like the query ensemble in colbert")
    parser.add_argument("--text_encoder_pretrained", default=0, type=int,
                        help="whether using a pretrained text encoder, use together with text_encoder_type")
    parser.add_argument("--text_encoder_type", default='gpt',
                        help="text encoder type")

if __name__ == "__main__":
    main()