import torch
import argparse
from PIL import Image
from create_toklip import create_toklip
from open_clip import get_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, default='ViT-SO400M-16-SigLIP2-384-toklip')
    parser.add_argument("--pretrained", type=str, default='TokLIP_L_384.pt')
    args = parser.parse_args()

    model, _, preprocess = create_toklip(model=args.model_config, image_size=384, model_path=args.pretrained)
    model.half()
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = get_tokenizer(args.model_config)

    image = preprocess(Image.open("../docs/CLIP.png")).unsqueeze(0).half()
    text = tokenizer(["a diagram", "a dog", "a cat"])


    with torch.no_grad(), torch.autocast("cuda"):
        image_features = model.encode_image(image.cuda())
        text_features = model.encode_text(text.cuda())
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)


    text_probs = torch.round(text_probs * 10) / 10  # Format to 1 decimal place
    print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]



if __name__ == "__main__":
    main()