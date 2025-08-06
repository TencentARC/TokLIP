import torch
from open_clip.factory_openclip import create_model_from_pretrained, get_tokenizer
import argparse
import os


def download_and_save_model(model_name, save_path):
    """
    Download pre-trained model and save to specified path
    
    Args:
        model_name (str): Model name, e.g. 'ViT-SO400M-14-SigLIP-384'
        save_path (str): Model save path
    """
    print(f"Downloading model: {model_name}")
    
    # Load pre-trained model and tokenizer
    model, preprocess = create_model_from_pretrained(model_name, pretrained='webli')
    tokenizer = get_tokenizer(model_name)
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model
    print(f"Saving model to: {save_path}")
    torch.save(model.state_dict(), save_path)
    
    print("Model download and save completed!")
    return model, tokenizer, preprocess

def main():
    parser = argparse.ArgumentParser(description='Download and save OpenCLIP model')
    parser.add_argument('--model_name', type=str, default='ViT-SO400M-16-SigLIP2-384',
                        help='Model name (default: ViT-SO400M-16-SigLIP2-384)')
    parser.add_argument('--save_path', type=str, 
                        default='./model/siglip2-so400m-vit-l16-384.pt',
                        help='Model save path')
    
    args = parser.parse_args()
    
    # Download and save model
    model, tokenizer, preprocess = download_and_save_model(args.model_name, args.save_path)
    
    print(f"Model information:")
    print(f"  Model name: {args.model_name}")
    print(f"  Save path: {args.save_path}")
    print(f"  Model type: {type(model).__name__}")

if __name__ == "__main__":
    main()



# TIMM_MODEL='original' python covert.py