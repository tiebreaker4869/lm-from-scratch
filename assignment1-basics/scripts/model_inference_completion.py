from argparse import ArgumentParser
import torch
from cs336_basics.models import MiniLM
from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.inference import lm_decode
from cs336_basics.training_utils import get_dtype


def main():
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--tokenizer_dir", type=str, required=True, help="Directory containing vocab.json, merges.txt, special_tokens.txt")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt for completion")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling")
    # Model config
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    dtype = get_dtype(args.dtype)
    device = torch.device(args.device)

    # Load special tokens
    special_tokens = []
    with open(f"{args.tokenizer_dir}/special_tokens.txt", "r") as f:
        for line in f:
            token = line.strip()
            if token:
                special_tokens.append(token)

    # Load tokenizer
    tokenizer = BPETokenizer.from_files(
        vocab_filepath=f"{args.tokenizer_dir}/vocab.json",
        merges_filepath=f"{args.tokenizer_dir}/merges.txt",
        special_tokens=special_tokens
    )

    # Build model
    model = MiniLM(
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        theta=args.theta,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        device=device,
        dtype=dtype
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_weights"])
    model.eval()

    # Generate
    with torch.no_grad():
        output = lm_decode(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )

    print(output)


if __name__ == "__main__":
    main()
