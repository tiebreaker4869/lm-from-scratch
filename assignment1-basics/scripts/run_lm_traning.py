from argparse import ArgumentParser
from cs336_basics.training_utils import cross_entropy, learning_rate_schedule, load_as_array, gradient_clipping, get_next_batch, save_checkpoint, sample_dataset, get_dtype, eval_validation
from cs336_basics.models import MiniLM
from cs336_basics.optimizers import AdamW
import os
import tqdm
import wandb
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = ArgumentParser()
    parser.add_argument('--d_model', type=int, default=896)
    parser.add_argument('--d_ff', type=int, default=4864)
    parser.add_argument('--num_heads', type=int, default=14)
    parser.add_argument('--theta', type=float, default=10000.0)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--context_length', type=int, default=4096)
    parser.add_argument('--num_layers', type=int, default=24)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dtype', type=str, choices=['float32', 'bfloat16', 'float16'], default='float32')
    parser.add_argument('--max_lr', type=float, default=6e-4)
    parser.add_argument('--min_lr', type=float, default=6e-5)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--total_steps', type=int, default=100000)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--eval_period', type=int, default=1000)
    parser.add_argument('--checkpoint_period', type=int, default=5000)
    parser.add_argument('--wandb_entity', type=str, default='lifansun1412')
    parser.add_argument('--wandb_project', type=str, default='lm-from-scratch')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--datasets', nargs='+', type=str, required=True)
    parser.add_argument('--datasets_sampling_probs', nargs='+', type=float, required=True)
    parser.add_argument('--validation_datasets', nargs='+', type=str, required=True)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    training_datasets = [load_as_array(d) for d in args.datasets]
    validation_datasets = [load_as_array(d) for d in args.validation_datasets]
    sampling_probs = args.datasets_sampling_probs

    wandb.init(entity=args.wandb_entity, project=args.wandb_project, config=vars(args))

    model = MiniLM(args.d_model, args.d_ff, args.num_heads, args.theta, args.vocab_size, args.context_length, args.num_layers, args.device, get_dtype(args.dtype))
    model.train()
    optimizer = AdamW(model.parameters(), lr=args.max_lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

    best_validation_loss = float('inf')

    for step in tqdm.trange(1, args.total_steps + 1, desc="Training"):
        dataset = sample_dataset(training_datasets, sampling_probs)
        sequences, targets = get_next_batch(dataset, args.batch_size, args.context_length, args.device)

        logits = model(sequences)
        loss = cross_entropy(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)

        lr = learning_rate_schedule(step, args.max_lr, args.min_lr, args.warmup_steps, args.total_steps)
        optimizer.adjust_lr(lr)
        optimizer.step()

        wandb.log({"train_loss": loss.item(), "lr": lr}, step=step)

        if step % args.eval_period == 0:
            model.eval()
            validation_loss = eval_validation(model, validation_datasets, args.batch_size, args.context_length, args.device)
            wandb.log({"validation_loss": validation_loss}, step=step)
            model.train()

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                ckpt_path = os.path.join(args.checkpoint_dir, "best.bin")
                save_checkpoint(model, optimizer, step, ckpt_path)
                logging.info(f"Step {step}: New best validation loss: {validation_loss:.4f}")

        if step % args.checkpoint_period == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step}.bin")
            save_checkpoint(model, optimizer, step, ckpt_path)

    wandb.finish()
    logging.info(f"Training finished. Best validation loss: {best_validation_loss:.4f}")

if __name__ == "__main__":
    main()
