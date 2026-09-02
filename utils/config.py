import yaml
import argparse

def load_config(path="config/default.yaml"):
    with open(path, "r") as f:
        config=yaml.safe_load(f)
    return config

def get_args():
    parser=argparse.ArgumentParser(description="DeepLab Training Pipeline")
    parser.add_argument("--config", type=str, default="config/default.yaml",help="Path to YAML config file")
    parser.add_argument("--model", type=str, help="model name")
    parser.add_argument("--epochs", type=int, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, help="batch size")
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--resume", type=str, help="path to checkpoint to resume from")
    parser.add_argument("--loss", type=str, help="loss type: bce_dice, focal")
    parser.add_argument("--use-amp", action="store_true", help="enable mixed precision")
    parser.add_argument("--experiment", type=str, default="default", help="experiment name for TensorBoard")

    args = parser.parse_args()
    return args

def get_config():
    args = get_args()
    config = load_config(args.config)

    if args.model:
        config["model"]["name"] = args.model
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.loss:
        config["loss"]["type"] = args.loss
    if args.use_amp:
        config["training"]["use_amp"] = True
    if args.resume:
        config["checkpoint"]["resume"] = args.resume

    config["device"] = args.device
    config["experiment"] = args.experiment
    
    return config