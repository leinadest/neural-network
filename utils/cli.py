import argparse

parser = argparse.ArgumentParser(
    description="CLI for training and evaluating a neural network.",
)

subparsers = parser.add_subparsers(dest="command", required=True)

train_parser = subparsers.add_parser("train", help="Train the neural network")
train_parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training")
train_parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for training")
train_parser.add_argument("--minibatch", type=int, default=100, help="Size of minibatches for training")
train_parser.add_argument("--reg", type=float, default=0.001, help="L2 regularization for training")
train_parser.add_argument("--model", default="models/model.json", help="Path to save the trained model")

eval_parser = subparsers.add_parser("evaluate", help="Evaluate the neural network")
eval_parser.add_argument("--model", default="models/model.json", help="Path to the trained model")

predict_parser = subparsers.add_parser("predict", help="Make predictions using the neural network")
predict_parser.add_argument("--model", default="models/model.json", help="Path to the trained model")
predict_parser.add_argument("--output", default="predictions.csv", help="Path to save predictions")

args = parser.parse_args()

if args.command == "train":
    print(f"Epochs = {args.epochs}")
    print(f"Learning Rate = {args.lr}")
    print(f"Mini-batch Size = {args.minibatch}")
    print(f"L2 Regularization = {args.reg}")
    print("Training model...")
elif args.command == "evaluate":
    print(f"Loading model from {args.model}...")
    print("Evaluating model...")
elif args.command == "predict":
    print(f"Loading model from {args.model}...")
    print("Making predictions...")