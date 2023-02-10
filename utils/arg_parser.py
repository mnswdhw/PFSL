import argparse


def parse_arguments():
    # Training settings
    parser = argparse.ArgumentParser(
        description="Split Learning Research Simulation entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--number_of_clients",
        type=int,
        default=10,
        metavar="C",
        help="Number of Clients",
    )
    parser.add_argument(
        "--sst",
        # action=argparse.BooleanOptionalAction,
        action = "store_true",
        help="State if server side tuning needs to be done",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=128,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=128,
        metavar="TB",
        help="Input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="Total number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="Learning rate",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "--server_sigma",
        type=float,
        default=0,
        metavar="SS",
        help="Noise multiplier for central layers",
    )
    parser.add_argument(
        "-g",
        "--max_per_sample_grad_norm",
        type=float,
        default=1.0,
        metavar="G",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(        # needs to be implemented
        "--save_model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--disable_dp",
        action="store_true",
        default=False,
        help="Disable privacy training and just train with vanilla SGD",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="States dataset to be used",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18_split",
        help="Model you would like to train",
    )
    parser.add_argument(
        "--epoch_batch",
        type=str,
        default="5",
        help="Number of epochs after which next batch of clients should join",
    )
    parser.add_argument(
        "--opt_iden",
        type=str,
        default="",
        help="optional identifier of experiment",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        default=False,
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=False,
        help="Use transfer learning using a pretrained model",
    )
    parser.add_argument(
        "--datapoints",
        type=int,
        default=500,
        help="Number of samples of training data allotted to each client",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default='setting1',
        help='Setting you would like to run for, i.e, setting1 , setting2 or setting4'
       
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=50,
        help="Epoch at which personalisation phase will start",
    )

    args = parser.parse_args()
    return args
