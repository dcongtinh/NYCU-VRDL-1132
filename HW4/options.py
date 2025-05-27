import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=200,
                    help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int, default=6,
                    help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['derain', 'desnow'],
                    help='which type of degradations is training and testing for.')

parser.add_argument('--patch_size', type=int, default=128,
                    help='patchsize of input.')
parser.add_argument('--num_workers', type=int,
                    default=16, help='number of workers.')

# path
parser.add_argument('--derain_dir', type=str, default='data_split/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--desnow_dir', type=str, default='data_split/Train/Desnow/',
                    help='where training images of desnowing saves.')

parser.add_argument("--num_gpus", type=int, default=1,
                    help="Number of GPUs to use for training")
parser.add_argument("--num_duplicate", type=int, default=15,)

options = parser.parse_args()
