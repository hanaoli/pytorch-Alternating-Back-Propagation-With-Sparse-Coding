from parsers import get_parser
from model.abp_sc import ABPSC
import torch
from torchvision import datasets, transforms

if __name__ == "__main__":
    parser = get_parser().parse_args()
    print("Alternating Back Propagation with Sparse Coding")

    torch.manual_seed(parser.seed)
    torch.cuda.manual_seed(parser.seed)

    train_dateset = datasets.MNIST(root='./mnist_data', train=True, transform=transforms.ToTensor(), download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dateset, batch_size=parser.batch_size)

    abpsc = ABPSC(parser)
    abpsc.train(train_loader)