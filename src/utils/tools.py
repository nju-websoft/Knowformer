import torch
import torch.nn as nn
from scipy.stats import truncnorm
from rich.console import Console
from rich.table import Table

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def truncated_normal(size, threshold=0.02):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


def truncated_normal_init(x, size, initializer_range):
    """
    Init a module x, init x.weight with truncated normal, and init x.bias with 0 if x has bias.

    Args:
        x: a pytorch module, it has attr weight or bias, we init the weight tensor or bias tensor with truncated normal.
        size: a list, depicts the shape of x.weight
        initializer_range: the range for truncated normal.

    Returns:
        None
    """
    x.weight.data.copy_(
        torch.from_numpy(truncated_normal(size, initializer_range)))
    if hasattr(x, "bias"):
        nn.init.constant_(x.bias, 0.0)


def norm_layer_init(x):
    """
    Init a norm layer x, init x.weight with 1, and init x.bias with 0.

    Args:
        x: nn.LayerNorm.

    Returns:
        None
    """
    nn.init.constant_(x.weight, 1.0)  # init
    nn.init.constant_(x.bias, 0.0)  # init


def print_results(dataset_name, eval_performance, k=3):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Dataset", justify="center")
    table.add_column("MRR", justify="center")
    table.add_column("Hits@1", justify="center")
    table.add_column("Hits@3", justify="center")
    table.add_column("Hits@10", justify="center")
    table.add_row(dataset_name, str(round(eval_performance['fmrr'], k)), str(round(eval_performance['fhits1'], k)),
                  str(round(eval_performance['fhits3'], k)), str(round(eval_performance['fhits10'], k)))
    console.print(table)
