import pandas as pd
import numpy as np
import torch

def eval_sudokus(sudokus, zero_base=True):
    def is_valid_seqs(seqs):
        sorted_seqs = torch.sort(seqs, dim=-1)[0]
        if zero_base:
            correct = torch.arange(9, device=seqs.device).expand_as(sorted_seqs)
        else:
            correct = torch.arange(1, 10, device=seqs.device).expand_as(sorted_seqs)
        return torch.all(sorted_seqs == correct, dim=-1)

    row_seqs = torch.stack([sudokus[:, i:i+1, :].reshape(-1, 9) for i in range(9)], dim=1)
    col_seqs = torch.stack([sudokus[:, :, j:j+1].reshape(-1, 9) for j in range(9)], dim=1)
    box_seqs = torch.stack([sudokus[:, i:i+3, j:j+3].reshape(-1, 9) for i in [0, 3, 6] for j in [0, 3, 6]], dim=1)

    row_vals = is_valid_seqs(row_seqs)
    col_vals = is_valid_seqs(col_seqs)
    box_vals = is_valid_seqs(box_seqs)

    row_acc = ( row_vals.sum(dim=1)/row_vals.size(dim=1) )
    col_acc = ( col_vals.sum(dim=1)/col_vals.size(dim=1) )
    box_acc = ( box_vals.sum(dim=1)/box_vals.size(dim=1) )
    sdk_acc = torch.stack([row_vals.all(dim=1), col_vals.all(dim=1), box_vals.all(dim=1)]).all(dim=0).long()

    return pd.DataFrame(dict(row=row_acc.numpy(), col=col_acc.numpy(), box=box_acc.numpy(), sudoku=sdk_acc.numpy()))

def eval_num_accs(preds, trues):
    num_accs = []
    for pred, true in zip(preds, trues):
        num_acc = (pred==true).sum()/81
        num_accs.append(num_acc.numpy())
    return np.array(num_accs)
