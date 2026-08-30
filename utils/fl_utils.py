import collections
import torch
import torch.nn.functional as F


def avg_EW(ws, ms):
    """Federated-average a list of state dicts, weighted by a 0/1 (or bool)
    per-client mask indicating which clients actually contributed."""
    m = torch.stack(ms, dim=0)
    count = int((m==True).sum())
    weight_keys = ws[0].keys()
    avg_state_dict = collections.OrderedDict()
    for key in weight_keys:
        key_sum = None
        for c in range(len(ws)):
            if key_sum is None:
                key_sum = ms[c] * ws[c][key].data.cpu()
            else:
                key_sum += ms[c] * ws[c][key].data.cpu()

        avg_state_dict[key] = key_sum / count
    return avg_state_dict


def apply_morphological_operations(mask, operation='dilation', kernel_size=3, iterations=1):
    """
    Apply morphological operations (dilation or erosion) to a binary mask using PyTorch.

    Parameters:
    - mask: torch tensor of shape (1, 4, 80, 80, 80)
    - operation: 'dilation' or 'erosion'
    - kernel_size: size of the kernel for the operation
    - iterations: number of iterations for the operation

    Returns:
    - result_mask: torch tensor of shape (1, 4, 80, 80, 80) after applying the operation
    """
    kernel = torch.ones((kernel_size, kernel_size, kernel_size), dtype=torch.float32, device=mask.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    result_mask = mask.clone()
    for _ in range(iterations):
        if operation == 'dilation':
            for channel in range(mask.shape[1]):
                result_mask[:, channel: channel+1] = F.conv3d(result_mask[:, channel: channel+1].float(), kernel, padding=kernel_size//2) > 0
        elif operation == 'erosion':
            for channel in range(mask.shape[1]):
                result_mask[:, channel: channel+1] = F.conv3d(result_mask[:, channel: channel+1].float(), kernel, padding=kernel_size//2) == kernel.numel()
        else:
            raise ValueError("Operation must be 'dilation' or 'erosion'")

    return result_mask.float()
