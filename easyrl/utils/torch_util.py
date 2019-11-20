import torch


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


def torch_to_numpy(tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('tensor has to be a torch tensor!')
    return tensor.cpu().detach().numpy()