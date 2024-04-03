import torch

def VectorStep(network, grad_vector, eta):
    """
    network: nn.Module
    grad_vector: (n, 1) torch.tensor
    eta: scalar, step size
    """
    with torch.no_grad():
        for param in network.parameters():
            numel = param.numel() 
            param_update = grad_vector[index:index + numel].reshape(param.shape)
            param -= eta * param_update 
            index += numel


def VectorizedGrad(grad):
    """
    grad = torch.autograd.grad(loss, network.parameters())
    """
    with torch.no_grad():
        return torch.cat([g.view(-1) for g in grad])

def OuterProdGradMatrix(trajectories, rewards_y, policy_x, policy_y):
    # TODO: get correct trajectorie probs on which we compute the grad
    M = torch.zeros(TODO, TODO)
    for (t, r_y) in (trajectories, rewards_y):
        grad_x = VectorizedGrad( torch.autograd.grad(t, policy_x.parameters) )
        grad_y = VectorizedGrad( torch.autograd.grad(t, policy_y.parameters) )
        M = M + torch.outer(grad_x, grad_y)
    return M

def GetPolicyYstar(policy_y):
    policy_y = policy_y.deepcopy()
    # compute normal policy gradient update on policy_y
    return policy_y

def YStarGrad(policy_y_star, trajectory):
    grad = # get grad by regular policy gradient way
    grad = VectorizedGrad(grad)
    return grad


def OpponentLearningGrad(OuterProdGradMatrix, ):