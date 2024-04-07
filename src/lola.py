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


def get_lola_matrix(trajectories, policy_x, policy_y):
    # TODO: get correct trajectorie probs on which we compute the grad
    M = torch.zeros(TODO, TODO)
    for trajectory in trajectories:
        grad_x = VectorizedGrad(torch.autograd.grad(t, policy_x.parameters) )
        grad_y = VectorizedGrad(torch.autograd.grad(t, policy_y.parameters) )
        M = M + torch.outer(grad_x, grad_y)
    return M / n

def do_regular_pg_update(trajectories, policy, optimizer, off_policy_adjuster=None):
    loss = 0
    for trajectory in trajectories:
        states, rewards = trajectory
        if off_policy_adjuster is not None:
            rewards = rewards * off_policy_adjuster(states)
        action_probs = policy(states)
        action_distribution = torch.distributions.Categorical(action_probs)
        log_probs = action_distribution.log_prob()
        loss += -(log_probs * rewards).sum()
    loss /= len(trajectories)
    loss.backward()
    optimizer.step()

class off_policy_adjuster:
    def __init__(self, policy_num, policy_den):
        self.policy_num = policy_num
        self.policy_den + policy_den
    def self(self, states):
        with torch.no_grad:
            return self.policy_num(states) / self.policy_den(states)
    

def lola_pg_step(trajectories, policy_x, policy_y):
    M = get_lola_matrix(trajectories, policy_x, policy_y)
    for trajectory in trajectories:


