import torch
import copy

def do_vector_step(network, grad_vector, eta):
    """
    network: nn.Module
    grad_vector: (n, 1) torch.tensor
    eta: scalar, step size
    """
    index = 0
    with torch.no_grad():
        for param in network.parameters():
            numel = param.numel()
            param_update = grad_vector[index:index + numel].reshape(param.shape)
            param -= eta * param_update
            index += numel

def torchgrad_to_vect(grad):
    """
    grad = torch.autograd.grad(loss, network.parameters())
    """
    with torch.no_grad():
        return torch.cat([g.view(-1) for g in grad])


def get_lola_matrix(states, actions_x, actions_y, rewards_y, policy_x, policy_y):
    # TODO: get correct trajectorie probs on which we compute the grad
    M = None
    n = len(states) # number of trajectories
    # For every trajectory
    for states, actions_x, actions_y, rewards_y in zip(states, actions_x, actions_y, rewards_y):
        # get log(policy_x(T)) for every T
        action_dist = policy_x(states)
        log_probs_x = action_dist.log_prob(actions_x).mean()
        # get log(policy_y(T)) for every T
        action_dist = policy_y(states)
        log_probs_y = action_dist.log_prob(actions_y).mean()

        # Get gradients in vector form
        grad_x = torchgrad_to_vect( torch.autograd.grad(log_probs_x, policy_x.parameters()) )
        grad_y = torchgrad_to_vect( torch.autograd.grad(log_probs_y, policy_y.parameters()) )
        if M is not None: M += rewards_y.sum() * torch.outer(grad_x, grad_y)
        else: M = rewards_y.sum() * torch.outer(grad_x, grad_y)

    return M / n

def get_regular_pg_loss(states, actions, rewards, policy, off_policy_adjuster=None):
    loss = 0
    for (states, actions, rewards) in zip(states, actions, rewards): # for every trajectory
        if off_policy_adjuster is not None:
            rewards = rewards * off_policy_adjuster(states, actions)
        action_dist = policy(states)
        log_probs = action_dist.log_prob(actions)
        loss += -(log_probs * rewards).sum()
    loss /= len(rewards) # get mean
    return loss


class off_policy_adjuster:
    def __init__(self, policy_num, policy_den):
        self.policy_num = policy_num
        self.policy_den = policy_den
    def __call__(self, states, actions):

        # Obtain the probabilities from both policies
        probs_num = self.policy_num(states).probs
        probs_den = self.policy_den(states).probs

        # Gather the probabilities corresponding to the taken actions
        pi_num = probs_num.gather(1, actions.unsqueeze(1)).squeeze()
        pi_den = probs_den.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute weights (avoiding division by zero)
        with torch.no_grad():
            weights = pi_num / (pi_den + 1e-10)  # Adding a small epsilon to avoid division by zero

        return weights

        # with torch.no_grad():
        #     return self.policy_num(states) / self.policy_den(states)


def lola_pg_step(trajectories, policy_x, policy_y, lr_x, lr_y):

    # Get the lola matrix. Gives info about how y will change with respect to x
    lola_matrix = get_lola_matrix(
        trajectories['states'],
        trajectories['actions_x'],
        trajectories['actions_y'],
        trajectories['rewards_y'],
        policy_x, policy_y)

    # Get anticipatory y policy
    # policy_y_star = policy_y.deepcopy()
    policy_y_star = copy.deepcopy(policy_y)
    optimizer = torch.optim.AdamW(policy_y_star.parameters(), lr=lr_y)
    loss = get_regular_pg_loss(
        trajectories['states'],
        trajectories['actions_y'],
        trajectories['rewards_y'],
        policy_y_star)
    loss.backward()
    optimizer.step()

    # Get the LOLA gradient
    off_pol_adj = off_policy_adjuster(policy_y_star, policy_y) # importance sampling without gradient
    pg_y_star_loss = get_regular_pg_loss(
        trajectories['states'],
        trajectories['actions_y'],
        trajectories['rewards_x'],
        policy_y_star,
        off_pol_adj)
    grad_y_star = torch.autograd.grad(pg_y_star_loss, policy_y_star.parameters())
    grad_lola = torch.matmul(lola_matrix, torchgrad_to_vect(grad_y_star))

    # Get anticipatory grad
    grad_ap_loss = get_regular_pg_loss(
        trajectories['states'],
        trajectories['actions_x'],
        trajectories['rewards_x'],
        policy_x,
        off_pol_adj)

    grad_ap = torchgrad_to_vect(torch.autograd.grad(grad_ap_loss, policy_x.parameters()))

    # Get the full gradient and update policy x
    grad = grad_lola + grad_ap
    do_vector_step(policy_x, grad, lr_x)
