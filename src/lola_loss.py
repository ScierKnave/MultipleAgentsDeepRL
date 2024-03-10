import torch
import copy 


def lola_other_loss(actions, other_actions, other_rewards, policy, other_policy):
    log_probs = policy.log_prob(actions) * \
                other_policy.log_prob(other_actions)
    loss = -(log_probs * other_rewards).sum()
    return loss


def lola_imaginary_loss(actions, 
                        other_actions, 
                        rewards, 
                        other_rewards, 
                        policy, 
                        other_policy,
                        other_optimizer,
                        other_lr
                        ):
    """
    Multi-policy gradient loss but where the first policy imagines the other performing a naive step
    with its current parameters. A little mind-bending on the mathematical side.

    Parameters:
    - actions (B, T, A): actions of policy that is being optimized here
    - other_actions (B, T, A): actions of the other policy
    - rewards (B, T): rewards of of policy that is being optimized here
    - other_rewards (B, T)
    - policy: 
    - other_policy: The other policy in the multi-agent setting.
    - lr: learning rate

    Returns:
    - policy gradient loss that differentiates through the update of the other
    """

    # Compute imaginary naive update on second policy (that depends on 
    imaginary_policy = copy.deepcopy(other_policy)
    imaginary_loss = lola_other_loss(actions, 
                                        other_actions, 
                                        other_rewards, 
                                        policy, 
                                        other_policy)
    imaginary_loss.backward(retain_graph=True) # retain graph is what makes it second differentiable
    imaginary_optimizer = other_optimizer(imaginary_policy, learning_rate=other_lr)
    imaginary_optimizer.step()

    # Differentiate through V1(theta_1, theta_2 + lr * theta_2_grad(theta_1))
    log_probs = policy.log_prob(actions) * \
                imaginary_policy.log_prob(other_actions)
    loss = -(log_probs * rewards).sum()

    return loss

    
