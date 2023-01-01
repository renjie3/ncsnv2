import torch

def anneal_dsm_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None):
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)

def anneal_dsm_score_estimation_given_sigmas_noise(scorenet, samples, sigmas, labels, used_sigmas, random_noise, anneal_power=2., hook=None):
    # if labels is None:
    #     labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    # used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = random_noise * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)

def zero_grad_no_detach(model_params):
    for param in model_params:
        # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
        if param.grad is not None:
            param.grad.zero_()

def anneal_dsm_score_estimation_target_gradient(scorenet, samples, sigmas, labels, used_sigmas, random_noise, anneal_power=2., hook=None, differentiable_params=None):
    zero_grad_no_detach(differentiable_params)

    noise = random_noise * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = (1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power).mean(dim=0)

    # if hook is not None:
    #     hook(loss, labels)

    grad = torch.autograd.grad(loss, differentiable_params)

    return grad


def anneal_dsm_score_estimation_poison_gradient(scorenet, samples, sigmas, labels, used_sigmas, random_noise, anneal_power=2., hook=None, differentiable_params=None):
    zero_grad_no_detach(differentiable_params)

    noise = random_noise * used_sigmas
    perturbed_samples = samples + noise
    target = - 1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = (1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power).mean(dim=0)

    # if hook is not None:
    #     hook(loss, labels)

    grad = torch.autograd.grad(loss, differentiable_params, retain_graph=True, create_graph=True)

    return grad
