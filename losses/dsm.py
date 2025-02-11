import torch
from tqdm import tqdm

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

def train_bilevel(score, optimizer, dataloader, config, data_transform, sigmas, adv_perturb, idx_bilevel):
    all_epochs = 70
    for epoch in range(all_epochs):
        train_bar = tqdm(dataloader)
        count = 0
        loss_sum = 0
        for i, (X, y, idx) in enumerate(train_bar):

            adv_perturb_numpy = adv_perturb[idx]
            noise = torch.tensor(adv_perturb_numpy).to(config.device).float()

            score.train()

            X = X.to(config.device)
            X = data_transform(config, X)

            loss = anneal_dsm_score_estimation(score, X + noise, sigmas, None,
                                                config.training.anneal_power,
                                                None)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            count += 1

            train_bar.set_description("Bilevel_training Round {} Epoch[{}/{}] Loss:{:.4f}".format(idx_bilevel, epoch, all_epochs, loss_sum / float(count)))
