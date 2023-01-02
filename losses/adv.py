import torch
from losses.dsm import anneal_dsm_score_estimation_given_sigmas_noise, anneal_dsm_score_estimation_target_gradient, anneal_dsm_score_estimation_poison_gradient
from tqdm import tqdm

def single_level(sigmas, X, score, args, config, _idx, dataloader, init_noise):
    # labels = torch.randint(0, len(sigmas), (x_adv.shape[0],), device=x_adv.device)
    # used_sigmas = sigmas[labels].view(x_adv.shape[0], *([1] * len(x_adv.shape[1:])))
    # random_noise = torch.randn_like(x_adv)

    eot_gaussian_num = config.adv.eot_gaussian_num
    t_seg_num = config.adv.t_seg_num
    adv_alpha = float(config.adv.adv_alpha) / 255.0
    adv_epsilon = float(config.adv.adv_epsilon) / 255.0
    img_max = 1.0
    img_min = 0.0
    if config.data.rescaled:
        adv_alpha *= 2
        adv_epsilon *= 2
        img_min = -1.0

    seg = len(sigmas) // t_seg_num
    all_labels = []
    for i in range(t_seg_num):
        for _ in range(eot_gaussian_num):
            labels = torch.randint(i * seg, (i + 1) * seg, (X.shape[0],), device=X.device)
            all_labels.append(labels)
    all_labels = torch.stack(all_labels, dim=0).view(t_seg_num, eot_gaussian_num, X.shape[0])
    all_gaussian_noise = torch.randn([t_seg_num, eot_gaussian_num, *X.shape]).to(X.device)

    x_adv = X.detach().float() + init_noise.detach() # TODO: bilevel has to use X + noise as init
    adv_step_loop_bar = tqdm(range(config.adv.adv_step))
    for _ in adv_step_loop_bar:
        adv_step_loop_bar.set_description("Batch [{}/{}]".format(_idx, len(dataloader) // 3)) # // 3 because we have 3 class, and we only train adv on bird.
        x_adv.requires_grad_()
        accumulated_grad = 0
        accumulated_loss = 0
        for i in range(t_seg_num):
            for j in range(eot_gaussian_num):
                # print('t_seg_num: ', i, 'eot_gaussian_num: ', j)
                labels = all_labels[i, j]
                random_noise = all_gaussian_noise[i, j]
                used_sigmas = sigmas[labels].view(x_adv.shape[0], *([1] * len(x_adv.shape[1:])))
                with torch.enable_grad():
                    # if self.random_noise_every_adv_step:
                    #     random_noise = torch.randn([*X.shape]).to(dist_util.dev())
                    if args.adv_loss_type in ["min_forward_loss", "bilevel_min_forward_loss", ]:
                        loss = anneal_dsm_score_estimation_given_sigmas_noise(score, x_adv, sigmas, labels=labels, used_sigmas=used_sigmas, random_noise=random_noise, anneal_power=config.training.anneal_power)
                    elif args.adv_loss_type in ["max_forward_loss", "bilevel_max_forward_loss", ]:
                        loss = - anneal_dsm_score_estimation_given_sigmas_noise(score, x_adv, sigmas, labels=labels, used_sigmas=used_sigmas, random_noise=random_noise, anneal_power=config.training.anneal_power)
                    grad = torch.autograd.grad(loss, [x_adv])[0]
                    accumulated_grad += grad
                # print(loss.item())
                accumulated_loss += loss.item()
        print("accumulated_loss:", accumulated_loss)
                
        x_adv = x_adv.detach() - adv_alpha * torch.sign(accumulated_grad.detach())
        x_adv = torch.min(torch.max(x_adv, X - adv_epsilon), X + adv_epsilon)
        x_adv = torch.clamp(x_adv, img_min, img_max)

    return x_adv

def gradient_matching_loss(source_grad, poison_grad, adv_loss_type="gradient_matching"):
        if adv_loss_type == "gradient_matching":

            gradient_count = len(source_grad)

            indices = torch.arange(gradient_count)
            gm_loss = 0
            debug_sum0_count = 0

            for i in indices:
                # print(th.sum(poison_grad[i]))
                # if torch.sum(poison_grad[i]) == 0:
                #     debug_sum0_count += 1
                # else:
                #     print(poison_grad[i])
                # print(source_grad[i].shape)
                # input("check")
                gm_loss += torch.nn.functional.cosine_similarity(source_grad[i].flatten().detach(), poison_grad[i].flatten(), dim=0)
            
            return gm_loss / float(gradient_count)

def check_diff_sigma_gradient(sigmas, X, score, args, config, _idx, dataloader):

    differentiable_params = [p for p in score.parameters() if p.requires_grad]

    eot_gaussian_num = config.adv.eot_gaussian_num
    t_seg_num = config.adv.t_seg_num
    if config.data.rescaled:
        adv_alpha *= 2
        adv_epsilon *= 2
        img_min = -1.0

    seg = len(sigmas) // t_seg_num
    all_labels = []
    for i in range(t_seg_num):
        for _ in range(eot_gaussian_num):
            labels = torch.randint(i * seg, (i + 1) * seg, (X.shape[0],), device=X.device)
            all_labels.append(labels)
    all_labels = torch.stack(all_labels, dim=0).view(t_seg_num, eot_gaussian_num, X.shape[0])
    all_gaussian_noise = torch.randn([t_seg_num, eot_gaussian_num, *X.shape]).to(X.device)

    for j in range(eot_gaussian_num):
        accumulated_loss = 0
        count = 0

        source_i = 6

        i = source_i
        labels = all_labels[i, j]
        random_noise = all_gaussian_noise[0, j]
        used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))
        with torch.enable_grad():
            loss = anneal_dsm_score_estimation_given_sigmas_noise(score, X, sigmas, labels=labels, used_sigmas=used_sigmas, random_noise=random_noise, anneal_power=config.training.anneal_power)
            source_grad = torch.autograd.grad(loss, differentiable_params)

        for i in range(t_seg_num):
            # print('t_seg_num: ', i, 'eot_gaussian_num: ', j)
            labels = all_labels[i, j]
            random_noise = all_gaussian_noise[0, j]
            used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:])))
            with torch.enable_grad():
                loss = anneal_dsm_score_estimation_given_sigmas_noise(score, X, sigmas, labels=labels, used_sigmas=used_sigmas, random_noise=random_noise, anneal_power=config.training.anneal_power)
                grad = torch.autograd.grad(loss, differentiable_params)
                if i == source_i:
                    source_grad = grad
                else:
                    gm_loss = gradient_matching_loss(source_grad, grad)
                    accumulated_loss += gm_loss.item()
                    print(i, "gm_loss.item():", gm_loss.item())
                    count += 1
        print(accumulated_loss / float(count))
    input("check 000")

    return


def gradient_matching(sigmas, X, target_X, score, args, config, _idx, dataloader):

    differentiable_params = [p for p in score.parameters() if p.requires_grad]

    eot_gaussian_num = config.adv.eot_gaussian_num
    t_seg_num = config.adv.t_seg_num
    adv_alpha = float(config.adv.adv_alpha) / 255.0
    adv_epsilon = float(config.adv.adv_epsilon) / 255.0
    img_max = 1.0
    img_min = 0.0
    if config.data.rescaled:
        adv_alpha *= 2
        adv_epsilon *= 2
        img_min = -1.0

    seg = len(sigmas) // t_seg_num
    all_labels = []
    for i in range(t_seg_num):
        for _ in range(eot_gaussian_num):
            labels = torch.randint(i * seg, (i + 1) * seg, (X.shape[0],), device=X.device)
            all_labels.append(labels)
    all_labels = torch.stack(all_labels, dim=0).view(t_seg_num, eot_gaussian_num, X.shape[0])
    all_gaussian_noise = torch.randn([t_seg_num, eot_gaussian_num, *X.shape]).to(X.device)

    x_adv = X.detach().float() # TODO: bilevel has to use X + noise as init
    adv_step_loop_bar = tqdm(range(config.adv.adv_step))
    for _ in adv_step_loop_bar:
        adv_step_loop_bar.set_description("Batch [{}/{}]".format(_idx, len(dataloader) // 3)) # // 3 because we have 3 class, and we only train adv on bird.
        accumulated_grad = 0
        accumulated_loss = 0
        accumulated_count = 0
        for i in range(t_seg_num):
            for j in range(eot_gaussian_num):
                print('t_seg_num: ', i, 'eot_gaussian_num: ', j)
                x_adv.requires_grad_()
                labels = all_labels[i, j]
                random_noise = all_gaussian_noise[i, j]
                used_sigmas = sigmas[labels].view(x_adv.shape[0], *([1] * len(x_adv.shape[1:])))
                with torch.enable_grad():
                    # if self.random_noise_every_adv_step:
                    #     random_noise = torch.randn([*X.shape]).to(dist_util.dev())
                    if args.adv_loss_type == "gradient_matching":
                        target_gradient = anneal_dsm_score_estimation_target_gradient(score, target_X, sigmas, labels=labels, used_sigmas=used_sigmas, random_noise=random_noise, anneal_power=config.training.anneal_power, differentiable_params=differentiable_params)
                    else:
                        raise("wrong adv_loss_type!")
                    poison_gradient = anneal_dsm_score_estimation_poison_gradient(score, x_adv, sigmas, labels=labels, used_sigmas=used_sigmas, random_noise=random_noise, anneal_power=config.training.anneal_power, differentiable_params=differentiable_params)

                    one_step_gm_loss = gradient_matching_loss(target_gradient, poison_gradient)

                    one_step_gm_loss.backward()
                    accumulated_loss += one_step_gm_loss.item()
                    accumulated_count += 1

                    if config.adv.optim_mode == "pgd":
                        accumulated_grad += x_adv.grad.detach()
                        # x_adv = x_adv.detach() + adv_alpha * torch.sign(x_adv.grad.detach())

                    # x_adv.data = torch.min(torch.max(x_adv.data, X - adv_epsilon), X + adv_epsilon)
                    # x_adv.data = torch.clamp(x_adv.data, img_min, img_max)

                # print(loss.item())
                # accumulated_loss += loss.item()
        print("accumulated_loss:", accumulated_loss / float(accumulated_count))
                
        x_adv = x_adv.detach() + adv_alpha * torch.sign(accumulated_grad.detach())
        x_adv = torch.min(torch.max(x_adv, X - adv_epsilon), X + adv_epsilon)
        x_adv = torch.clamp(x_adv, img_min, img_max)

    return x_adv
