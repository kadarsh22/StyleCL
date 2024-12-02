
import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from fvcore.nn import FlopCountAnalysis

import legacy
from metrics import metric_main
import wandb


# ----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)


# ----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size, cur_tick, wandb_log=False):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)
    if wandb_log:
        wandb.log({"Generated Image": wandb.Image(PIL.Image.fromarray(img, 'RGB'))})


# ----------------------------------------------------------------------------

def training_loop(
        run_dir='.',  # Output directory.
        training_set_kwargs={},  # Options for training set.
        data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        Adaptor_kwargs={},  # Options for Adaptor network
        G_opt_kwargs={},  # Options for generator optimizer.
        D_opt_kwargs={},  # Options for discriminator optimizer.
        augment_kwargs=None,  # Options for augmentation pipeline. None = disable.
        loss_kwargs={},  # Options for loss function.
        metrics=[],  # Metrics to evaluate during training.
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus[.
        batch_size=4,  # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu=4,  # Number of samples processed at a time by one GPU.
        codebook_size=16,
        ema_kimg=10,  # Half-life of the exponential moving average (EMA) of generator weights.
        ema_rampup=None,  # EMA ramp-up coefficient.
        G_reg_interval=4,  # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval=16,  # How often to perform regularization for D? None = disable lazy regularization.
        augment_p=0,  # Initial value of augmentation probability.
        ada_target=None,  # ADA target value. None = fixed p.
        ada_interval=4,  # How often to perform ADA adjustment?
        ada_kimg=500,
        # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
        total_kimg=25000,  # Total length of the training, measured in thousands of real images.
        kimg_per_tick=4,  # Progress snapshot interval.
        image_snapshot_ticks=50,  # How often to save image snapshots? None = disable.
        network_snapshot_ticks=50,  # How often to save network snapshots? None = disable.
        resume_pkl=None,  # Network pickle to resume training from.
        cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
        allow_tf32=False,  # Enable torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32?
        abort_fn=None,
        # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn=None,  # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32  # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.

    # Initialise WANDB
    if rank == 0:
        wandb.login(key='please paste your wandb key here')
        wandb.init(project="stylecl")
        wandb.run.log_code(".")
    # Load training set.
    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)  # subclass of training.dataset.Dataset
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus,
                                                seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler,
                                                             batch_size=batch_size // num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()

    # Construct networks.
    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=3)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(
        device)  # subclass of torch.nn.Module
    adaptor = dnnlib.util.construct_class_by_name(**Adaptor_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()
    adaptor_ema = copy.deepcopy(adaptor).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('D', D)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

        with dnnlib.util.open_url('pretrained_models/lsun_cats/lsun_cats_adaptor.pkl') as f:
            pretrained_adaptor = legacy.load_network_pkl(f)
        for name, module in [('adaptor', adaptor), ('G', G), ]:
            misc.copy_params_and_buffers(pretrained_adaptor[name], module, require_all=False)
        print("Printing Differing Tensors")
        misc.print_differing_tensor(resume_data['G_ema'], G)

    # Print network summary tables.
    if rank == 0:
        x = torch.empty([batch_gpu, G.num_ws, codebook_size], device=device)
        ws, num_adaptor_params, num_adaptor_buffers = misc.print_module_summary(adaptor, [x])
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        img, num_generator_params, num_generator_buffers = misc.print_module_summary(G, [ws])
        misc.print_module_summary(D, [img, c])

    # Print Flops Information
    if rank == 0:
        flops_adaptor = FlopCountAnalysis(adaptor, x)
        print("Flops Adaptor : ")
        print(flops_adaptor.by_module())
        flops_generator = FlopCountAnalysis(G, ws)
        print("Flops Generator : ")
        print(flops_generator.by_module())

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(
            device)  # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()
    for name, module in [('adaptor', adaptor), ('G_synthesis', G.synthesis), ('D', D), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, **ddp_modules,
                                               **loss_kwargs)  # subclass of training.loss.Loss
    phases = []
    for name, module, opt_kwargs, reg_interval in [('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(),
                                                      **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + 'both', module=module, opt=opt, interval=1)]
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(),
                                                      **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + 'main', module=[module], opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name + 'reg', module=[module], opt=opt, interval=reg_interval)]

    name = 'G'
    opt_kwargs = G_opt_kwargs
    named_parameters_list = list(G.named_parameters())
    learnable_g_params = ['skip', 'skip_strength']
    params_list = []
    total_learnable_g_params, total_learnable_g_buffers = 0, 0

    print("Learnable Parameters : ")
    for param_name, parameter in named_parameters_list:
        network_type = param_name.split('.')[2]
        try:
            module_type = param_name.split('.')[3]
        except IndexError:
            module_type = 'Not exist'
        if network_type in learnable_g_params or module_type == 'noise_strength':
            print(param_name)
            params_list.append(parameter)
            total_learnable_g_params = total_learnable_g_params + parameter.numel()

    print("Learnable Buffers : ")
    for param_name, parameter in list(G.named_buffers()):
        network_type = param_name.split('.')[2]
        try:
            module_type = param_name.split('.')[3]
        except IndexError:
            module_type = 'Not exist'
        if network_type in learnable_g_params or module_type == 'noise_const':
            print(param_name)
            params_list.append(parameter)
            total_learnable_g_buffers = total_learnable_g_buffers + parameter.numel()

    if rank == 0:
        total_task_specific_generator_params = total_learnable_g_params + total_learnable_g_buffers
        total_task_specific_adaptor_params = num_adaptor_params + num_adaptor_buffers
        print("Task Specific G parameters : " + str(total_learnable_g_params))
        print("Task Specific G buffers : " + str(total_learnable_g_buffers))
        print("Total Task Specific Generator parameters : " + str(total_task_specific_generator_params))
        print("Total Task Specific Adaptor parameters : " + str(total_task_specific_adaptor_params))
        print("Total Task Specific Parameters : " + str(
            total_task_specific_generator_params + total_task_specific_adaptor_params))

    params = params_list + list(adaptor.parameters())
    reg_interval = G_reg_interval
    mb_ratio = reg_interval / (reg_interval + 1)
    opt_kwargs = dnnlib.EasyDict(opt_kwargs)
    opt_kwargs.lr = opt_kwargs.lr * mb_ratio
    opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
    opt = dnnlib.util.construct_class_by_name(params, **opt_kwargs)  # subclass of torch.optim.Optimizer
    phases += [dnnlib.EasyDict(name=name + 'main', module=[adaptor, G.synthesis], opt=opt, interval=1)]
    phases += [dnnlib.EasyDict(name=name + 'reg', module=[adaptor, G.synthesis], opt=opt, interval=reg_interval)]

    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images.
    grid_size = None
    grid_x = None
    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0, 255], grid_size=grid_size,
                        cur_tick="real_samples", wandb_log=False)
        grid_x = torch.randn(labels.shape[0], G.num_ws, codebook_size).to(device)
        w_s = adaptor(grid_x).split(batch_gpu)
        images = torch.cat([G.synthesis(w).cpu() for w in w_s]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1, 1], grid_size=grid_size,
                        cur_tick="generated_samples_init", wandb_log=True)

    # Initialize logs.
    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:

        # Fetch training data.
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            if phase_real_img.shape[1] != 3:
                phase_real_img = phase_real_img.repeat(1, 3, 1, 1)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_x = torch.randn(len(phases) * batch_size, G.num_ws, codebook_size).to(device)
            all_gen_x = [phase_gen_x.split(batch_gpu) for phase_gen_x in all_gen_x.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in
                         range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        # Execute training phases.
        for phase, phase_gen_x, phase_gen_c in zip(phases, all_gen_x, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)

            for module_ in phase.module:
                module_.zero_grad()
                module_.requires_grad_(True)

            # Accumulate gradients over multiple rounds.
            for round_idx, (real_img, real_c, gen_x, gen_c) in enumerate(
                    zip(phase_real_img, phase_real_c, phase_gen_x, phase_gen_c)):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_x=gen_x, gen_c=gen_c,
                                          sync=sync, gain=gain)

            # Update weights.
            for module_ in phase.module:
                module_.requires_grad_(False)

            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for module_ in phase.module:
                    for param in module_.parameters():
                        if param.grad is not None:
                            misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            for (param_name, p_ema), p_src in zip(G_ema.named_parameters(), G.parameters()):
                network_type = param_name.split('.')[2]
                try:
                    module_type = param_name.split('.')[3]
                except IndexError:
                    module_type = 'Not exist'
                if network_type in learnable_g_params or module_type == 'noise_strength':
                    p_ema.copy_(0.999 * p_ema + (1. - 0.999) * p_src)
            for (param_name, b_ema), b in zip(G_ema.named_buffers(), G.buffers()):
                network_type = param_name.split('.')[2]
                try:
                    module_type = param_name.split('.')[3]
                except IndexError:
                    module_type = 'Not exist'
                if network_type in learnable_g_params or module_type == 'noise_const':
                    b_ema.copy_(b)

            # moving average of adaptor
            for p_ema, p_src in zip(adaptor_ema.parameters(), adaptor.parameters()):
                p_ema.copy_(0.999 * p_ema + (1. - 0.999) * p_src)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (
                        ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [
            f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2 ** 30):<6.2f}"]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2 ** 30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [
            f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.

        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            adaptor.eval()
            G.eval()
            with torch.no_grad():
                w_s = adaptor(grid_x).split(batch_gpu)
                images = torch.cat([G.synthesis(w).cpu() for w in w_s]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg // 1000:06d}.png'), drange=[-1, 1],
                            grid_size=grid_size, cur_tick=cur_tick, wandb_log=True)
            adaptor.train()
            G.train()

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(training_set_kwargs=dict(training_set_kwargs))
            for name, module in [('adaptor', adaptor), ('D', D), ('G', G), ('G_ema', G_ema), ('adaptor_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module  # conserve memory
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg // 1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=snapshot_data['G'],
                                                      adaptor=snapshot_data["adaptor"],
                                                      dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank,
                                                      device=device)
                if rank == 0:
                    if result_dict["metric"] == "fid50k_full":
                        fid_log = {"fid": result_dict["results"]["fid50k_full"]}
                        wandb.log(fid_log)
                    elif result_dict["metric"] == "pr50k3_full":
                        precision_recall_log = {"precision": result_dict["results"]["pr50k3_full_precision"],
                                                "recall": result_dict["results"]["pr50k3_full_recall"]}
                        wandb.log(precision_recall_log)
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)
                stats_metrics.update(result_dict.results)
        del snapshot_data  # conserve memory

        # Collect statistics.
        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if rank == 0:
            wandb.log({"fake_score": stats_dict["Loss/scores/fake"]["mean"],
                       "G loss": stats_dict["Loss/G/loss"]["mean"],
                       "path_length penalty": stats_dict["Loss/pl_penalty"]["mean"],
                       "G regularisation": stats_dict["Loss/pl_penalty"]["mean"],
                       "real_score": stats_dict["Loss/scores/real"]["mean"],
                       "D Loss": stats_dict["Loss/D/loss"]["mean"],
                       "R1 Penalty": stats_dict["Loss/r1_penalty"]["mean"],
                       "D regularisation": stats_dict["Loss/D/reg"]["mean"]
                       })
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

# ----------------------------------------------------------------------------
