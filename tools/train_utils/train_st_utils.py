import torch
import os
import glob
import tqdm
from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils
from pcdet.utils import self_training_utils
from pcdet.config import cfg
from pcdet.models.model_utils.dsnorm import set_ds_source, set_ds_target

from .train_utils import save_checkpoint, checkpoint_state


def train_one_epoch_st(model, optimizer, source_reader, target_loader, model_func, lr_scheduler,
                       accumulated_iter, optim_cfg, rank, tbar, total_it_each_epoch,
                       dataloader_iter, tb_log=None, leave_pbar=False, ema_model=None, cur_epoch=None):
    if total_it_each_epoch == len(target_loader):
        dataloader_iter = iter(target_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    ps_bbox_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    ign_ps_bbox_nmeter = common_utils.NAverageMeter(len(cfg.CLASS_NAMES))
    loss_meter = common_utils.AverageMeter()
    st_loss_meter = common_utils.AverageMeter()

    disp_dict = {}

    for cur_it in range(total_it_each_epoch):
        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()

        optimizer.zero_grad()
        if cfg.SELF_TRAIN.SRC.USE_DATA:
            # forward source data with labels
            source_batch = source_reader.read_data()

            if cfg.SELF_TRAIN.get('DSNORM', None):
                model.apply(set_ds_source)

            if cfg.SELF_TRAIN.SRC.get('SEP_LOSS_WEIGHTS', None):
                source_batch['SEP_LOSS_WEIGHTS'] = cfg.SELF_TRAIN.SRC.SEP_LOSS_WEIGHTS

            loss, tb_dict, disp_dict = model_func(model, source_batch)
            loss = cfg.SELF_TRAIN.SRC.get('LOSS_WEIGHT', 1.0) * loss
            loss.backward()
            loss_meter.update(loss.item())
            disp_dict.update({'loss': "{:.3f}({:.3f})".format(loss_meter.val, loss_meter.avg)})

            if not cfg.SELF_TRAIN.SRC.get('USE_GRAD', None):
                optimizer.zero_grad()

        if cfg.SELF_TRAIN.TAR.USE_DATA:
            try:
                target_batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(target_loader)
                target_batch = next(dataloader_iter)
                print('new iters')

            if cfg.SELF_TRAIN.get('DSNORM', None):
                model.apply(set_ds_target)

            if cfg.SELF_TRAIN.TAR.get('SEP_LOSS_WEIGHTS', None):
                target_batch['SEP_LOSS_WEIGHTS'] = cfg.SELF_TRAIN.TAR.SEP_LOSS_WEIGHTS

            # parameters for save pseudo label on the fly
            st_loss, st_tb_dict, st_disp_dict = model_func(model, target_batch)
            st_loss = cfg.SELF_TRAIN.TAR.get('LOSS_WEIGHT', 1.0) * st_loss
            st_loss.backward()
            st_loss_meter.update(st_loss.item())

            # count number of used ps bboxes in this batch
            pos_pseudo_bbox = target_batch['pos_ps_bbox'].mean(dim=0).cpu().numpy()
            ign_pseudo_bbox = target_batch['ign_ps_bbox'].mean(dim=0).cpu().numpy()
            ps_bbox_nmeter.update(pos_pseudo_bbox.tolist())
            ign_ps_bbox_nmeter.update(ign_pseudo_bbox.tolist())
            pos_ps_result = ps_bbox_nmeter.aggregate_result()
            ign_ps_result = ign_ps_bbox_nmeter.aggregate_result()

            st_tb_dict = common_utils.add_prefix_to_dict(st_tb_dict, 'st_')
            disp_dict.update(common_utils.add_prefix_to_dict(st_disp_dict, 'st_'))
            disp_dict.update({'st_loss': "{:.3f}({:.3f})".format(st_loss_meter.val, st_loss_meter.avg),
                              'pos_ps_box': pos_ps_result,
                              'ign_ps_box': ign_ps_result})

        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()
        accumulated_iter += 1

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter, pos_ps_box=pos_ps_result,
                                  ign_ps_box=ign_ps_result))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                if cfg.SELF_TRAIN.SRC.USE_DATA:
                    tb_log.add_scalar('train/loss', loss, accumulated_iter)
                    for key, val in tb_dict.items():
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)
                if cfg.SELF_TRAIN.TAR.USE_DATA:
                    tb_log.add_scalar('train/st_loss', st_loss, accumulated_iter)
                    for key, val in st_tb_dict.items():
                        tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
        for i, class_names in enumerate(target_loader.dataset.class_names):
            tb_log.add_scalar(
                'ps_box/pos_%s' % class_names, ps_bbox_nmeter.meters[i].avg, cur_epoch)
            tb_log.add_scalar(
                'ps_box/ign_%s' % class_names, ign_ps_bbox_nmeter.meters[i].avg, cur_epoch)

    return accumulated_iter


def train_model_st(model, optimizer, source_loader, target_loader, model_func, lr_scheduler, optim_cfg,
                   start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, ps_label_dir,
                   source_sampler=None, target_sampler=None, lr_warmup_scheduler=None, ckpt_save_interval=1,
                   max_ckpt_save_num=50, merge_all_iters_to_one_epoch=False, logger=None, ema_model=None):
    accumulated_iter = start_iter
    source_reader = common_utils.DataReader(source_loader, source_sampler)
    source_reader.construct_iter()

    # for continue training.
    # if already exist generated pseudo label result
    ps_pkl = self_training_utils.check_already_exsit_pseudo_label(ps_label_dir, start_epoch)
    if ps_pkl is not None:
        logger.info('==> Loading pseudo labels from {}'.format(ps_pkl))

    # for continue training
    if cfg.SELF_TRAIN.get('PROG_AUG', None) and cfg.SELF_TRAIN.PROG_AUG.ENABLED and \
        start_epoch > 0:
        for cur_epoch in range(start_epoch):
            if cur_epoch in cfg.SELF_TRAIN.PROG_AUG.UPDATE_AUG:
                target_loader.dataset.data_augmentor.re_prepare(
                    augmentor_configs=None, intensity=cfg.SELF_TRAIN.PROG_AUG.SCALE)

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True,
                     leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(target_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(target_loader.dataset, 'merge_all_iters_to_one_epoch')
            target_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(target_loader) // max(total_epochs, 1)

        dataloader_iter = iter(target_loader)
        for cur_epoch in tbar:
            if target_sampler is not None:
                target_sampler.set_epoch(cur_epoch)
                source_reader.set_cur_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # update pseudo label
            if (cur_epoch in cfg.SELF_TRAIN.UPDATE_PSEUDO_LABEL) or \
                    ((cur_epoch % cfg.SELF_TRAIN.UPDATE_PSEUDO_LABEL_INTERVAL == 0)
                     and cur_epoch != 0):
                target_loader.dataset.eval()
                self_training_utils.save_pseudo_label_epoch(
                    model, target_loader, rank,
                    leave_pbar=True, ps_label_dir=ps_label_dir, cur_epoch=cur_epoch
                )
                target_loader.dataset.train()
            
            # curriculum data augmentation
            if cfg.SELF_TRAIN.get('PROG_AUG', None) and cfg.SELF_TRAIN.PROG_AUG.ENABLED and \
                (cur_epoch in cfg.SELF_TRAIN.PROG_AUG.UPDATE_AUG):
                target_loader.dataset.data_augmentor.re_prepare(
                    augmentor_configs=None, intensity=cfg.SELF_TRAIN.PROG_AUG.SCALE)

            accumulated_iter = train_one_epoch_st(
                model, optimizer, source_reader, target_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter, ema_model=ema_model, cur_epoch=cur_epoch
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                state = checkpoint_state(model, optimizer, trained_epoch, accumulated_iter)

                save_checkpoint(state, filename=ckpt_name)
