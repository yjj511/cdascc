# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from tqdm import tqdm
from lib.utils.utils import *


def train(config, epoch, num_epoch, epoch_iters, num_iters,
          train_source_iter, train_target_iter, optimizer, scheduler, model, writer_dict,
          device, img_vis_dir, mean, std, task_KPI):
    # Training
    model.train()
    batch_time = AverageMeter()
    avg_loss = AverageMeter()
    err_s_domain_loss = AverageMeter()
    err_t_domain_loss = AverageMeter()
    contrastive_loss = AverageMeter()
    tic = time.time()
    writer = writer_dict['writer']
    global_steps = writer_dict['train_global_steps']

    for i_iter in range(num_iters):
        p = float(i_iter + epoch * num_iters) / num_epoch / num_iters
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        x_s, labels_s, size, name_idx = next(train_source_iter)
        x_t, _, _, _ = next(train_target_iter)
        x_s = x_s.to(device)
        x_t = x_t.to(device)
        for i in range(len(labels_s)):
            labels_s[i] = labels_s[i].to(device)

        result = model(x_s, labels_s, 'train', x_t, alpha)
        losses = result['losses']
        s_losses = result['err_s_domain']
        t_losses = result['err_t_domain']
        c_losses = result['contrastive_loss']
        # import pdb
        # pdb.set_trace()
        pre_den = result['pre_den']['1']
        gt_den = result['gt_den']['1']

        loss = losses.mean()  # loss是多个样本的均值 方差更小
        s_loss = s_losses.mean()
        t_loss = t_losses.mean()
        c_loss = c_losses.mean()

        model.zero_grad()
        loss.backward()
        optimizer.step()

        task_KPI.add({
            'acc1': {'gt': result['acc1']['gt'], 'error': result['acc1']['error']},
            'x4': {'gt': result['x4']['gt'], 'error': result['x4']['error']},
            'x8': {'gt': result['x8']['gt'], 'error': result['x8']['error']},
            'x16': {'gt': result['x16']['gt'], 'error': result['x16']['error']},
            'x32': {'gt': result['x32']['gt'], 'error': result['x32']['error']}

        })

        KPI = task_KPI.query()
        x4_acc = KPI['x4']
        x8_acc = KPI['x8']
        x16_acc = KPI['x16']
        x32_acc = KPI['x32']
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss
        avg_loss.update(loss.item())
        err_s_domain_loss.update(s_loss.item())
        err_t_domain_loss.update(t_loss.item())
        contrastive_loss.update(c_loss.item())

        #
        scheduler.step_update(epoch * epoch_iters + i_iter)

    lr = optimizer.param_groups[0]['lr']

    # 打印结果
    print_loss = avg_loss.average()
    msg = 'Epoch: [{}/{}], Time: {:.2f}, ' \
          'lr: {:.4f}, Loss: {:.4f},  s_loss: {:.4f}, t_Loss: {:.4f}, c_Loss: {:.4f},' \
          'acc:{:.2f}, accx8:{:.2f},  accx16:{:.2f},accx32:{:.2f}'.format(
        epoch, num_epoch,
        batch_time.sum, lr * 1e5, print_loss, err_s_domain_loss.average(), err_t_domain_loss.average(),
        contrastive_loss.average(),
        x4_acc.item(), x8_acc.item(), x16_acc.item(), x32_acc.item())
    logging.info(msg)

    writer.add_scalar('train_loss', print_loss, global_steps)
    global_steps = writer_dict['train_global_steps']
    writer_dict['train_global_steps'] = global_steps + 1
    image = x_s[0]

    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)

    save_results_more(global_steps, img_vis_dir, image.cpu().data, \
                      pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),
                      pre_den[0].sum().item(), labels_s[0][0].sum().item())


def validate(config, testloader, model, writer_dict, device,
             img_vis_dir, mean, std):
    model.eval()
    avg_loss = AverageMeter()
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(),
                  'nae': AverageMeter(), 'acc1': AverageMeter()}
    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            # if _>100:
            #     break
            image, label, _, name = batch
            image = image.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)
            result = model(image, label, 'val')

            losses = result['losses']
            pre_den = result['pre_den']['1']
            gt_den = result['gt_den']['1']

            #    -----------Counting performance------------------
            gt_count, pred_cnt = label[0].sum(), pre_den.sum()

            s_mae = torch.abs(gt_count - pred_cnt)

            s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))

            avg_loss.update(losses.item())
            cnt_errors['mae'].update(s_mae.item())
            cnt_errors['mse'].update(s_mse.item())

            s_nae = (torch.abs(gt_count - pred_cnt) / (gt_count + 1e-10))
            cnt_errors['nae'].update(s_nae.item())

            if idx % 20 == 0:
                image = image[0]
                for t, m, s in zip(image, mean, std):
                    t.mul_(s).add_(m)
                save_results_more(name[0], img_vis_dir, image.cpu().data, \
                                  pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),
                                  pred_cnt.item(), gt_count.item())
    print_loss = avg_loss.average()

    mae = cnt_errors['mae'].avg
    mse = np.sqrt(cnt_errors['mse'].avg)
    nae = cnt_errors['nae'].avg

    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']
    writer.add_scalar('valid_loss', print_loss, global_steps)
    writer.add_scalar('valid_mae', mae, global_steps)
    writer_dict['valid_global_steps'] = global_steps + 1

    return print_loss, mae, mse, nae


def test_cc(config, test_dataset, testloader, model
            , mean, std, sv_dir='', sv_pred=False, logger=None):
    model.eval()
    save_count_txt = ''
    device = torch.cuda.current_device()
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name = batch

            image, label, _, name = batch
            image = image.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)

            result = model(image, label, 'val')

            pre_den = result['pre_den']['1']
            gt_den = result['gt_den']['1']
            #    -----------Counting performance------------------
            gt_count, pred_cnt = label[0].sum().item(), pre_den.sum().item()  # pre_data['num'] #

            save_count_txt += '{} {}\n'.format(name[0], pred_cnt)
            msg = '{} {}'.format(gt_count, pred_cnt)
            logger.info(msg)
            s_mae = abs(gt_count - pred_cnt)
            s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
            cnt_errors['mae'].update(s_mae)
            cnt_errors['mse'].update(s_mse)
            if gt_count != 0:
                s_nae = (abs(gt_count - pred_cnt) / gt_count)
                cnt_errors['nae'].update(s_nae)

            image = image[0]
            if sv_pred:
                for t, m, s in zip(image, mean, std):
                    t.mul_(s).add_(m)
                save_results_more(name, sv_dir, image.cpu().data, \
                                  pre_den[0].detach().cpu(), gt_den[0].detach().cpu(), pred_cnt, gt_count,
                                  )

            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                mae = cnt_errors['mae'].avg
                mse = np.sqrt(cnt_errors['mse'].avg)
                nae = cnt_errors['nae'].avg
                msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
                       nae: {: 4.4f}, Class IoU: '.format(mae,
                                                          mse, nae)
                logging.info(msg)
        mae = cnt_errors['mae'].avg
        mse = np.sqrt(cnt_errors['mse'].avg)
        nae = cnt_errors['nae'].avg

    return mae, mse, nae, save_count_txt
