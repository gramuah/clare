import torch, os, time
import numpy as np
from dataloader import VisionDataset , get_statistics
from torch import nn, optim
from opts import parse_args
from utils import AverageMeter, get_accuracy, cutmix_data, get_logger, seed_everything, save_model, load_model, create_class_mask, create_class_mask_realistic, semirealistic_divide_classes
import clip
from models.CLIP import FrozenCLIP
from sklearn.metrics import confusion_matrix


def experiment(opt, train_loader, test_loader, model, logger, num_passes, task_num, total_tasks):
    best_prec1 = 0.0
    best_prev_task = 0.0
    model = model.cuda()  # Better speed with little loss in accuracy. If loss in accuracy is big, use apex.
    criterion = nn.CrossEntropyLoss().cuda()
    if opt.scenario == 'realistic' and task_num == 0 and opt.finetune == 'True':
        logger.info('==> Task 0: Realistic Scenario, Training from scratch')
        optimizer = optim.SGD(model.parameters(), lr=opt.maxlr, momentum=0.9, weight_decay=opt.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=opt.minlr)
    elif opt.scenario == 'realistic' and task_num > 0 and opt.finetune == 'True':
        logger.info('==> Task {}: Realistic Scenario, Fine-tuning from the previous task'.format(task_num))
        optimizer = optim.SGD(model.parameters(), lr=opt.maxlr, momentum=0.9, weight_decay=opt.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=opt.minlr)
        model = load_model(opt, model, logger)
    else:
        optimizer = optim.SGD(model.parameters(), lr=opt.maxlr, momentum=0.9, weight_decay=opt.weight_decay) # Optimizer
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=opt.minlr) # Scheduler
    logger.info("==> Opts for this training: %d" + str(opt))

    for epoch in range(num_passes):
            # Handle lr scheduling
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in optimizer.param_groups: # Set lr to 0.1*maxlr
                    param_group['lr'] = opt.maxlr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = opt.maxlr
            else:  # And go!
                scheduler.step()

            # Train and test loop
            logger.info(
                "==> Starting epoch number: " + str(epoch) + ", Learning rate: " + str(optimizer.param_groups[0]['lr']))
            model, optimizer, loss = train(opt=opt, loader=train_loader, model=model, criterion=criterion, optimizer=optimizer,
                                     epoch=epoch, logger=logger, task_id=task_num)
            if epoch%10==0:
                prec1, old_class_accu = test(loader=test_loader, model=model, criterion=criterion, logger=logger,
                                             epoch=epoch, task_id=task_num, total_tasks=total_tasks)

            logger.info('==> Current accuracy: [{:.3f}]\t'.format(prec1))
            if prec1 > best_prec1:
                save_model(opt,model)
                logger.info('==> Accuracies\tPrevious: [{:.3f}]\t'.format(best_prec1) + 'Current: [{:.3f}]\t'.format(prec1))
                best_prec1 = float(prec1)
                best_prev_task = float(old_class_accu)
    logger.info('==> Finished training task ' + str(task_num))
    logger.info('==> Training Task completed! Acc: [{0:.3f}]'.format(best_prec1))
    return best_prec1, model, best_prev_task

def test(loader, model, criterion, logger, epoch, task_id, total_tasks):
        model.eval()
        old_task_accuracy = 0
        # Initialize the AverageMeter objects
        losses, batch_time, accuracy, task_accuracy = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
        all_labels = []
        all_preds = []
        class_list = []
        with torch.no_grad():
            start = time.time()
            for inputs, labels in loader:
                # Get outputs
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                losses.update(loss.data, inputs.size(0))
                prob = torch.nn.functional.softmax(outputs, dim=1)
                acc, preds = get_accuracy(prob, labels)
                accuracy.update(acc, labels.size(0))
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                batch_time.update(time.time() - start)
                start = time.time()

        cm = confusion_matrix(all_labels, all_preds)
        num_classes = [cls for tsk in range(0, task_id+1) for cls in total_tasks[tsk]]
        for num in num_classes:
            if num not in class_list:
                class_list.append(num)
        old_classes = [item for item in class_list if item not in total_tasks[task_id]]
        if opt.scenario == 'realistic':
            if task_id > 0:
                old_classes = total_tasks[task_id-1]
            old_class_mask = create_class_mask_realistic(class_list, old_classes)
        elif opt.scenario == 'semirealistic':
             old_class_mask = create_class_mask_realistic(class_list, old_classes)
        else:
            old_class_mask = create_class_mask((task_id+1), ((len(class_list))//(task_id+1)), task_id)
        if task_id > 0:
            cm_masked = cm * old_class_mask
            if cm_masked.diagonal().sum() == 0:
                old_task_accuracy = 0.0
            else:
                old_task_accuracy = (cm_masked.diagonal().sum() / cm_masked.sum()) * 100
        return accuracy.avg, old_task_accuracy

def train(opt, loader, model, criterion, optimizer, epoch, logger, task_id):
        logger.info('==> Training task %d', task_id)
        model.train()
        losses, data_time, batch_time = AverageMeter(), AverageMeter(), AverageMeter()
        start = time.time()
        for inputs, labels in loader:
            # Tweak inputs
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs, labels = inputs.to(device), labels.to(device)

            do_cutmix = opt.regularization == 'cutmix' and np.random.rand(1) < opt.cutmix_prob
            if do_cutmix: inputs, labels_a, labels_b, lam = cutmix_data(x=inputs, y=labels, alpha=opt.cutmix_alpha)  # Cutmix
            data_time.update(time.time() - start)  # Measure data loading time


            outputs = model(inputs)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs,labels_b) if do_cutmix else criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()

            # Log losses
            losses.update(loss.data.item(), labels.size(0))
            batch_time.update(time.time() - start)
            start = time.time()
            loss_average = losses.avg
            formatted_loss = "{:.2f}".format(loss_average)
        logger.info('==> Train:[{0}]\tTime:{batch_time.sum:.4f}\tData:{data_time.sum:.4f}\tLoss:{loss.avg:.4f}\t'
                    .format(epoch, batch_time=batch_time, data_time=data_time, loss=losses))
        return model, optimizer, formatted_loss


if __name__ == '__main__':
    opt = parse_args()
    seed_everything(seed=opt.seed)
    tasks_list = None
    class_order = None

    # Handle 'path does not exist errors'
    if not os.path.isdir(opt.log_dir+'/'+opt.exp_name):
        os.makedirs(opt.log_dir+'/'+opt.exp_name)
    if opt.old_exp_name!='test' and not os.path.isdir(opt.log_dir+'/'+opt.old_exp_name):
        os.makedirs(opt.log_dir+'/'+opt.old_exp_name)
    console_logger = get_logger(folder=opt.log_dir + '/' + opt.exp_name + '/')

    # Handle fixed class orders. Note: Class ordering code hacky. Would need to manually adjust here to test for different datasets.
    console_logger.debug("==> Loading dataset..")

    if opt.scenario == 'semirealistic':
        mean, std, opt.total_num_classes, opt.inp_size, opt.in_channels = get_statistics(opt.dataset)
        tasks_list = semirealistic_divide_classes(opt.total_num_classes, opt.num_tasks)
    else:
        if opt.dataset == 'CIFAR10':
            class_order = [6, 8, 9, 7, 5, 3, 0, 4, 1, 2]
        elif opt.dataset == 'CIFAR100' or opt.dataset =='ImageNet100':
            class_order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45,
                       88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28,
                       6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42,
                       65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35,
                       39]  # Currently testing using iCARL test order-- restricted to CIFAR100. For the other class orders refer to https://github.com/arthurdouillard/incremental_learning.pytorch/tree/master/options/data
        else:
            class_order = [183, 23, 80, 74, 76, 125, 150, 122, 96, 60, 28, 143, 93, 160, 193, 148, 70, 162, 141, 118, 54, 7, 42, 94, 178, 114,
                        37, 8, 156, 138, 188, 171, 95, 191, 166, 32, 83, 73, 48, 151, 65, 21, 53, 41, 104, 144, 176, 0, 177, 194, 137, 17, 45, 89, 173,
                        129, 185, 55, 174, 149, 90, 128, 15, 14, 69, 167, 132, 46, 168, 190, 109, 59, 155, 19, 58, 120, 36, 64, 175, 2, 63, 135, 113, 130,
                        153, 43, 75, 106, 119, 102, 25, 62, 139, 116, 107, 92, 84, 186, 142, 117, 78, 192, 121, 101, 61, 81, 26, 71, 24, 5, 180,
                        170, 163, 88, 108, 159, 112, 97, 1, 140, 79, 198, 199, 12, 30, 127, 197, 33, 18, 38, 52, 77, 85, 103, 51, 49, 146, 158,
                        196, 35, 22, 9, 10, 169, 179, 195, 27, 181, 29, 124, 20, 16, 31, 189, 72, 152, 131, 6, 105, 44, 164, 68, 3, 126, 82, 134, 11,
                        133, 123, 47, 40, 145, 57, 110, 147, 161, 187, 13, 157, 136, 67, 182, 34, 50, 56, 154, 39, 91, 165, 4, 172, 66, 99, 100, 115,
                        86, 184, 98, 111, 87]

    dobj = VisionDataset(opt, class_order=class_order)
    console_logger.info("==> Starting Continual Learning Training..")
    average_accu = []
    forgetting_general= []
    forgetting_task_list = []
    previous_best_acc1 = 0.0
    for task in range(opt.num_tasks):
        dobj.gen_cl_mapping(task, tasks_list)
        numb_classes = len(dobj.encountered_classes)
        if opt.model == 'CLIP':
            console_logger.info("==> Loading CLIP model..")
            clip_model, _ = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
            console_logger.info("==> Freezing CLIP model..")
            model = FrozenCLIP(clip_model, numb_classes)
        best_acc1, model, old_tasks_acc = experiment(opt=opt, train_loader=dobj.cltrain_loader, test_loader=dobj.cltest_loader,
                model=model, logger=console_logger, num_passes=opt.num_passes, task_num=task, total_tasks = dobj.total_tasks)

        if task ==0:
            console_logger.info("==> Task %d forgetting general: %f forgetting old task: %f", task, 0.0, 0.0)
            average_accu.append(best_acc1)
            previous_best_acc1 = best_acc1
            forgetting_general.append(0.0)
            forgetting_task_list.append(0.0)
        else:
            average_accu.append(best_acc1)
            forgetting_gen = previous_best_acc1 - best_acc1
            forgetting_old_task = previous_best_acc1 - old_tasks_acc
            forgetting_general.append(forgetting_gen)
            forgetting_task_list.append(forgetting_old_task)
            previous_best_acc1 = best_acc1
            console_logger.info("==> Task %d forgetting general: %f forgetting old task: %f", task, forgetting_gen, forgetting_old_task)

    console_logger.info('==> Average accuracy: [{0:.3f}]'.format(np.mean(average_accu)))
    console_logger.info('==> Average global forgetting: [{0:.3f}]'.format(np.mean(forgetting_general)))
    console_logger.info('==> Average tasks forgetting: [{0:.3f}]'.format(np.mean(forgetting_task_list)))
    console_logger.debug("==> Completed!")
