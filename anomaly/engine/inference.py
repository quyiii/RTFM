import torch
import numpy as np
import os.path as osp

from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

def inference(dataloader, model, args, device, viz=None):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda(device=args.gpus[0])

        gt = dataloader.dataset.ground_truths
        dataset = args.dataset.lower()

        # if args.inference:
        #     video_list = dataloader.dataset.video_list
        #     result_dict = dict()

        for i, input in enumerate(dataloader):
            input = input.cuda(device=args.gpus[0])
            input = input.permute(0, 2, 1, 3)

            outputs = model(inputs=input)

            # >> parse outputs
            logits = outputs['scores']

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            # if args.inference:
            #     video_id = video_list[i]
            #     result_dict[video_id] = logits.cpu().detach().numpy()

            sig = logits
            pred = torch.cat((pred, sig))

        # if args.inference:
        #     out_dir = f'output/{dataset}'
        #     import pickle
        #     with open(osp.join(out_dir, f'{dataset}_taskaware_results.pickle'), 'wb') as fout:
        #         pickle.dump(result_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        roc_auc = auc(fpr, tpr)

        if viz is not None:
            viz.plot_lines('roc_auc', roc_auc)

        return roc_auc

def div_auc(result_dict, class_list, classratios):
    results = {}
    for classname, videolist in class_list.items():
        gt = []
        pred = []
        for video_name in videolist:
            gt.append(result_dict[video_name]['gt'])
            pred.append(result_dict[video_name]['pred'])
        gt = np.concatenate(gt)
        pred = np.concatenate(pred)

        results[classname] = {'gt': gt.repeat(classratios[classname]), 'pred': pred.repeat(classratios[classname])}
    
    auc_dic = {}
    for classname, result in results.items():
        if classname != 'Normal':
            fpr, tpr, threshold = roc_curve(np.concatenate([results['Normal']['gt'], result['gt']]), np.concatenate([results['Normal']['pred'], result['pred']]))

            auc_dic[classname] = auc(fpr, tpr)
    return auc_dic

def inference_div(dataloader, model, args, device):
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0).cuda(device=args.gpus[0])

        classratios = dataloader.dataset.classratios
        gt_detail = dataloader.dataset.ground_truths_details

        gt = dataloader.dataset.ground_truths
        dataset = args.dataset.lower()

        class_list = dataloader.dataset.classlist
        video_list = dataloader.dataset.video_list
        result_dict = dict()

        for i, input in enumerate(dataloader):
            input = input.cuda(device=args.gpus[0])
            input = input.permute(0, 2, 1, 3)

            outputs = model(inputs=input)

            logits = outputs['scores']

            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)

            result_dict[video_list[i]] = {'gt': gt_detail[video_list[i]],'pred': np.repeat(np.array(logits.cpu().detach().numpy()), 16)}

            sig = logits
            pred = torch.cat((pred, sig))

        pred = list(pred.cpu().detach().numpy())
        pred = np.repeat(np.array(pred), 16)

        fpr, tpr, threshold = roc_curve(list(gt), pred)
        roc_auc = auc(fpr, tpr)

        auc_dic = div_auc(result_dict, class_list, classratios)
        auc_dic['Total'] = roc_auc
        return auc_dic