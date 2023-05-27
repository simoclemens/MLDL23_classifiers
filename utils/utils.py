from collections import Mapping
import torch
import numpy as np


def dense_sampling(start_frame_clip, end_frame_clip, num_frames_per_clip, stride):
    centroid = (start_frame_clip + end_frame_clip) // 2

    if num_frames_per_clip % 2 == 0:
        num_frames_left = num_frames_right = num_frames_per_clip // 2
    else:
        num_frames_left = num_frames_per_clip // 2
        num_frames_right = num_frames_per_clip // 2 + 1

    stride_left = max(stride, 1)  # ensure stride is at least 1
    stride_right = max(stride, 1)  # ensure stride is at least 1

    num_frames_left_adjusted = min(num_frames_left, int((centroid - start_frame_clip) / stride_left))
    num_frames_right_adjusted = min(num_frames_right, int((end_frame_clip - centroid) / stride_right))

    frames = [centroid]
    for i in range(1, num_frames_left_adjusted + 1):
        frame_left = int(round(centroid - i * stride_left))
        frames.insert(0, frame_left)
    for i in range(1, num_frames_right_adjusted + 1):
        frame_right = int(round(centroid + i * stride_right))
        frames.append(frame_right)

    num_frames = len(frames)
    if num_frames < num_frames_per_clip:
        num_frames_to_add = num_frames_per_clip - num_frames
        if num_frames_left_adjusted >= num_frames_left:
            num_frames_right_adjusted += num_frames_to_add
        else:
            num_frames_left_adjusted += num_frames_to_add
        for i in range(num_frames_to_add):
            if i % 2 == 0:
                frame_left = int(round(centroid - (num_frames_left_adjusted + i // 2 + 1) * stride_left))
                frames.insert(0, frame_left)
            else:
                frame_right = int(round(centroid + (num_frames_right_adjusted + i // 2 + 1) * stride_right))
                frames.append(frame_right)

    return frames

def uniform_sampling(start_frame_clip, end_frame_clip, num_frames_per_clip):
    frame_indices = np.linspace(start_frame_clip, end_frame_clip, num=num_frames_per_clip, dtype=int)
    sampled_frames = [frame_indices[i] for i in range(num_frames_per_clip)]
    return sampled_frames


def get_domains_and_labels(args):
    num_verbs = 8
    domains = {'D1': 8, 'D2': 1, 'D3': 22}
    source_domain = domains[args.dataset.shift.split("-")[0]]
    target_domain = domains[args.dataset.shift.split("-")[1]]
    valid_labels = [i for i in range(num_verbs)]
    num_class = num_verbs
    return num_class, valid_labels, source_domain, target_domain


class Accuracy(object):
    """Computes and stores the average and current value of different top-k accuracies from the outputs and labels"""

    def __init__(self, topk=(1,), classes=8):
        assert len(topk) > 0
        self.topk = topk
        self.classes = classes
        self.avg, self.val, self.sum, self.count, self.correct, self.total = None, None, None, None, None, None
        self.reset()

    def reset(self):
        self.val = {tk: 0 for tk in self.topk}
        self.avg = {tk: 0 for tk in self.topk}
        self.sum = {tk: 0 for tk in self.topk}
        self.count = {tk: 0 for tk in self.topk}
        self.correct = list(0 for _ in range(self.classes))
        self.total = list(0 for _ in range(self.classes))

    def update(self, outputs, labels):
        batch = labels.size(0)
        # compute separately all the top-k accuracies and the per-class accuracy
        for i_tk, top_k in enumerate(self.topk):
            if i_tk == 0:
                res = self.accuracy(outputs, labels, perclass_acc=True, topk=[top_k])
                class_correct = res[1]
                class_total = res[2]
                res = res[0]
            else:
                res = self.accuracy(outputs, labels, perclass_acc=False, topk=[top_k])[0]
            self.val[top_k] = res
            self.sum[top_k] += res * batch
            self.count[top_k] += batch
            self.avg[top_k] = self.sum[top_k] / self.count[top_k]

        for i in range(0, self.classes):
            self.correct[i] += class_correct[i]
            self.total[i] += class_total[i]

    def accuracy(self, output, target, perclass_acc=False, topk=(1,)):
        """
        Computes the precision@k for the specified values of k
        output: torch.Tensor -> the predictions
        target: torch.Tensor -> ground truth labels
        perclass_acc -> bool, True if you want to compute also the top-1 accuracy per class
        """
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).to(torch.float32).sum(0)
            res.append(float(correct_k.mul_(100.0 / batch_size)))
        if perclass_acc:
            # getting also top1 accuracy per class
            class_correct, class_total = self.accuracy_per_class(correct[:1].view(-1), target)
            res.append(class_correct)
            res.append(class_total)
        return res

    def accuracy_per_class(self, correct, target):
        """
        function to compute the accuracy per class
        correct -> (batch, bool): vector which, for each element of the batch, contains True/False depending on if
                                  the element in a specific poisition was correctly classified or not
        target -> (batch, label): vector containing the ground truth for each element
        """
        class_correct = list(0. for _ in range(0, self.classes))
        class_total = list(0. for _ in range(0, self.classes))
        for i in range(0, target.size(0)):
            class_label = target[i].item()
            class_correct[class_label] += correct[i].item()
            class_total[class_label] += 1
        return class_correct, class_total


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()
        self.val, self.acc, self.avg, self.sum, self.count = 0, 0, 0, 0, 0

    def reset(self):
        self.val = 0
        self.acc = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.acc += val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def pformat_dict(d, indent=0):
    fstr = ""
    for key, value in d.items():
        fstr += '\n' + '  ' * indent + str(key) + ":"
        if isinstance(value, Mapping):
            fstr += pformat_dict(value, indent + 1)
        else:
            fstr += ' ' + str(value)
    return fstr
