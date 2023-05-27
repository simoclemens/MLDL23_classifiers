import torch


# function used to return top1 and top5 accuracy
def compute_accuracy(predicted_labels, ground_truth_labels, topk=(1, 5)):
    _, predicted_classes = torch.topk(predicted_labels, max(topk), dim=1)
    correct_topk = torch.eq(predicted_classes, ground_truth_labels.view(-1, 1)).float()
    results = []
    for k in topk:
        topk_accuracy = torch.mean(torch.sum(correct_topk[:, :k], dim=1))
        results.append(topk_accuracy.item() * 100)
    return tuple(results)

# function used in order to get the loss function
def get_loss_function():
    loss_function = torch.nn.CrossEntropyLoss()
    return loss_function


# function used in order to get the optimizer (parameters are defined in main)
def get_optimizer(net, lr, wd, momentum):
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    return optimizer


