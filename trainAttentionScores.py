from utils.train_functions import get_loss_function, get_optimizer, compute_accuracy
import torch.nn.parallel
import torch.optim
import torch
from utils.loaders import ActionSenseDataset
import os
import torch.utils.data
from models.multiModalScores import ScoreClassifier
import datetime
import sys


# train function
def train(file, net, train_loader, val_loader, optimizer, cost_function, n_classes, n_clips=5, batch_size=32,
          loss_weight=1, training_iterations=1000, device="cuda:0"):
    top_accuracy = 0
    data_loader_source = iter(train_loader)

    optimizer.zero_grad()  # reset the optimizer gradient
    for iteration in range(training_iterations):
        net.train(True)
        # this snippet is used because we reason in iterations and if we finish the dataset we need to start again
        try:
            data_source = next(data_loader_source)
        except StopIteration:
            data_loader_source = iter(train_loader)
            data_source = next(data_loader_source)
        # IMPORTANT
        # data are in the shape rows -> item of the batch, columns -> clips, 3rd dim -> classes features
        label = data_source['label'].to(device)
        inputs = {}

        inputs['RGB'] = data_source['RGB'].to(device)
        inputs['EMG'] = data_source['EMG'].to(device)

        logits = net.forward(inputs)  # get predictions from the net
        # compute the loss and divide for the number of clips in order to get the average for clip
        loss = cost_function(logits, label)
        loss.backward()  # apply the backward

        optimizer.step()  # update the parameters
        optimizer.zero_grad()  # reset gradient of the optimizer for next iteration

        test_metrics = validate(net, val_loader, n_classes, n_clips, batch_size)

        file.write('[{}/{}] ITERATION COMPLETED\n'.format(iteration, training_iterations))
        '''
        file.write('TRAIN: acc@top1={:.2f}%  acc@top5={:.2f}% loss={:.2f}\n',
                   train_metrics['top1'], train_metrics['top5'], loss.item())
        '''
        file.write('TEST: acc@top1={:.2f}%  acc@top5={:.2f}%\n\n'.format(test_metrics['top1']*100, 0))

        if test_metrics['top1'] >= top_accuracy:
            top_accuracy = test_metrics['top1']

        if iteration % 10 == 0:
            print('ITERATION:' + str(iteration) + ' - BEST ACCURACY: {:.2f}'.format(top_accuracy*100))

    file.write('TOP ACCURACY {:.2f}'.format(top_accuracy))

    return top_accuracy


# validation function
def validate(net, val_loader, n_classes, n_clips=5, batch_size=32, device="cuda:0"):

    net.train(False)  # set model to validate

    total_size = len(val_loader.dataset)
    val_correct = 0

    with torch.no_grad():  # do not update the gradient
        for iteration, (data_source) in enumerate(val_loader):  # extract batches from the val_loader

            label = data_source['label'].to(device)  # send label to gpu

            # create a zero array with logits shape
            # rows -> clips, columns -> item of the batch, 3rd dim -> classes prob
            logits = torch.zeros((n_clips, batch_size, n_classes)).to(device)

            inputs = {}
            inputs['RGB'] = data_source['RGB'].to(device)
            inputs['EMG'] = data_source['EMG'].to(device)

            logits = net(inputs)  # get predictions from the net

            _, predicted = torch.max(logits.data, 1)

            val_correct += (predicted == label).sum().item()

            #accuracy = compute_accuracy(logits, label, topk=(1, 5))

            #top1_acc += accuracy[0] * size / total_size
            #top5_acc += accuracy[1] * size / total_size

    # compute the accuracy
    accuracy = val_correct/total_size
    test_results = {'top1': accuracy}

    return test_results


def main():
    device = "cuda:0"

    lr = 0.01
    wd = 1e-7
    momentum = 0.9
    loss_weight = 1

    batch_size = 32
    num_frames = 16

    log_path = 'logs'

    dataset = sys.argv[1]
    split = sys.argv[2]
    n_clips = int(sys.argv[3])
    n_classes = int(sys.argv[4])
    annotations_path = sys.argv[5]
    features_path = sys.argv[6]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = dataset + '_' + timestamp + '.txt'

    file = open(os.path.join(log_path, file_name), 'w')

    train_loader = torch.utils.data.DataLoader(ActionSenseDataset(split, ['RGB', 'EMG'],
                                                                  'train', num_frames_per_clip=num_frames,
                                                                  num_clips=n_clips,
                                                                  annotations_path=annotations_path,
                                                                  features_path=features_path),
                                               batch_size=batch_size, shuffle=True,
                                               pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(ActionSenseDataset(split, ['RGB', 'EMG'],
                                                                'test', num_frames_per_clip=num_frames,
                                                                num_clips=n_clips,
                                                                annotations_path=annotations_path,
                                                                features_path=features_path),
                                             batch_size=batch_size, shuffle=True,
                                             pin_memory=True, drop_last=True)

    net = ScoreClassifier(n_classes=n_classes)
    net = net.to(device)
    optimizer = get_optimizer(net=net, wd=wd, lr=lr, momentum=momentum)
    loss = get_loss_function()

    top_accuracy = train(file=file, net=net, train_loader=train_loader, val_loader=val_loader,
                         optimizer=optimizer, cost_function=loss, n_classes=n_classes, n_clips=n_clips,
                         batch_size=batch_size)

    file.close()
    print('TOP TEST ACCURACY:' + str(top_accuracy))
    print('THE END!!!')


if __name__ == '__main__':
    main()
