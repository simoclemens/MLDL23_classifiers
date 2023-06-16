from abc import ABC
import pandas as pd
import torch.utils.data as data
import os
import os.path
import torch


# ACTION SENSE DATASET
# built in order to work directly with features loaded from a pkl in the format used by feature extraction
class ActionSenseDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, num_frames_per_clip, num_clips, annotations_path, features_path,**kwargs):

        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.annotations_path = annotations_path # path of the annotation file
        self.features_path = features_path # path of the features file
        self.num_frames_per_clip = num_frames_per_clip # number of frames per clip
        self.num_clips = num_clips # number of clips per video (= number of features arrays per modality)

        # creation of the pickle file name considering the split and the modality (e.g. D1 + _ + test + .pkl)
        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        else:
            pickle_name = split + "_test.pkl"

        # get the pickle file location (path + name). The path must be inserted in dataset_conf.annotations_path
        self.actions_list = pd.read_pickle(os.path.join(self.annotations_path, pickle_name))
        print(f"Dataloader for {split}-{self.mode} with {len(self.actions_list)} samples generated")

        # generate a list of dictionaries with uid and label for each action
        self.uid_list = [{'uid': item[1]['uid'], 'label':item[1]['verb_class']} for item in self.actions_list.iterrows()]

        self.model_features = None

        features_name = 'features_' + split + '.pkl'
        # for every modality [RGB,EMG]

        self.model_features = pd.DataFrame(pd.read_pickle(os.path.join(features_path,features_name))['features'])[
                ["uid", "features_RGB", "features_EMG"]]

        # merge df obtaining for each action features, uid and labels
        self.model_features = pd.merge(self.model_features, self.actions_list, how="inner", on="uid")

    def __getitem__(self, index):
        # record is a row of the pkl file containing one sample/action
        action = self.uid_list[index]
        sample = {}
        sample_row = self.model_features[self.model_features["uid"] == int(action['uid'])]
        assert len(sample_row) == 1
        for m in self.modalities:
            sample[m] = sample_row["features_" + m].values[0]
        image_features = sample['RGB']
        emg_features = sample['EMG']
        labels = action['label']
        output = {'RGB': image_features, 'EMG': emg_features, 'label': labels}
        return output
        # return torch.stack([image_features,emg_features], dim=1), labels

    def __len__(self):
        return len(self.actions_list)


# EPIC KITCHEN DATASET
# built in order to work directly with features loaded from a pkl in the format used by feature extraction
class EpicKitchenDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, num_frames_per_clip, num_clips, annotations_path, features_path,**kwargs):

        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.annotations_path = annotations_path # path of the annotation file
        self.features_path = features_path # path of the features file
        self.num_frames_per_clip = num_frames_per_clip # number of frames per clip
        self.num_clips = num_clips # number of clips per video (= number of features arrays per modality)

        # creation of the pickle file name considering the split and the modality (e.g. D1 + _ + test + .pkl)
        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        else:
            pickle_name = split + "_test.pkl"

        # get the pickle file location (path + name). The path must be inserted in dataset_conf.annotations_path
        self.actions_list = pd.read_pickle(os.path.join(self.annotations_path, pickle_name))
        print(f"Dataloader for {split}-{self.mode} with {len(self.actions_list)} samples generated")

        # generate a list of dictionaries with uid and label for each action
        self.uid_list = [{'uid': item[1]['uid'], 'label':item[1]['verb_class']} for item in self.actions_list.iterrows()]

        self.model_features = None

        features_name = 'features_' + split + '.pkl'
        # for every modality [RGB,EMG]

        self.model_features = pd.DataFrame(pd.read_pickle(os.path.join(features_path,features_name))['features'])[
                ["uid", "features_EMG"]]
        print(self.model_features.info())

        # merge df obtaining for each action features, uid and labels
        self.model_features = pd.merge(self.model_features, self.actions_list, how="inner", on="uid")

    def __getitem__(self, index):
        # record is a row of the pkl file containing one sample/action
        action = self.uid_list[index]
        sample_row = self.model_features[self.model_features["uid"] == int(action['uid'])]
        assert len(sample_row) == 1
        image_features = sample_row["features_EMG"].values[0]
        labels = action['label']
        output = {'EMG': image_features, 'label': labels}
        return output
        # return torch.stack([image_features,emg_features], dim=1), labels

    def __len__(self):
        return len(self.actions_list)
