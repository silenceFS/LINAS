import json
import torch
import torch.utils.data as data
import numpy as np
import random
import os

from basic.util import getVideoId
from util.vocab import clean_str

VIDEO_MAX_LEN=64


def read_video_ids(cap_file):
    video_ids_list = []
    with open(cap_file, 'r') as cap_reader:
        for line in cap_reader.readlines():
            cap_id, caption = line.strip().split(' ', 1)
            video_id = getVideoId(cap_id)
            if video_id not in video_ids_list:
                video_ids_list.append(video_id)
    return video_ids_list

def collate_frame_gru_fn(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][1] is not None:
        data.sort(key=lambda x: len(x[1]), reverse=True)
    videos, captions, cap_bows, idxs, cap_ids, video_ids, vid_tag, all_captions, all_cap_bows = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0
    videos_tag = torch.stack(vid_tag, 0)
    
    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
        all_lengths = []
        m = 0
        for i in range(len(all_captions)):
                all_lengths.append([])
                for j in range(len(all_captions[i])):
                    ans = len(all_captions[i][j])
                    all_lengths[i].append(ans)
                    if ans > m:
                        m = ans
        all_target = torch.zeros(len(all_captions), len(all_captions[0]), m).long()
        all_words_mask = torch.zeros(len(all_captions), len(all_captions[0]), m)
        for i in range(len(all_captions)):
            for j,cap in enumerate(all_captions[i]):
                end = all_lengths[i][j]
                all_target[i,j,:end] = cap[:end]
                all_words_mask[i,j,:end] = 1.0

    else:
        target = None
        lengths = None
        words_mask = None
        all_target = None
        all_lengths = None
        all_words_mask = None


    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None
    all_cap_bows = torch.stack(all_cap_bows, 0) if all_cap_bows[0] is not None else None

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)
    text_data = (target, cap_bows, lengths, words_mask)
    support_text_data = (all_target, all_cap_bows, all_lengths, all_words_mask)

    return video_data, text_data, support_text_data, videos_tag, idxs, cap_ids, video_ids


def collate_frame(data):

    videos, idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0

    video_data = (vidoes, videos_origin, video_lengths, vidoes_mask)

    return video_data, idxs, video_ids


def collate_text_GT(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, cap_bows, idxs, cap_ids, all_captions, all_cap_bows = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
        all_lengths = []
        m = 0
        for i in range(len(all_captions)):
                all_lengths.append([])
                for j in range(len(all_captions[i])):
                    ans = len(all_captions[i][j])
                    all_lengths[i].append(ans)
                    if ans > m:
                        m = ans
        all_target = torch.zeros(len(all_captions), len(all_captions[0]), m).long()
        all_words_mask = torch.zeros(len(all_captions), len(all_captions[0]), m)
        for i in range(len(all_captions)):
            for j,cap in enumerate(all_captions[i]):
                end = all_lengths[i][j]
                all_target[i,j,:end] = cap[:end]
                all_words_mask[i,j,:end] = 1.0

    else:
        target = None
        lengths = None
        words_mask = None
        all_target = None
        all_lengths = None
        all_words_mask = None

    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None
    all_cap_bows = torch.stack(all_cap_bows, 0) if all_cap_bows[0] is not None else None

    text_data = (target, cap_bows, lengths, words_mask)
    support_text_data = (all_target, all_cap_bows, all_lengths, all_words_mask)

    return text_data, support_text_data, idxs, cap_ids


def collate_text_distill(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, cap_bows, idxs, cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None


    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    text_data = (target, cap_bows, lengths, words_mask)

    return text_data, idxs, cap_ids


class Dataset4DualEncoding(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat, tag_path, tag_vocab_path, bow2vec, vocab, support_set_number, video2frames=None):
        # Captions
        self.captions = {}
        self.ref_captions = {}
        self.cap_ids = []
        self.video_ids = set()
        self.video2frames = video2frames
        self.tag_path = tag_path
        self.support_set_number = support_set_number
        with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                cap_id, caption = line.strip().split(' ', 1)
                video_id = getVideoId(cap_id)
                self.captions[cap_id] = caption
                if video_id in self.ref_captions.keys():
                    self.ref_captions[video_id].append(caption)
                else:
                    self.ref_captions[video_id] = [caption]
                self.cap_ids.append(cap_id)
                self.video_ids.add(video_id)
        self.visual_feat = visual_feat
        self.bow2vec = bow2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)

        if not os.path.exists(tag_vocab_path):
            self.tag_vocab_list = []
        else:
            self.tag_vocab_list = json.load(open(tag_vocab_path, 'r'))
        self.tag_vocab_size = len(self.tag_vocab_list)
        self.tag2idx = dict(zip(self.tag_vocab_list, range(self.tag_vocab_size)))

        # self.vid2tags = json.load(open(tag_path, 'r'))    #read the json file of tag
        self.vid2tags = {} 
        if tag_path is not None and os.path.exists(tag_path):
            for line in open(tag_path).readlines():
                # print(line)
                if len(line.strip().split("\t", 1)) < 2:  # no tag available for a specific video
                    vid = line.strip().split("\t", 1)[0]
                    self.vid2tags[vid] = []
                else:
                    vid, or_tags = line.strip().split("\t", 1)
                    tags = [x.split(':')[0] for x in or_tags.strip().split()]
                    
                    # weighed concept scores
                    scores = [float(x.split(':')[1]) for x in or_tags.strip().split()]
                    scores = np.array(scores) / max(scores)

                    self.vid2tags[vid] = list(zip(tags, scores))
        else:
            self.vid2tags = None


    def __getitem__(self, index):
        # import ipdb;ipdb.set_trace()
        cap_id = self.cap_ids[index]
        video_id = getVideoId(cap_id)

        # video
        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        # text
        caption = self.captions[cap_id]
        cap_bow, cap_tensor = self.process_cap(caption)

        # all reference text
        captions = self.ref_captions[video_id].copy()
        num_cap = len(captions)
        if self.support_set_number < 20:
            captions.remove(caption)
            captions = random.sample(captions, self.support_set_number)
        cap_bows = torch.zeros(len(captions), self.bow2vec.ndims)
        cap_tensors = []
        for i, caption in enumerate(captions):
            cap_bow, cap_tensor = self.process_cap(caption)
            cap_bows[i] = cap_bow
            cap_tensors.append(cap_tensor)

        caption = self.captions[cap_id]
        cap_bow, cap_tensor = self.process_cap(caption)
        
        if self.tag_path is not None and os.path.exists(self.tag_path):
            vid_tag_str = self.vid2tags[video_id]     # string representation
            tag_in_vocab = [tag_score for tag_score in vid_tag_str if tag_score[0] in self.tag2idx]
            tag_list = [self.tag2idx[tag_score[0]] for tag_score in tag_in_vocab ]  # index representation
            score_list = [tag_score[1] for tag_score in tag_in_vocab]
            tag_one_hot = torch.zeros(self.tag_vocab_size)  # build zero vector of tag vocabulary that is used to represent tags by one-hot
            for idx, tag_idx in enumerate(tag_list):
                tag_one_hot[tag_idx] = score_list[idx]  # one-hot
        else:
            tag_one_hot = torch.zeros(self.tag_vocab_size)
        vid_tag = torch.Tensor(np.array(tag_one_hot))

        return frames_tensor, cap_tensor, cap_bow, index, cap_id, video_id, vid_tag, cap_tensors, cap_bows

    def __len__(self):
        return self.length

    def process_cap(self, caption):
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.vocab is not None:
            tokens = clean_str(caption)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        return cap_bow, cap_tensor
     

class VisDataSet4DualEncoding(data.Dataset):
    """
    Load video frame features by pre-trained CNN model.
    """
    def __init__(self, visual_feat, video2frames=None, video_ids=None):
        self.visual_feat = visual_feat
        self.video2frames = video2frames
        if video_ids is not None:
            self.video_ids = video_ids
        else:
            self.video_ids = video2frames.keys()
        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]

        frame_list = self.video2frames[video_id]
        frame_vecs = []
        for frame_id in frame_list:
            frame_vecs.append(self.visual_feat.read_one(frame_id))
        frames_tensor = torch.Tensor(frame_vecs)

        return frames_tensor, index, video_id

    def __len__(self):
        return self.length


class TxtDataSet4DualEncoding(data.Dataset):
    """
    Load captions
    """
    def __init__(self, cap_file, bow2vec, vocab, style, video2frames, visual_feat, support_set_number):
        # Captions
        self.style = style
        self.video2frames = video2frames
        self.visual_feat = visual_feat
        self.support_set_number = support_set_number
        if self.style == 'GT':
            self.captions = {}
            self.cap_ids = []
            self.ref_captions = {}
            with open(cap_file, 'r') as cap_reader:
                for line in cap_reader.readlines():
                    cap_id, caption = line.strip().split(' ', 1)
                    video_id = getVideoId(cap_id)
                    self.captions[cap_id] = caption
                    if video_id in self.ref_captions.keys():
                        self.ref_captions[video_id].append(caption)
                    else:
                        self.ref_captions[video_id] = [caption]
                    self.cap_ids.append(cap_id)
            self.bow2vec = bow2vec
            self.vocab = vocab
            self.length = len(self.cap_ids)
        elif self.style == 'distill_from_best_model':
            self.captions = {}
            self.cap_ids = []
            with open(cap_file, 'r') as cap_reader:
                for line in cap_reader.readlines():
                    cap_id, caption = line.strip().split(' ', 1)
                    self.captions[cap_id] = caption
                    self.cap_ids.append(cap_id)
            self.bow2vec = bow2vec
            self.vocab = vocab
            self.length = len(self.cap_ids)

    def __getitem__(self, index):
        if self.style == 'GT':
            cap_id = self.cap_ids[index]
            video_id = getVideoId(cap_id)

            # text
            caption = self.captions[cap_id]
            captions = self.ref_captions[video_id].copy()
            num_cap = len(captions)
            # assert num_cap==20
            if self.support_set_number < 20:
                captions.remove(caption)
                captions = random.sample(captions, self.support_set_number)
            cap_bows = torch.zeros(len(captions), self.bow2vec.ndims)
            cap_tensors = []
            for i, cap in enumerate(captions):
                cap_bow, cap_tensor = self.process_cap(cap)
                cap_bows[i] = cap_bow
                cap_tensors.append(cap_tensor)

            cap_bow, cap_tensor = self.process_cap(caption)

            return cap_tensor, cap_bow, index, cap_id, cap_tensors, cap_bows

        elif self.style == 'distill_from_best_model':
            cap_id = self.cap_ids[index]

            caption = self.captions[cap_id]
            cap_bow, cap_tensor = self.process_cap(caption)

            return cap_tensor, cap_bow, index, cap_id

    def __len__(self):
        return self.length

    def process_cap(self, caption):
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.vocab is not None:
            tokens = clean_str(caption)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        return cap_bow, cap_tensor

def get_data_loaders(cap_files, visual_feats, tag_path, tag_vocab_path, vocab, bow2vec, batch_size=100, num_workers=2, video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], tag_path, tag_vocab_path, bow2vec, vocab, video2frames=video2frames['train']),
            'val': Dataset4DualEncoding(cap_files['val'], visual_feats['val'], None, tag_vocab_path, bow2vec, vocab, video2frames=video2frames['val']) }

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in cap_files }
    return data_loaders


def get_train_data_loaders(cap_files, visual_feats, tag_path, tag_vocab_path, vocab, bow2vec, support_set_number, batch_size=100, num_workers=2, video2frames=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    Args:
        cap_files: caption files (dict) keys: [train, val]
        visual_feats: image feats (dict) keys: [train, val]
    """
    dset = {'train': Dataset4DualEncoding(cap_files['train'], visual_feats['train'], tag_path, tag_vocab_path, bow2vec, vocab, support_set_number, video2frames=video2frames['train'])}

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in cap_files  if x=='train' }
    return data_loaders



def get_test_data_loaders(cap_files, visual_feats, tag_path, tag_vocab_path, vocab, bow2vec, batch_size=100, num_workers=2, video2frames = None):
    """
    Returns torch.utils.data.DataLoader for test dataset
    Args:
        cap_files: caption files (dict) keys: [test]
        visual_feats: image feats (dict) keys: [test]
    """
    dset = {'test': Dataset4DualEncoding(cap_files['test'], visual_feats['test'], tag_path['test'], tag_vocab_path, bow2vec, vocab, video2frames = video2frames['test'])}


    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    # num_workers=0,
                                    num_workers=num_workers,
                                    collate_fn=collate_frame_gru_fn)
                        for x in cap_files }
    return data_loaders


def get_vis_data_loader(vis_feat, batch_size=100, num_workers=2, video2frames=None, video_ids=None):
    dset = VisDataSet4DualEncoding(vis_feat, video2frames, video_ids=video_ids)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_frame)
    return data_loader


def get_txt_data_loader(cap_file, vocab, bow2vec, support_set_number, batch_size=100, num_workers=2, style='distill_from_best_model', video2frames=None, visual_feat=None):
    dset = TxtDataSet4DualEncoding(cap_file, bow2vec, vocab, style, video2frames, visual_feat, support_set_number)

    if style == 'GT':
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=num_workers,
                                                collate_fn=collate_text_GT)
    elif style == 'distill_from_best_model':
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=num_workers,
                                                collate_fn=collate_text_distill)
    return data_loader



if __name__ == '__main__':
    pass
