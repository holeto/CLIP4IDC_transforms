import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_clevr_caption import CLEVR_DataLoader as CLEVR_Caption_DataLoader
from dataloaders.dataloader_spot_caption import SPOT_DataLoader as SPOT_Caption_DataLoader
from dataloaders.dataloader_clevr_retrieval import CLEVR_DataLoader
from dataloaders.dataloader_spot_retrieval import SPOT_DataLoader
from dataloaders.dataloader_trance_caption import TRANCE_DataLoader as TRANCE_Caption_DataLoader
from dataloaders.dataloader_trance_retrieval import TRANCE_DataLoader


def dataloader_clevr_train(args, tokenizer):
    if args.task_type == "retrieval":
        DataSet_DataLoader = CLEVR_DataLoader
    else:
        DataSet_DataLoader = CLEVR_Caption_DataLoader

    clevr_dataset = DataSet_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(clevr_dataset)
    dataloader = DataLoader(
        clevr_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(clevr_dataset), train_sampler

def dataloader_clevr_test(args, tokenizer, subset="test"):
    if args.task_type == "retrieval":
        DataSet_DataLoader = CLEVR_DataLoader
    else:
        DataSet_DataLoader = CLEVR_Caption_DataLoader

    clevr_testset = DataSet_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )
    dataloader_clevr = DataLoader(
        clevr_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_clevr, len(clevr_testset)

def dataloader_spot_train(args, tokenizer):
    if args.task_type == "retrieval":
        DataSet_DataLoader = SPOT_DataLoader
    else:
        DataSet_DataLoader = SPOT_Caption_DataLoader

    spot_dataset = DataSet_DataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(spot_dataset)
    dataloader = DataLoader(
        spot_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(spot_dataset), train_sampler

def dataloader_spot_test(args, tokenizer, subset="test"):
    if args.task_type == "retrieval":
        DataSet_DataLoader = SPOT_DataLoader
    else:
        DataSet_DataLoader = SPOT_Caption_DataLoader

    spot_testset = DataSet_DataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
    )
    dataloader_spot = DataLoader(
        spot_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_spot, len(spot_testset)


def _trance_dataset(args, tokenizer, subset):
    kwargs = dict(subset=subset, data_path=args.data_path, features_path=args.features_path,
                  max_words=args.max_words, tokenizer=tokenizer,
                  test_transformation_generalization=getattr(args, "test_transformation_generalization", False))
    if getattr(args, "target", "text") == "grammar":
        # imported here, so that the other datasets do not need the TRANCE grammar
        from dataloaders.dataloader_trance_grammar import TRANCE_Grammar_DataLoader
        return TRANCE_Grammar_DataLoader(task_type=args.task_type, grammar_mask=args.grammar_mask,
                                         init_desc_scope=getattr(args, "init_desc_scope", "own"),
                                         mask_cache_dir=getattr(args, "grammar_mask_cache", None),
                                         mask_workers=getattr(args, "grammar_mask_workers", 0), **kwargs)
    if args.task_type == "retrieval":
        return TRANCE_DataLoader(**kwargs)
    return TRANCE_Caption_DataLoader(**kwargs)


def dataloader_trance_train(args, tokenizer):
    trance_dataset = _trance_dataset(args, tokenizer, "train")

    # The shuffle follows --seed (the sampler's own default is a fixed 0).
    train_sampler = torch.utils.data.distributed.DistributedSampler(trance_dataset, seed=args.seed)
    # drop_last=True would silently give an epoch of zero steps
    if len(train_sampler) < args.batch_size // args.n_gpu:
        raise ValueError("train split has {} samples per process, fewer than one batch of {}; lower --batch_size".format(
            len(train_sampler), args.batch_size // args.n_gpu))
    dataloader = DataLoader(
        trance_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(trance_dataset), train_sampler

def dataloader_trance_test(args, tokenizer, subset="test"):
    trance_testset = _trance_dataset(args, tokenizer, subset)
    dataloader_trance = DataLoader(
        trance_testset,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader_trance, len(trance_testset)


DATALOADER_DICT = {}
DATALOADER_DICT["clevr"] = {"train":dataloader_clevr_train, "val":dataloader_clevr_test, "test":dataloader_clevr_test}
DATALOADER_DICT["spot"] = {"train":dataloader_spot_train, "val":dataloader_spot_test, "test":dataloader_spot_test}
DATALOADER_DICT["trance"] = {"train":dataloader_trance_train, "val":dataloader_trance_test, "test":dataloader_trance_test}
