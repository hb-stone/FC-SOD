from torch.utils.data import DataLoader


class SOD_Dataloader(object):

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None):
        super().__init__()
        self.dataloader = DataLoader(dataset, batch_size, shuffle, sampler,
                 batch_sampler, num_workers, collate_fn,
                 pin_memory, drop_last, timeout,
                 worker_init_fn, multiprocessing_context)
        self.iter = iter(self.dataloader)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.dataloader)
            return next(self.iter)

