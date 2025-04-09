#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch


class Optimizer(object):
    def __init__(self,
                model,
                lr0,
                momentum,
                wd,
                warmup_steps,
                warmup_start_lr,
                max_iter,
                power,
                it,
                *args, **kwargs):
        self.warmup_steps = warmup_steps
        self.warmup_start_lr = warmup_start_lr
        self.lr0 = lr0
        self.lr = self.lr0
        self.max_iter = float(max_iter)
        self.power = power
        self.it = it


        self.optim = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr = lr0,
                momentum = momentum,
                weight_decay = wd)
        self.warmup_factor = (self.lr0/self.warmup_start_lr)**(1./self.warmup_steps)


    def get_lr(self):
        if self.it <= self.warmup_steps:
            lr = self.warmup_start_lr*(self.warmup_factor**self.it)
        else:
            factor = (1-(self.it-self.warmup_steps)/(self.max_iter-self.warmup_steps))**self.power
            lr = self.lr0 * factor
        return lr


    def step(self):
        self.lr = self.get_lr()

        self.it += 1
        self.optim.step()


    def zero_grad(self):
        self.optim.zero_grad()

