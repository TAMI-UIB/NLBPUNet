# -*- coding: utf-8 -*-
from numpy import min, max, mean, median
#from matplotlib import pyplot as plt
import pickle
from os.path import join


class Recorder(object):
    def __init__(self, current_epoch=0, output_dir=None):
        self.current_epoch = current_epoch
        self.output_dir = output_dir
        self.train_loss = dict()  # key: epoch_id
        self.test_loss = dict()

        self.sam = dict()  # key: epoch_id, values: [max, min, avg, median]
        self.erags = dict()
        self.scc = dict()
        self.qn = dict()

        self.d_lambda = dict()  # key: epoch_id, values: [max, min, avg, median]
        self.d_s = dict()
        self.qnr = dict()

        self.total = {'sam': self.sam, 'erags': self.erags, 'scc': self.scc,
                      'qnr': self.qnr, 'd_lambda': self.d_lambda, 'd_s': self.d_s}

    def write_loss(self, train_loss=None, test_loss=None, _print=True):
        if train_loss:
            self.train_loss[self.current_epoch] = train_loss

        if test_loss:
            self.test_loss[self.current_epoch] = test_loss

        if _print:
            print()
            print('Loss \t train_loss \t test_loss')
            print('eopch\t {:.4f}     \t {:.4f}'.format(self.train_loss[self.current_epoch],
                                                        self.test_loss[self.current_epoch]))

    def write_assment_ref(self, _sam, _erags, _scc, _qn, _print=True):
        s = {'min': min(_sam), 'max': max(_sam), 'mean': mean(_sam), 'median': median(_sam)}
        self.sam[self.current_epoch] = s

        e = {'min': min(_erags), 'max': max(_erags), 'mean': mean(_erags), 'median': median(_erags)}
        self.erags[self.current_epoch] = e

        s2 = {'min': min(_scc), 'max': max(_scc), 'mean': mean(_scc), 'median': median(_scc)}
        self.scc[self.current_epoch] = s2

        q = {'min': min(_qn), 'max': max(_qn), 'mean': mean(_qn), 'median': median(_qn)}
        self.qn[self.current_epoch] = q

        if _print:
            print()
            print('Reference\t min\t    max\t      mean\t      meadian\t')
            print('sam      \t {:.4f}\t  {:.4f}\t  {:.4f}\t  {:.4f}'.format(s['min'], s['max'], s['mean'], s['median']))
            print('erags    \t {:.4f}\t  {:.4f}\t  {:.4f}\t  {:.4f}'.format(e['min'], e['max'], e['mean'], e['median']))
            print('scc      \t {:.4f}\t  {:.4f}\t  {:.4f}\t  {:.4f}'.format(s2['min'], s2['max'], s2['mean'], s2['median']))
            print('q4      \t {:.4f}\t  {:.4f}\t  {:.4f}\t  {:.4f}'.format(q['min'], q['max'], q['mean'], q['median']))

    def write_assment_noref(self, _qnr, _d_lambda, _d_s, _print=True):
        self.qnr[self.current_epoch] = _qnr
        self.d_lambda[self.current_epoch] = _d_lambda
        self.d_s[self.current_epoch] = _d_s

        if _print:
            print()
            print('No-reference\t qnr\t    d_lambda\t   d_s\t')
            print('values      \t {:.4f} \t{:.4f}  \t{:.4f}'.format(_qnr, _d_lambda, _d_s))

    def update_epoch(self):
        self.current_epoch += 1
        try:
            with open(join(self.output_dir, 'recorder.model'), 'wb') as fb:
                pickle.dump(self, fb)
        except Exception as e:
            pass
