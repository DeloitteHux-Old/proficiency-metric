# predictor evaluation

# Author:: Sam Steingold (<sds@magnetic.com>)
# Copyright:: Copyright (c) 2014 Magnetic Media Online, Inc.
# License:: Apache License, Version 2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import sys
import operator
import util
import runstat
import bisect
import subprocess
import argparse
import os
import random
import itertools
import csv

def file2path (x,name):
    """Convert file or string to a path.
    If argument is a file, close it and return the name.
    """
    if isinstance(x,file):
        x.close()
        return x.name
    if isinstance(x,str):
        return x
    raise ValueError(name,x)

class VowpalWabbitPipe (object):
    logger = util.get_logger("VowpalWabbitPipe")
    devnull = open(os.devnull,'wb')

    aparse = argparse.ArgumentParser(add_help=False)
    aparse.add_argument('-vw-exe',required=True,type=argparse.FileType('r'),
                        help='VowpalWabbit executable path')
    aparse.add_argument('-vw-model',required=True,type=argparse.FileType('r'),
                        help='VowpalWabbit model path')

    def __init__ (self, model = None, vw = None, parser = None, logger = None):
        if vw is None:
            self.vw = parser.vw_exe = file2path(parser.vw_exe,"VowpalWabbit.exe")
        else:
            self.vw = file2path(vw,"VowpalWabbit.exe")
        if model is None:
            self.model = parser.vw_model = file2path(parser.vw_model,"VowpalWabbit.model")
        else:
            self.model = file2path(model,"VowpalWabbit.model")
        self.logger = VowpalWabbitPipe.logger if logger is None else logger
        cmd = [self.vw,"--initial_regressor",self.model,"--raw_predictions",
               "/dev/stdout","--testonly"]
        util.reading(self.model,self.logger)
        self.logger.info("Running %s",cmd)
        self.s = subprocess.Popen(cmd,stdout=subprocess.PIPE,stdin=subprocess.PIPE,
                                  stderr=VowpalWabbitPipe.devnull)
        self.logger.info("Started PID=%d",self.s.pid)
        self.model = model
        self.called = 0
        self.pending = 0

    def __str__ (self):
        return "VowpalWabbitPipe(pid=%d,exe=%s,model=%s,called=%d,pending=%d)" % (
            self.s.pid,self.vw,self.model,self.called,self.pending)

    def send (self, example):
        self.pending += 1
        self.s.stdin.write(example + '\n')

    def recv (self):
        if self.pending <= 0:
            raise ValueError("VowpalWabbitPipe.recv: nothing is pending",self)
        self.pending -= 1
        self.called += 1
        return float(self.s.stdout.readline().split()[0])

    def eval (self, example):
        if self.pending >= 0:
            # the return value relates to the first pending example, not this one!
            raise ValueError("VowpalWabbitPipe.eval: pending",self,example)
        self.send(example)
        return self.recv()

    def close (self):
        self.s.stdin.close()
        left = len([s for s in self.s.stdout])
        if left > 0:
            self.logger.warning("PID=%d: %d lines remained in the pipe",self.s.pid,left)
        self.s.stdout.close()
        self.logger.info("Waiting for %s to finish",str(self))
        if self.s.wait() != 0:
            raise Exception("VowpalWabbitPipe.close",self.s.pid,self.s.returncode)

    def __del__ (self):
        self.close()

#pylint: disable=no-member
import cffi
class VowpalWabbitFFI (object):
    logger = util.get_logger("VowpalWabbitFFI")

    aparse = argparse.ArgumentParser(add_help=False)
    aparse.add_argument('-vw-build',required=True,help='VowpalWabbit build path')
    aparse.add_argument('-vw-model',required=True,type=argparse.FileType('r'),
                        help='VowpalWabbit model path')

    def __init__(self, model = None, build = None, parser = None, logger = None):
        if build:
            self.build = build
        elif parser:
            self.build = parser.vw_build
        else:
            self.build = None
        if self.build == '/':
            self.build = None
        if model:
            self.model = file2path(model,"VowpalWabbit.model")
        elif parser:
            self.model = parser.vw_model = file2path(parser.vw_model,"VowpalWabbit.model")
        else:
            self.model = None
        self.logger = VowpalWabbitFFI.logger if logger is None else logger
        ffi = cffi.FFI()        # create ffi wrapper object
        # https://github.com/JohnLangford/vowpal_wabbit/blob/master/vowpalwabbit/vwdll.h
        ffi.cdef('''
            void* VW_InitializeA(char *);
            void* VW_ReadExampleA(void*, char*);
            float VW_Predict(void*, void*);
            void VW_FinishExample(void*, void*);
            void VW_Finish(void*);
        ''')
        self.vwffilib = ffi.verify('''
            typedef short char16_t;
            #define bool int
            #define true (1)
            #define false (0)
            #include "vwdll.h"
        ''',include_dirs=[os.path.join(self.build,"vowpalwabbit")
                          if self.build else "/usr/include/vowpalwabbit"],
            library_dirs=[os.path.join(self.build,"vowpalwabbit")
                          if self.build else "/usr/lib64"],
            libraries=["vw_c_wrapper", "vw", "allreduce"],
            ext_package='vw')
        self._vw = self.vwffilib.VW_InitializeA(
            "--testonly --raw_predictions"+
            (" --initial_regressor "+self.model if self.model else ""))
        if not self._vw:
            raise ValueError("VW_InitializeA",model)
        self.called = 0
        self.saved = list()

    def eval (self, example):
        vwex = self.vwffilib.VW_ReadExampleA(self._vw, example)
        score = self.vwffilib.VW_Predict(self._vw, vwex)
        self.vwffilib.VW_FinishExample(self._vw, vwex)
        self.called += 1
        return score

    def send (self, example):
        self.saved.append(self.eval(example))

    def recv (self):
        try:
            return self.saved.pop()
        except IndexError:
            raise ValueError("VowpalWabbitFFI.recv: nothing is pending",self)

    def close (self):
        if self._vw:
            self.vwffilib.VW_Finish(self._vw)
            self._vw = None

    def __del__ (self):
        self.close()

    def __str__ (self):
        return "VowpalWabbitFFI(%s,build=%s,model=%s,called=%d%s)" % (
            self._vw,self.build,self.model,self.called,
            (",saved=%f" % (self.saved) if self.saved else ""))

    @staticmethod
    def test (build):
        vw_model_test = VowpalWabbitFFI(build=build)
        print vw_model_test
        score = vw_model_test.eval("1 |s p^the_man w^the w^man |t p^un_homme w^un w^homme")
        vw_model_test.close()
        print vw_model_test
        assert 0.0==score

class VowpalWabbit (object):
    aparse = argparse.ArgumentParser(add_help=False)
    aparse.add_argument('-vw-model',required=True,type=argparse.FileType('r'),
                        help='VowpalWabbit model path')
    gr = aparse.add_mutually_exclusive_group(required=True)
    gr.add_argument('-vw-build',help='VowpalWabbit build path')
    gr.add_argument('-vw-exe',type=argparse.FileType('r'),
                    help='VowpalWabbit executable path')

    @staticmethod
    def get (parser, logger = None):
        if parser.vw_build:
            return VowpalWabbitFFI(parser=parser,logger=logger)
        else:
            return VowpalWabbitPipe(parser=parser,logger=logger)

def safe_div (a, b):
    return 0 if a == 0 else float(a) / b


class RandomPair (object):
    def __init__ (self, klass, title, **kwargs):
        self.objects = [klass(title + "#" + str(i), **kwargs) for i in range(2)]

    def add (self, *posargs, **kwargs):
        self.objects[random.randint(0,1)].add(*posargs,**kwargs)

    def __str__ (self):
        return "\n".join([str(o) for o in self.objects])

    @staticmethod
    def stat (pairs):
        'Return stats for all_metrics() call on each element of each pair.'
        ret = dict()
        for pair in pairs:
            m1,m2 = [o.all_metrics() for o in pair.objects]
            for metric,value in m1.iteritems():
                try:
                    rs = ret[metric]
                except KeyError:
                    rs = ret[metric] = runstat.NumStat('RP')
                rs.add(value)
            for metric,value in m2.iteritems():
                ret[metric].add(value)
        return ret

    @staticmethod
    def stat2string (pairs):
        if len(pairs) == 0:
            return ""
        sp = RandomPair.stat(pairs)
        w = max([len(m) for m in sp.iterkeys()])
        return "\n"+"\n".join([" * %*s %s" % (w,m,rs.__str__("{:.2%}".format))
                               for (m,rs) in sp.iteritems()])

class ConfusionMX (object):
    logger = util.get_logger("ConfusionMX")

    def __init__ (self, title, NumRP=0):
        self.title = title
        self.mx = dict()
        self.afreq = dict()
        self.pfreq = dict()
        self.halves = [RandomPair(ConfusionMX,title+"/half",NumRP=0)
                       for _ in range(NumRP)]

    def add (self, actual, predicted, weight = 1):
        assert weight > 0
        self.afreq[actual] = self.afreq.get(actual,0) + 1
        self.pfreq[predicted] = self.pfreq.get(predicted,0) + 1
        self.mx[(actual,predicted)] = self.mx.get((actual,predicted),0) + weight
        for rp in self.halves:
            rp.add(actual, predicted, weight)

    # compute the marginal distributions
    def marginals (self):
        actual = dict()
        predicted = dict()
        for ((a,p),w) in self.mx.iteritems():
            actual[a] = actual.get(a,0) + w
            predicted[p] = predicted.get(p,0) + w
        fa = sum(self.afreq.itervalues())
        fp = sum(self.pfreq.itervalues())
        assert fa == fp
        total = sum(self.mx.itervalues())
        sa = sum(actual.itervalues())
        sp = sum(predicted.itervalues())
        assert abs(total - sa) < sys.float_info.epsilon * (fa+fp)
        assert abs(total - sp) < sys.float_info.epsilon * (fa+fp)
        return (total,actual,predicted)

    @staticmethod
    def random_accuracy (actual):
        "The accuracy for the random predictor with the correct marginals."
        total = float(sum(actual.itervalues()))
        ret = 0
        for a in actual.itervalues():
            a /= total
            ret += a * a
        return ret

    def proficiency (self):
        if len(self.mx) == 0:
            return float("NaN")
        total,actual,predicted = self.marginals()
        ConfusionMX.logger.debug("proficiency({}): total={}"
                                 "\n    actual:{}"
                                 "\n predicted:{}"
                                 .format(self.title,total,actual,predicted))
        actual_entropy = 0
        for w in actual.itervalues():
            actual_entropy += w * math.log(w)
        total = float(total)
        actual_entropy = math.log(total) - actual_entropy / total
        if actual_entropy == 0:
            return 0
        mutual_information = 0
        for ((a,p),w) in self.mx.iteritems():
            ConfusionMX.logger.debug("{} {} {} {} {}"
                                     .format(a,p,w,actual[a],predicted[p]))
            mutual_information += w * math.log(total * w / (actual[a] * predicted[p]))
        mutual_information /= total
        return mutual_information / actual_entropy

    def accuracy2 (self):
        if len(self.mx) == 0:
            return (0,0)
        total = 0
        correct = 0
        for ((a,p),w) in self.mx.iteritems():
            total += w
            if a == p:
                correct += w
        return (correct,total)

    def accuracy (self):
        correct,total = self.accuracy2()
        if total == 0:
            return float("NaN")
        return float(correct) / total

    # http://en.wikipedia.org/wiki/Phi_coefficient
    # same as abs(mcc) for binary classification
    def phi (self):
        if len(self.mx) == 0:
            return float("NaN")
        total,actual,predicted = self.marginals()
        chi = 0
        total = float(total)
        for (pl,pw) in predicted.iteritems():
            for (al,aw) in actual.iteritems():
                e = pw * aw / total
                d = self.mx.get((al,pl),0) - e
                chi += d * d / e
        return math.sqrt(chi / total)

    # http://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    def mcc (self):
        if len(self.mx) == 0:
            return float("NaN")
        _total,actual,predicted = self.marginals()
        if len(actual) != 2 or len(predicted) != 2:
            return float("NaN")
        pos,neg = actual.keys()
        return ((self.mx.get((pos,pos),0) * self.mx.get((neg,neg),0) -
                 self.mx.get((pos,neg),0) * self.mx.get((neg,pos),0)) /
                math.sqrt(actual[pos] * actual[neg] * predicted[pos] * predicted[neg]))

    # https://en.wikipedia.org/wiki/Binary_classification
    def binary_metrics (self):
        if len(self.mx) == 0:
            return None
        total,actual,predicted = self.marginals()
        if (len(actual) != 2 or len(predicted) != 2 or
            sorted(actual.keys()) != [False, True] or
            sorted(predicted.keys()) != [False, True]):
            return None
        tp = self.mx.get((True,True),0)
        fn = self.mx.get((True,False),0)
        fp = self.mx.get((False,True),0)
        tn = self.mx.get((False,False),0)
        predictedPositive = tp + fp
        predictedNegative = fn + tn
        actualPositive = tp + fn
        actualNegative = fp + tn
        # correct = tp + tn
        return {'precision': safe_div(tp, predictedPositive),
                'specificity': safe_div(tn, actualNegative),
                'recall': safe_div(tp, actualPositive),
                'npv': safe_div(tn, predictedNegative),
                'lift': safe_div(tp * total,predictedPositive * actualPositive)}

    def is_binary (self):
        return len(self.afreq) == 2 and len(self.pfreq) == 2

    def all_metrics (self):
        if self.is_binary():
            ret = self.binary_metrics()
            ret["mcc"] = self.mcc()
        else:
            ret = dict([("phi",self.phi())])
        ret["proficiency"] = self.proficiency()
        ret["accuracy"] = self.accuracy()
        return ret

    def __str__ (self):
        bm = self.binary_metrics()
        if bm is None:
            bm = ""
        else:
            bm = "\n  Pre={:.2%} Spe={:.2%} Rec={:.2%} Npv={:.2%} Lift={:.2f}".format(
                bm['precision'],bm['specificity'],bm['recall'],bm['npv'],bm['lift'])
        if self.is_binary():
            mcc = self.mcc()
            freq = sum(self.afreq.itervalues()) + sum(self.pfreq.itervalues())
            assert abs(abs(mcc) - self.phi()) < sys.float_info.epsilon * freq
            correlation = "MCC={:.2%}".format(mcc)
        else:
            correlation = "Phi={:.2f}".format(self.phi())
        total,actual,predicted = self.marginals()
        return ("ConfusionMX({}/{:,d}/{:,d}/{:,d}*{:,d}): Pro={:.2%} Acc={:.2%}({:.2%}) ".format(
            self.title,len(self.mx),total,len(actual),len(predicted),
            self.proficiency(),self.accuracy(),ConfusionMX.random_accuracy(actual))
                + correlation + bm + RandomPair.stat2string(self.halves))

    @staticmethod
    def read_VW_demo_mnist (ins):
        "Parse the confusion matrix as printed by vw/demo/mnist/Makefile."
        header = ins.readline().split()
        if len(header) == 0:
            return None
        cmx = ConfusionMX(header[0])
        assert header[1] == 'test'
        assert header[2] == 'errors:'
        errors = int(header[3])
        assert header[4] == 'out'
        assert header[5] == 'of'
        outof = int(header[6])
        assert len(header) == 7
        assert ins.readline().strip() == 'confusion matrix (rows = truth, columns = prediction):'
        first = ins.readline().split()
        for i in range(len(first)):
            w = int(first[i])
            if w > 0:
                cmx.add(i,0,w)
        for i in range(1,len(first)):
            line = ins.readline().split()
            for j in range(len(first)):
                w = int(line[j])
                if w > 0:
                    cmx.add(j,i,w)
        correct,total = cmx.accuracy2()
        assert outof == total
        assert errors == total - correct
        return cmx

    @staticmethod
    def score_vw_oaa (predictions, base=None, NumRP=500):
        """Score $(vw -t) output from a $(vw --oaa) model.
        IDs should be the true values, unless base is supplied."""
        cmx = ConfusionMX(os.path.basename(predictions),NumRP=NumRP)
        if base is not None:
            base = ord(base[0])-1
        util.reading(predictions,ConfusionMX.logger)
        with open(predictions) as ins:
            for line in ins:
                vals = line.split()
                cmx.add(int(vals[1]) if base is None else ord(vals[1][0])-base,
                        int(float(vals[0])))
        return cmx

    @staticmethod
    def test (sample_size,num_cat):
        cmx = ConfusionMX("1.Random")
        for _i in range(sample_size):
            cmx.add(random.randint(1,num_cat),random.randint(1,num_cat))
        print cmx
        cmx = ConfusionMX("1.Perfect")
        for _i in range(sample_size):
            c = random.randint(1,num_cat)
            cmx.add(c,c)
        print cmx
        cmx = ConfusionMX("1.Mislabled")
        for _i in range(sample_size):
            c = random.randint(1,num_cat)
            cmx.add(c,c%num_cat+1)
        print cmx

class MuLabCat (object):        # MultiLabelCategorization
    logger = util.get_logger("MuLabCat")

    def __init__ (self, title, reassign = True, NumRP = 0):
        self.title = title
        self.match = dict()
        self.actual = dict()
        self.predicted = dict()
        self.tp = dict()        # true positives
        self.observations = 0
        self.reassign = reassign
        self.ace = runstat.NumStat("actual: cat/ex",integer=True)
        self.pce = runstat.NumStat("predicted: cat/ex",integer=True)
        self.halves = [RandomPair(MuLabCat,title+"/half",NumRP=0,reassign=False)
                       for _ in range(NumRP)]

    def taxonomy (self):
        taxonomy = set(self.predicted.iterkeys())
        taxonomy.update(self.actual.iterkeys())
        return taxonomy

    def check_taxonomy (self, taxonomy = None):
        if taxonomy is None:
            taxonomy = self.taxonomy()
        missing = taxonomy.difference(self.actual.iterkeys())
        if len(missing) > 0:
            MuLabCat.logger.warn("Predicted but not Actual (%d): %s",
                                 len(missing),list(missing))
        missing = taxonomy.difference(self.predicted.iterkeys())
        if len(missing) > 0:
            MuLabCat.logger.warn("Actual but not Predicted (%d): %s",
                                 len(missing),list(missing))
        return taxonomy

    def add (self, actual, predicted):
        self.observations += 1
        for p in predicted:
            for a in actual:
                self.tp[(a,p)] = self.tp.get((a,p),0) + 1
        for a in actual:
            self.actual[a] = self.actual.get(a,0) + 1
        for p in predicted:
            self.predicted[p] = self.predicted.get(p,0) + 1
        for c in actual & predicted:
            self.match[c] = self.match.get(c,0) + 1
        self.ace.add(len(actual))
        self.pce.add(len(predicted))
        for rp in self.halves:
            rp.add(actual, predicted)

    def precision (self):
        return safe_div(sum(self.match.itervalues()),
                        sum(self.predicted.itervalues()))

    def recall (self):
        return safe_div(sum(self.match.itervalues()),
                        sum(self.actual.itervalues()))

    def f1score (self):
        m = sum(self.match.itervalues())
        a = sum(self.actual.itervalues())
        p = sum(self.predicted.itervalues())
        return (safe_div(2*m,a+p),safe_div(m,p),safe_div(m,a))

    def proficiency2 (self):
        "weighted average of n^2 proficiencies by entropy*entropy"
        top = 0
        ae = dict([(c,util.bin_entropy(self.observations,n))
                   for (c,n) in self.actual.iteritems()])
        spe = 0                 # sum of predicted entropies
        for (pc,pn) in self.predicted.iteritems():
            pe = util.bin_entropy(self.observations,pn)
            spe += pe
            if pe > 0:
                for (ac,an) in self.actual.iteritems():
                    top += ae[ac] * pe * util.bin_mutual_info(
                        self.observations,pn,an,self.tp.get((ac,pc),0))
        return top / (spe * sum(ae.itervalues()))

    def assignment (self):
        # https://pypi.python.org/pypi/munkres
        import munkres
        taxonomy = list(self.check_taxonomy())
        costs = []
        actual_entropy = []
        for ac in taxonomy:
            an = self.actual.get(ac,0)
            ae = util.bin_entropy(self.observations,an)
            actual_entropy.append(ae)
            if ae == 0:
                costs.append([0 for _pc in taxonomy])
            else:
                # negative MI because munkres minimizes costs
                costs.append([- util.bin_mutual_info(
                    self.observations,self.predicted.get(pc,0),
                    an,self.tp.get((ac,pc),0))
                              for pc in taxonomy])
        m = munkres.Munkres()
        indexes = m.compute(costs)
        mutual_information = 0
        reassigned = []
        for row, col in indexes:
            mutual_information += - costs[row][col]
            if row != col:
                ac = taxonomy[row]
                pc = taxonomy[col]
                c = -100*costs[row][col]
                if c > 0:
                    c /= actual_entropy[row]
                reassigned.append((c,ac,self.actual.get(ac,0),pc,self.predicted.get(pc,0)))
        if len(reassigned) > 0:
            reassigned.sort(key=operator.itemgetter(0),reverse=True)
            MuLabCat.logger.warn("Reassigned %d categories:\n%s",len(reassigned),"\n".join(
                ["  Proficiency=%.2f%%: Actual [%s](%d) = Predicted [%s](%d)" % (p,ac,an,pc,pn)
                 for (p,ac,an,pc,pn) in reassigned if (p >= 10 and an >= 5 and pn >= 5)]))
        actual_entropy = sum(actual_entropy)
        return (taxonomy,indexes, 0 if actual_entropy == 0 else mutual_information / actual_entropy)

    def proficiency_assigned (self):
        _taxonomy,_indexes,proficiency = self.assignment()
        return proficiency

    def proficiency_raw (self):
        mutual_information = 0.0
        actual_entropy = 0.0
        for (c,an) in self.actual.iteritems():
            ae = util.bin_entropy(self.observations,an)
            actual_entropy += ae
            mutual_information += util.bin_mutual_info(
                self.observations,self.predicted.get(c,0),an,self.tp.get((c,c),0))
        if actual_entropy == 0:
            return 0
        return mutual_information / actual_entropy

    def all_metrics (self):
        f1,p,r = self.f1score()
        return dict([("proficiency",self.proficiency_raw()),
                     ("f1score",f1),("precision",p),("recall",r)])

    def __str__ (self):
        m = sum(self.match.itervalues())
        a = sum(self.actual.itervalues())
        p = sum(self.predicted.itervalues())
        taxonomy = self.taxonomy()
        aec = runstat.NumStat("ex/cat",values=self.actual.itervalues(),integer=True)
        am = len(taxonomy.difference(self.actual.iterkeys()))
        for _ in range(am):
            aec.add(0)
        pec = runstat.NumStat("ex/cat",values=self.predicted.itervalues(),integer=True)
        pm = len(taxonomy.difference(self.predicted.iterkeys()))
        for _ in range(pm):
            pec.add(0)
        toSt = "{:.2f}".format
        return ("MuLabCat({:}:m={:,d}/{:,d};a={:,d}/{:,d};p={:,d}/{:,d}):"
                " Pre={:.2%} Rec={:.2%} (F1={:.2%}) Pro={:.2%}{:s}"
                "\n    {:} {:}\n {:} {:}{:}".format(
                    self.title,len(self.match),m,len(self.actual),a,
                    len(self.predicted),p,safe_div(m,p),safe_div(m,a),
                    safe_div(2*m,a+p),self.proficiency_raw(),
                    "/{:.2%}".format(self.proficiency_assigned()) if self.reassign else "",
                    self.ace.__str__(toSt),aec.__str__(toSt),
                    self.pce.__str__(toSt),pec.__str__(toSt),
                    RandomPair.stat2string(self.halves)))

    @staticmethod
    def erd (actual, predicted, delimiter='\t', idcol=0, catcol=2, NumRP=500):
        """Score one file against another.
        File format: ...<id>...<category>...
        """
        mlc = MuLabCat(util.title_from_2paths(actual,predicted),
                       reassign=False,NumRP=NumRP)
        adict = util.read_multimap(actual,delimiter,col1=idcol,col2=catcol,logger=MuLabCat.logger)
        pdict = util.read_multimap(predicted,delimiter,col1=idcol,col2=catcol,logger=MuLabCat.logger)
        for obj,acat in adict.iteritems():
            mlc.add(acat,pdict.get(obj,frozenset()))
        for obj,pcat in pdict.iteritems():
            if obj not in adict:
                mlc.add(frozenset(),pcat)
        return mlc

    @staticmethod
    def score (actual, predicted, delimiter='\t', abeg=1, pbeg=1, NumRP=500):
        """Score one file against another.
        File format: <query string>[<delimiter><category>]+
        The queries in both files must be identical.
        abeg and pbeg are the columns where actual and pedicted categories start.
        """
        mlc = MuLabCat(util.title_from_2paths(actual,predicted),NumRP=NumRP)
        util.reading(actual,MuLabCat.logger)
        util.reading(predicted,MuLabCat.logger)
        empty = frozenset([''])
        with open(actual) as af, open(predicted) as pf:
            for sa,sp in itertools.izip_longest(csv.reader(af,delimiter=delimiter),
                                                csv.reader(pf,delimiter=delimiter)):
                if sa is None or sp is None:
                    raise ValueError("uneven files",af,pf,sa,sp)
                if sa[0] != sp[0]:
                    raise ValueError("query string mismatch",sa,sp)
                mlc.add(frozenset(sa[abeg:])-empty,frozenset(sp[pbeg:])-empty)
        return mlc

    @staticmethod
    def random_stats (actual, repeat=1000, delimiter='\t', abeg=1):
        "Score a file against shuffled self."
        util.reading(actual,MuLabCat.logger)
        nca = runstat.NumStat("cat/ex",integer=True)
        empty = frozenset([''])
        data = list()
        with open(actual) as af:
            for sa in csv.reader(af,delimiter=delimiter):
                sa = frozenset(sa[abeg:])-empty
                data.append(sa)
                nca.add(len(sa))
        MuLabCat.logger.info("random_stats({:,d}): Read {:,d} lines: {:}".format(
            repeat,len(data),nca.__str__("{:.2f}".format)))
        prre = runstat.NumStat("prec/recall")
        prof = runstat.NumStat("proficiency")
        for _ in range(repeat):
            indexes = range(len(data))
            random.shuffle(indexes)
            mlc = MuLabCat('shuffle+'+actual,reassign=False)
            for i in range(len(data)):
                mlc.add(data[i],data[indexes[i]])
            # MuLabCat.logger.info("%s",mlc)
            f1,p,r = mlc.f1score()
            assert f1 == p and f1 == r
            prre.add(p)
            prof.add(mlc.proficiency_raw())
        MuLabCat.logger.info("%d runs:\n %s\n %s",repeat,
                             prof.__str__("{:.2%}".format),
                             prre.__str__("{:.2%}".format))

    @staticmethod
    def category_set (taxonomy_size, max_cat = 4):
        return frozenset(random.sample(range(taxonomy_size),random.randint(1,max_cat)))

    @staticmethod
    def test (sample_size, taxonomy_size):
        mlc = MuLabCat("MLC.Perfect")
        for i in range(sample_size):
            s = MuLabCat.category_set(taxonomy_size)
            mlc.add(s,s)
        print mlc
        mlc = MuLabCat("MLC.Mislabled")
        relabling = range(taxonomy_size)
        random.shuffle(relabling)
        print "Relabling: %s" % (relabling)
        for i in range(sample_size):
            s = MuLabCat.category_set(taxonomy_size)
            mlc.add(s,frozenset([relabling[i] for i in s]))
        print mlc
        mlc = MuLabCat("MLC.Random")
        for i in range(sample_size):
            mlc.add(MuLabCat.category_set(taxonomy_size),MuLabCat.category_set(taxonomy_size))
        print mlc
        mlc = MuLabCat("MLC.NotOne")
        for i in range(sample_size):
            mlc.add(frozenset(range(taxonomy_size))-frozenset([random.randint(1,taxonomy_size)-1]),
                    frozenset(range(taxonomy_size))-frozenset([random.randint(1,taxonomy_size)-1]))
        print mlc


import abc

class LqObservations (object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def __init__ (self): pass
    @abc.abstractmethod
    def add (self, actual, score, weight = 1): raise NotImplementedError("LqObservations.add")
    @abc.abstractmethod
    def len (self): raise NotImplementedError("LqObservations.len")
    @abc.abstractmethod
    def size (self): raise NotImplementedError("LqObservations.size")
    @abc.abstractmethod
    def baseRate (self): raise NotImplementedError("LqObservations.size")
    @abc.abstractmethod
    def lq (self, thresholds, logger, title): raise NotImplementedError("LqObservations.lq")

class LqObExact (LqObservations):
    def __init__ (self):
        super(LqObExact, self).__init__()
        self.observations = []

    def add (self, actual, score, weight = 1):
        self.observations.append((actual, score, weight))

    def len (self):
        return len(self.observations)

    def size (self):
        return sum([weight for _a,_s,weight in self.observations])

    @staticmethod
    def sum (observations, beg = 0, end = None):
        if observations == []:
            return [0,0,0,0]
        return [sum(x) for x in zip(*[(1 if isTarget else 0, 1,
                                       weight if isTarget else 0, weight)
                                      for isTarget, _score, weight in
                                      observations[beg:len(observations)
                                                   if end is None else end]])]

    def baseRate (self):
        tc, co, tw, ws = LqObExact.sum(self.observations)
        return (float(tc)/co, float(tw)/ws)

    def bestLift (self):        # find the location where lift'=1
        ta = 0.0
        su = 0.0
        prev_score = None
        best_cut = None
        best_lift = None
        best_pos = None
        best_diff = 0
        pos = 0
        _stc, _sco, stw, sws = LqObExact.sum(self.observations)
        if stw == 0:
            return (None, None, None)
        self.observations.sort(key=operator.itemgetter(1),reverse=True)
        for istarget, score, weight in self.observations:
            if istarget:
                ta += weight
            su += weight
            diff = ta / stw - su / sws
            if best_diff < diff:
                best_lift = (ta * sws) / (su * stw)
                best_pos = pos
                best_cut = 0.5 * (score + (score if prev_score is None else prev_score))
            pos += 1
            prev_score = score
            if 2*pos > len(self.observations):
                break
        return (best_cut,best_pos,best_lift)

    def lq (self, thresholds, logger, title):
        if self.observations == []:
            return (float('NaN'), 0, 0, [])
        brc,brw = self.baseRate()
        if thresholds is None:
            thresholds = [brc] if brc == brw else [brc,brw]
        self.observations.sort(key=operator.itemgetter(1),reverse=True)
        targetLevel = 0
        cph = 0
        totalWeight = 0
        for isTarget, _score, weight in self.observations:
            if isTarget:
                targetLevel += weight
            cph += targetLevel
            totalWeight += weight
        logger.info("%s: cph=%g targetLevel=%g totalWeight=%g",
                    title,cph,targetLevel,totalWeight)
        if targetLevel == 0:
            return (float('NaN'), brc, brw, [])
        lq = ((2.0*cph - targetLevel) / (targetLevel * totalWeight) - 1) / (1 - brw)
        mxl = []
        total = self.len()
        for threshold in thresholds:
            mx = ConfusionMX("{} at {:.2%}".format(title,threshold))
            mxl.append(mx)
            count = 0
            for isTarget, _score, weight in self.observations:
                count += 1
                mx.add(isTarget, total * threshold >= count, weight)
        return (lq, brc, brw, mxl)

class LqObBinned (LqObservations):
    """
    bins=list of (target count, count, target weight sum, weight sum)
    cuts=list of boundaries between bins
    len(bins) == len(cuts) + 1
    """
    def __init__ (self, observations, nbin):
        "observations is the eponymous field in LqObExact"
        n = len(observations)
        if n < nbin:
            raise ValueError("LqObBinned.get: too many bins",nbin,n)
        super(LqObBinned, self).__init__()
        observations.sort(key=operator.itemgetter(1),reverse=True)
        borders = [int(i*n/float(nbin))+1 for i in range(1,nbin)]
        self.cuts = [-(observations[i-1][1]+observations[i][1])*0.5 for i in borders]
        self.bins = ([LqObExact.sum(observations,end=borders[0])]+
                     [LqObExact.sum(observations,beg=borders[i-1],end=borders[i])
                      for i in range(1,nbin-1)]+
                     [LqObExact.sum(observations,beg=borders[nbin-2])])

    def add (self, actual, score, weight = 1):
        pos = bisect.bisect_left(self.cuts,-score)
        tc, co, tw, ws = self.bins[pos]
        self.bins[pos] = ((tc+1 if actual else tc), co + 1,
                          (tw+weight if actual else tw), ws + weight)

    def len (self):
        return sum([co for _tc, co, _tw, _ws in self.bins])

    def size (self):
        return sum([ws for _tc, _co, _tw, ws in self.bins])

    def sum (self, beg = 0, end = None):
        return [sum(x) for x in zip(*self.bins[beg:len(self.bins) if end is None else end])]

    def baseRate (self):
        stc, sco, stw, sws = self.sum()
        return (float(stc)/sco, float(stw)/sws)

    def bestLift (self):        # find the location where lift'=1
        ta = 0.0
        su = 0.0
        best_cut = None
        best_lift = None
        best_pos = None
        best_diff = 0
        pos = 0
        _stc, _sco, stw, sws = self.sum()
        if stw == 0:
            return (None, None, None)
        for _tc, _co, tw, ws in self.bins:
            ta += tw
            su += ws
            diff = ta / stw - su / sws
            if best_diff < diff:
                best_lift = (ta * sws) / (su * stw)
                best_pos = pos
                best_cut = -self.cuts[pos]
            pos += 1
            if 2*pos > len(self.bins):
                break
        return (best_cut,best_pos,best_lift)

    def lq (self, thresholds, logger, title):
        stc, sco, stw, sws = self.sum()
        if stc == 0:
            return (float('NaN'), 0, 0, [])
        # base rates
        brc = float(stc)/sco
        brw = float(stw)/sws
        if thresholds is None:
            thresholds = [brc] if brc == brw else [brc,brw]
        # convert thresholds to whole numbers of bins
        cumco = util.cumsum([co for _tc, co, _tw, _ws in self.bins])
        top = len(self.bins) - 2 # max bin index for thresholds
        thresholds = [min(top,bisect.bisect_left(cumco,th*sco)) for th in thresholds]
        thresholds = util.dedup(thresholds)
        # http://en.wikipedia.org/wiki/Trapezoidal_rule
        cph = sum([(stw - tw/2)*ws for stw,(tw,ws) in
                   zip(util.cumsum([tw for _tc, _co, tw, _ws in self.bins]),
                       [(tw,ws) for _tc, _co, tw, ws in self.bins])])
        cph = float(cph) / (sws*stw)
        logger.info("%s: cph=%g targetLevel=%g totalWeight=%g",
                    title,cph,stw,sws)
        lq = (2*cph - 1) / (1 - brw)
        mxl = []
        for threshold in thresholds:
            mx = ConfusionMX("{:} at {:.2%}".format(title,float(cumco[threshold])/sco))
            mxl.append(mx)
            for i in range(len(self.bins)):
                _tc, _co, tw, ws = self.bins[i]
                if tw > 0:
                    mx.add(True, i <= threshold, tw)
                if ws > tw:
                    mx.add(False, i <= threshold, ws - tw)
        return (lq, brc, brw, mxl)

class LiftQuality (object):
    """
    Keep data in LqObExact when the number of observation is below
    conversion_threshold, then use LqObBinned.
    This assumes that the range of scores is uniform, i.e.,
    no unexpected scores appear after conversion.
    """
    default_conversion_threshold = 1000
    logger = util.get_logger("LiftQuality")

    def __init__ (self, title, trueVal = True,
                  conversion_threshold = default_conversion_threshold):
        self.title = title
        self.trueVal = trueVal
        self.observations = LqObExact()
        self.conversion_threshold = conversion_threshold

    def add (self, actual, score, weight = 1):
        if (self.conversion_threshold is not None and
            self.observations.len() == self.conversion_threshold):
            self.observations = LqObBinned(self.observations.observations,
                                           self.conversion_threshold/10)
        self.observations.add(actual == self.trueVal, score, weight)

    def len (self): return self.observations.len()
    def size (self): return self.observations.size()
    def baseRate (self): return self.observations.baseRate()
    def bestLift (self): return self.observations.bestLift()
    def lq (self, thresholds = None):
        return self.observations.lq(thresholds,LiftQuality.logger,self.title)

    def __str__ (self):
        lq, _brc, brw, mxl = self.lq()
        main = "LiftQuality({}/{:,d} {:.2%}): Lq={:.2%}".format(
            self.title,self.len(),brw,lq)
        if mxl == []:
            return main
        return main + "\n " + "\n ".join(str(mx) for mx in mxl)

    @staticmethod
    def test (scale):
        lq = LiftQuality("random")
        for _i in range(10*scale):
            lq.add(random.randint(1,4) == 1, random.random())
        print lq
        print lq.bestLift()
        lq = LiftQuality("perfect")
        for _i in range(10*scale):
            if random.randint(1,4) == 1:
                lq.add(True, random.uniform(2,3))
            else:
                lq.add(False, random.uniform(0,1))
        print lq
        print lq.bestLift()
        lq = LiftQuality("inverted")
        for _i in range(10*scale):
            if random.randint(1,4) == 1:
                lq.add(True, random.uniform(0,1))
            else:
                lq.add(False, random.uniform(2,3))
        print lq
        print lq.bestLift()

def test ():
    MuLabCat.test(10000, 10)
    ConfusionMX.test(10000,2)
    ConfusionMX.test(10000,5)
    LiftQuality.test(LiftQuality.default_conversion_threshold)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Score Predictions')
    gr = ap.add_mutually_exclusive_group()
    gr.add_argument('-ConfusionMX',help='use ConfusionMX',action="store_true")
    gr.add_argument('-MuLabCat',help='use MuLabCat',action="store_true")
    ap.add_argument('-predictions',help='path to VW predictions file [ConfusionMX]')
    ap.add_argument('-base',help='base character [ConfusionMX]')
    ap.add_argument('-format',help='input file format [MuLabCat]',
                    choices=['erd', 'one'])
    ap.add_argument('-actual',help='path to the actual (annotations) [MuLabCat]')
    ap.add_argument('-predicted',help='path to the predicted [MuLabCat]')
    ap.add_argument('-abeg',type=int,default=1,
                    help='the column in actual where the categories start [MuLabCat]')
    ap.add_argument('-pbeg',type=int,default=1,
                    help='the column in predicted where the categories start [MuLabCat]')
    args = ap.parse_args()
    if args.ConfusionMX:
        if args.predictions is not None:
            print ConfusionMX.score_vw_oaa(args.predictions, args.base)
        else:
            # Parse output from $(make -C vw/demo/mnist raw png)
            vwdemo = os.path.expandvars("$HOME/src/sds-vw/demo")
            if os.path.isdir(vwdemo):
                for (dirpath, _dirnames, filenames) in os.walk(vwdemo):
                    for f in filenames:
                        if f.endswith('.out'):
                            f = os.path.join(dirpath,f)
                            util.reading(f,ConfusionMX.logger)
                            with open(f) as inst:
                                while True:
                                    try:
                                        cm = ConfusionMX.read_VW_demo_mnist(inst)
                                    except Exception as ex:
                                        print ex
                                        break
                                    if cm is None:
                                        break
                                    print cm
                        if f.endswith('.predictions'):
                            try:
                                print ConfusionMX.score_vw_oaa(os.path.join(dirpath,f),args.base)
                            except Exception as ex:
                                print ex
            else:
                ConfusionMX.logger("[%s] does not exist",vwdemo)
    elif args.MuLabCat:
        if args.actual is not None and args.predicted is not None:
            if args.format == 'erd':
                print MuLabCat.erd(args.actual,args.predicted)
            else:
                print MuLabCat.score(args.actual,args.predicted,abeg=args.abeg,pbeg=args.pbeg)
        else:                   # pre-supplied data
            data_dir = "../data"
            # Queries annotated by 3 different people for KDD Cup 2005
            l1 = os.path.join(data_dir,"labeler1.txt")
            l2 = os.path.join(data_dir,"labeler2.txt")
            l3 = os.path.join(data_dir,"labeler3.txt")
            MuLabCat.random_stats(l1)
            MuLabCat.random_stats(l2)
            MuLabCat.random_stats(l3)
            print MuLabCat.score(l1,l2)
            print MuLabCat.score(l2,l3)
            print MuLabCat.score(l3,l1)
            # Hand-annotated 10k queries (Magnetic)
            ann = os.path.join(data_dir,"magnetic_annotated.txt")
            res = os.path.join(data_dir,"magnetic_results.txt")
            res5 = os.path.join(data_dir,"magnetic_results_upto5.txt")
            MuLabCat.random_stats(ann,abeg=2)
            MuLabCat.random_stats(res)
            MuLabCat.random_stats(res5)
            print MuLabCat.score(ann,res,abeg=2)
            print MuLabCat.score(ann,res5,abeg=2)
            # Slovak queries in ERD format
            trecA = os.path.join(data_dir,"Trec_beta_annotations.txt")
            trecR = os.path.join(data_dir,"Trec_beta_results.txt")
            print MuLabCat.erd(trecA,trecR)
            qskA = os.path.join(data_dir,"sk_annotation.txt")
            qskR = os.path.join(data_dir,"sk_results.txt")
            qskRA = os.path.join(data_dir,"sk_results_ascii.txt")
            print MuLabCat.erd(qskA,qskR)
            print MuLabCat.erd(qskA,qskRA)
    else:
        test()
