# predictor evaluation

# Author:: Sam Steingold (<sds@magnetic.com>)
# Copyright:: Copyright (c) 2014, 2015, 2016 Magnetic Media Online, Inc.
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
import json
import datetime
import time
import progress

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

def parse_vw_line(line):
    vec = line.split('|')
    target,weight,tag = vec[0].split()
    attributes = {"target":target,"weight":weight,"tag":tag}
    for val in vec[1:]:
        nsv = val.split()
        if len(nsv) == 0:
            continue
        elif len(nsv) == 1:
            attributes[nsv[0]] = None
        else:
            attributes[nsv[0]] = "/".join(nsv[1:])
            if len(nsv) > 2:
                for i in xrange(len(nsv)-1):
                    attributes["%s.%s" % (nsv[0],i)] = nsv[i+1]
    return attributes

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
        self.logger = logger or VowpalWabbitPipe.logger
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
        self.logger = logger or VowpalWabbitFFI.logger
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
        self.objects = [klass(title + "#" + str(i), **kwargs) for i in xrange(2)]

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
        return "\n"+"\n".join([" * %*s %s" % (w,m,rs.__str__("{0:.2%}".format))
                               for (m,rs) in sp.iteritems()])

class ConfusionMX (object):
    logger = util.get_logger("ConfusionMX")

    def __init__ (self, title, NumRP=0, logger=None, debug=False):
        self.title = title
        self.mx = dict()
        self.afreq = dict()
        self.pfreq = dict()
        self.halves = [RandomPair(ConfusionMX,title+"/half",NumRP=0)
                       for _ in xrange(NumRP)]
        self.logger = logger or ConfusionMX.logger
        self.debug = debug

    def add (self, actual, predicted, weight = 1):
        """Add an observation.  Actual & predicted should be from the same
        universe for accuracy to make sense.  They should be boolean for
        binary_metrics (Recall, Precision &c) to make sense."""
        if weight <= 0:
            raise ValueError("ConfusionMX.add: bad weight",
                             weight,self,actual,predicted)
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
        assert abs(total - sa) < sys.float_info.epsilon * (total+sa)
        assert abs(total - sp) < sys.float_info.epsilon * (total+sp)
        return (total,actual,predicted)

    @staticmethod
    def random_accuracy (actual, predicted):
        "The accuracy for the random predictors with the correct and predicted marginals."
        total = float(sum(actual.itervalues()))
        retA = 0
        retP = 0
        for a,n in actual.iteritems():
            n /= total
            retA += n * n
            retP += n * predicted.get(a,0)/total
        return retA, retP

    def isempty (self):
        return not self.mx

    # https://en.wikipedia.org/wiki/Information_gain_ratio
    # https://en.wikipedia.org/wiki/Mutual_information
    # https://en.wikipedia.org/wiki/Uncertainty_coefficient
    def information4 (self): # full Venn diagram
        if self.isempty():
            return (None,None,None,None)
        total,actual,predicted = self.marginals()
        if self.debug:
            self.logger.info("Marginals(%s): total=%s"
                             "\n    actual:%s"
                             "\n predicted:%s"
                             "\n             %s\n%s",
                             self.title,total,actual,predicted,
                             " ".join("%5s=%5d" % (p,n)
                                      for p,n in sorted(predicted.iteritems(),
                                                        key=operator.itemgetter(0))),
                             "\n".join("%5s=%5d  %s" % (a,n," ".join(
                                 "%11d" % self.mx.get((a,p),0)
                                 for p in sorted(predicted.iterkeys())))
                                       for a,n in sorted(actual.iteritems(),
                                                         key=operator.itemgetter(0))))
        total = float(total)
        lt = math.log(total)
        joint_entropy = lt - sum(
            w*math.log(w) for w in self.mx.itervalues()) / total
        if joint_entropy == 0:
            return (0,0,0,0)
        actual_entropy = lt - sum(
            w*math.log(w) for w in actual.itervalues()) / total
        predicted_entropy = lt - sum( # aka intrinsic_value
            w*math.log(w) for w in predicted.itervalues()) / total
        mutual_information = sum( # aka information_gain
            w * math.log(total * w / (actual[a] * predicted[p]))
            for ((a,p),w) in self.mx.iteritems()) / total
        if self.debug:
            self.logger.info("Mutual=%g, Actual=%g, Predicted=%g, Joint=%g",
                             mutual_information,actual_entropy,
                             predicted_entropy,joint_entropy)
        return (mutual_information,actual_entropy,predicted_entropy,joint_entropy)

    def information_metrics (self):
        mi,ae,pe,je = self.information4()
        if mi is None:
            return (float("NaN"),float("NaN"),float("NaN"))
        return (0 if ae == 0 else mi/ae, # proficiency
                0 if pe == 0 else mi/pe, # information_gain_ratio
                0 if je == 0 else mi/je) # dependency

    def proficiency (self): return self.information_metrics()[0]
    def igr (self): return self.information_metrics()[1]
    def dependency (self): return self.information_metrics()[2]

    def accuracy2 (self):
        if self.isempty():
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
        if self.isempty():
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
        if self.isempty():
            return float("NaN")
        _total,actual,predicted = self.marginals()
        if (len(actual) != 2 or len(predicted) != 2 or
            sorted(actual.keys()) != sorted(predicted.keys())):
            return float("NaN")
        pos,neg = actual.keys()
        return safe_div(self.mx.get((pos,pos),0) * self.mx.get((neg,neg),0) -
                        self.mx.get((pos,neg),0) * self.mx.get((neg,pos),0),
                        math.sqrt(actual[pos] * actual[neg] *
                                  predicted[pos] * predicted[neg]))

    # https://en.wikipedia.org/wiki/Binary_classification
    def binary_metrics (self):
        if self.isempty() or not self.is_binary():
            return None
        tp = self.mx.get((True,True),0)
        fn = self.mx.get((True,False),0)
        fp = self.mx.get((False,True),0)
        tn = self.mx.get((False,False),0)
        predictedPositive = tp + fp
        predictedNegative = fn + tn
        actualPositive = tp + fn
        actualNegative = fp + tn
        total = tp + fn + fp + tn
        # correct = tp + tn
        return {'precision': safe_div(tp, predictedPositive),
                'specificity': safe_div(tn, actualNegative),
                'recall': safe_div(tp, actualPositive),
                'npv': safe_div(tn, predictedNegative),
                'prevalence': safe_div(actualPositive,total),
                'lift': safe_div(tp * total,predictedPositive * actualPositive)}

    def is_binary (self):
        return (len(self.afreq) == 2 and len(self.pfreq) == 2 and
                sorted(self.afreq.keys()) == [False, True] and
                sorted(self.pfreq.keys()) == [False, True])

    def all_metrics (self):
        ret = self.binary_metrics()
        if ret:
            ret["mcc"] = self.mcc()
        else:
            ret = dict([("phi",self.phi())])
        pro,igr,dep = self.information_metrics()
        ret["proficiency"] = pro
        ret["igr"] = igr
        ret["dependency"] = dep
        ret["accuracy"] = self.accuracy()
        return ret

    def get_res (self, group):
        r = self.all_metrics()
        r["group"] = group
        total,actual,predicted = self.marginals()
        r["observations"] = total
        for a,n in actual.iteritems():
            r["actual."+str(a)] = n
        for p,n in predicted.iteritems():
            r["predicted."+str(p)] = n
        return r

    def __str__ (self):
        bm = self.binary_metrics()
        if bm is None:
            bm = ""
        else:
            bm = "\n  Pre={p:.2%} Spe={s:.2%} Rec={r:.2%} Npv={n:.2%} Lift={l:.2f}".format(p=bm['precision'],s=bm['specificity'],r=bm['recall'],n=bm['npv'],l=bm['lift'])
        if self.is_binary():
            mcc = self.mcc()
            freq = sum(self.afreq.itervalues()) + sum(self.pfreq.itervalues())
            assert abs(abs(mcc) - self.phi()) < sys.float_info.epsilon * freq
            correlation = " MCC={m:.2%}".format(m=mcc)
        else:
            correlation = " Phi={p:.2f}".format(p=self.phi())
        total,actual,predicted = self.marginals()
        pro,igr,dep = self.information_metrics()
        rand_accA, rand_accP = ConfusionMX.random_accuracy(actual,predicted)
        if rand_accP == 0:       # actual & predicted are disjoint
            accuracy = ""
        else:
            accuracy = " Acc={A:.2%}(RA={RA:.2%}, RP={RP:.2%})".format(
                A=self.accuracy(),RA=rand_accA,RP=rand_accP)
        return ("ConfusionMX({n:s}/{l:,d}/{t:,.0f}/{a:,d}*{p:,d}): Pro={P:.2%} IGR={I:.2%} Dep={D:.2%}".format(
            n=self.title,l=len(self.mx),t=total,a=len(actual),p=len(predicted),
            P=pro,I=igr,D=dep)
                + accuracy + correlation + bm + RandomPair.stat2string(self.halves))

    @staticmethod
    def read_vw_demo_mnist (ins, logger = None):
        "Parse the confusion matrix as printed by vw/demo/mnist/Makefile."
        logger = logger or ConfusionMX.logger
        header = ins.readline().split()
        if len(header) == 0:
            return None
        cmx = ConfusionMX(header[0],logger)
        assert header[1] == 'test'
        assert header[2] == 'errors:'
        errors = int(header[3])
        assert header[4] == 'out'
        assert header[5] == 'of'
        outof = int(header[6])
        assert len(header) == 7
        assert ins.readline().strip() == 'confusion matrix (rows = truth, columns = prediction):'
        first = ins.readline().split()
        for i in xrange(len(first)):
            w = int(first[i])
            if w > 0:
                cmx.add(i,0,w)
        for i in xrange(1,len(first)):
            line = ins.readline().split()
            for j in xrange(len(first)):
                w = int(line[j])
                if w > 0:
                    cmx.add(j,i,w)
        correct,total = cmx.accuracy2()
        assert outof == total
        assert errors == total - correct
        return cmx

    @staticmethod
    def print_vw_demo_mnist (fname, logger = None):
        logger = logger or ConfusionMX.logger
        util.reading(fname,logger)
        with open(fname) as inst:
            while True:
                try:
                    cm = ConfusionMX.read_vw_demo_mnist(inst,logger=logger)
                except Exception as ex:
                    print ex
                    break
                if cm is None:
                    break
                print cm

    @staticmethod
    def score_vw_oaa (predictions, base=None, NumRP=0, logger=None):
        """Score $(vw -t) output from a $(vw --oaa) model.
        IDs should be the true values, unless base is supplied."""
        logger = logger or ConfusionMX.logger
        cmx = ConfusionMX(os.path.basename(predictions),NumRP=NumRP,logger=logger)
        if base is not None:
            base = ord(base[0])-1
        util.reading(predictions,logger)
        with open(predictions) as ins:
            for line in ins:
                vals = line.split()
                cmx.add(int(vals[1]) if base is None else ord(vals[1][0])-base,
                        int(float(vals[0])))
        return cmx

    @staticmethod
    def vw_demos (vwdemo, base=None, NumRP=0, logger=None):
        logger = logger or ConfusionMX.logger
        if os.path.isdir(vwdemo):
            for (dirpath, _dirnames, filenames) in os.walk(vwdemo):
                for f in filenames:
                    if f.endswith('.out'):
                        ConfusionMX.print_vw_demo_mnist(os.path.join(dirpath,f),
                                                        logger=logger)
                    if f.endswith('.predictions'):
                        try:
                            print ConfusionMX.score_vw_oaa(
                                os.path.join(dirpath,f),base=base,NumRP=NumRP,
                                logger=logger)
                        except Exception as ex:
                            print ex
        else:
            logger.error("[%s] does not exist",vwdemo)

    @staticmethod
    def test (sample_size,num_cat):
        print "\n *** ConfusionMX(sample_size=%d,num_cat=%d)" % (sample_size,num_cat)
        from string import uppercase
        booleans = [False,True]
        categories = booleans if num_cat == 2 else uppercase
        cmx = ConfusionMX("Random",debug=True)
        for _ in xrange(sample_size):
            cmx.add(categories[random.randint(0,num_cat-1)],
                    categories[random.randint(0,num_cat-1)])
        print cmx, "\n"
        cmx = ConfusionMX("Perfect",debug=True)
        for _ in xrange(sample_size):
            c = categories[random.randint(0,num_cat-1)]
            cmx.add(c,c)
        print cmx, "\n"
        cmx = ConfusionMX("Mislabled",debug=True)
        for _ in xrange(sample_size):
            c = random.randint(0,num_cat-1)
            cmx.add(categories[c],categories[(c+1)%num_cat])
        print cmx, "\n"
        cmx = ConfusionMX("Imperfect",debug=True)
        for _ in xrange(sample_size):
            c = categories[random.randint(0,num_cat-1)]
            cmx.add(c,c if random.choice(booleans) else
                    categories[random.randint(0,num_cat-1)])
        er = (num_cat-1)/(2.0*num_cat)
        mi = (1-er)*math.log((num_cat+1.0)/2) - er*math.log(2)
        print "{cmx:s}\nExpected MI={m:g} Pro={p:.2%} Acc={a:.2%}\n".format(
            cmx=cmx,m=mi,p=mi/math.log(num_cat),a=1-er)


class StreamingCMX (object):
    "Stream, scored using ConfusionMX - grouped."
    def __init__ (self,title,baseout,runid,groups,logger):
        self.cmx = ConfusionMX(title)
        self.logger = logger or util.get_logger("StreamingCMX")
        self.record = 0
        self.cmxd = dict()
        self.baseout = baseout
        self.runid = runid
        self.groups = groups
        self.start = time.time()

    def save (self):
        if self.cmx.isempty():
            self.logger.error("NOT saving empty ConfusionMX to %s",self.baseout)
            return
        self.logger.info("Writing total%s to %s.{json,csv}",
                         " and {g:,d} groups".format(g=len(self.cmxd))
                         if self.cmxd else "", self.baseout)
        self.cmxd["total"] = self.cmx
        resl = sorted([cmx.get_res(group) for group,cmx in self.cmxd.iteritems()],
                      key = operator.itemgetter('observations'),
                      reverse=True)
        fieldnames = ["group", "observations",
                      "proficiency", "igr", "dependency", "accuracy"]
        total,actual,predicted = self.cmx.marginals()
        self.logger.info("Total: {t:,d}\n {an: >10s}: {ac:s}\n {pn: >10s}: {pc:s}".format(
            t=total,an="actual",ac=util.counter2string(actual,maxlen=10),
            pn="predicted",pc=util.counter2string(predicted,maxlen=10)))
        fieldnames += ["actual."+str(a) for a in sorted(actual.iterkeys())]
        fieldnames += ["predicted."+str(p) for p in sorted(predicted.iterkeys())]
        if self.cmx.is_binary():
            fieldnames += ['mcc','precision','recall','specificity',
                           'npv','prevalence','lift']
        else:
            fieldnames.append('phi')
        known_keys = frozenset(self.cmx.get_res("total").iterkeys())
        saved_keys = frozenset(fieldnames)
        if saved_keys != known_keys:
            raise ValueError("StreamingCMX: known/saved keys mismatch",
                             known_keys-saved_keys,saved_keys-known_keys)
        with open(self.baseout+".csv","w") as o:
            w = csv.DictWriter(o, fieldnames=fieldnames)
            w.writeheader()
            for res in resl:
                w.writerow(res)
        util.wrote(self.baseout+".csv",logger=self.logger)
        with open(self.baseout+".json","w") as j:
            json.dump({
                'TimeStamp':datetime.datetime.now().strftime("%F %T"),
                'Elapsed':time.time()-self.start,
                'RunID':self.runid, 'Name':self.cmx.title,
                'Groups':self.groups,
                'groups':resl,
            },j,sort_keys=True,indent=2,separators=(',',':'))
        util.wrote(self.baseout+".json",logger=self.logger)

    def add (self, target, score, weight = 1, group = None):
        self.record += 1
        self.cmx.add(target,score,weight)
        if group:
            try:
                cmx = self.cmxd[group]
            except KeyError:
                cmx = self.cmxd[group] = ConfusionMX(group)
            cmx.add(target,score,weight)

    def report (self):
        self.logger.info("Done {t:s} ({r:,d} records) [{e:s}]\n{cmx:s}".format(
            t=self.cmx.title,r=self.record,e=progress.elapsed(self.start),cmx=self.cmx))
        if self.cmxd:
            pro_ns = runstat.NumStat("Proficiency")
            igr_ns = runstat.NumStat("InfoGainRat")
            dep_ns = runstat.NumStat("Dependency")
            acc_ns = runstat.NumStat("Accuracy")
            for cmx in self.cmxd.itervalues():
                pro,igr,dep = cmx.information_metrics()
                pro_ns.add(pro)
                igr_ns.add(igr)
                dep_ns.add(dep)
                acc_ns.add(cmx.accuracy())
            self.logger.info("%d groups:\n %s\n %s\n  %s\n    %s",
                             len(self.cmxd),pro_ns,igr_ns,dep_ns,acc_ns)
        if self.baseout:
            self.save()

    @staticmethod
    def row2tuple (row, ropa):
        return (ropa.get(row,ropa.tpos),
                [ropa.get(row,spos) for spos in ropa.sposl],
                float(ropa.get(row,ropa.wpos)) if ropa.wpos else 1,
                ropa.get_groups(row))


class MuLabCat (object):        # MultiLabelCategorization
    logger = util.get_logger("MuLabCat")

    def __init__ (self, title, reassign = True, NumRP=0, logger=None):
        self.title = title
        self.match = dict()
        self.actual = dict()
        self.predicted = dict()
        self.tp = dict()        # true positives
        self.observations = 0
        self.reassign = reassign
        self.assignment_cache = None
        self.ace = runstat.NumStat("actual: cat/ex",integer=True)
        self.pce = runstat.NumStat("predicted: cat/ex",integer=True)
        self.halves = [RandomPair(MuLabCat,title+"/half",NumRP=0,reassign=False)
                       for _ in xrange(NumRP)]
        self.logger = logger or MuLabCat.logger
        self.start = time.time()

    def isempty (self):
        return self.observations == 0

    def taxonomy (self):
        taxonomy = set(self.predicted.iterkeys())
        taxonomy.update(self.actual.iterkeys())
        return taxonomy

    def check_taxonomy (self, taxonomy = None):
        if taxonomy is None:
            taxonomy = self.taxonomy()
        missing = taxonomy.difference(self.actual.iterkeys())
        if len(missing) > 0:
            self.logger.warn("Predicted but not Actual (%d): %s",
                             len(missing),list(missing))
        missing = taxonomy.difference(self.predicted.iterkeys())
        if len(missing) > 0:
            self.logger.warn("Actual but not Predicted (%d): %s",
                             len(missing),list(missing))
        return taxonomy

    def add (self, actual, predicted):
        self.assignment_cache = None # invalidate cache
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

    def compute_assignment (self):
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
                reassigned.append((c,ac,self.actual.get(ac,0),
                                   pc,self.predicted.get(pc,0)))
        if len(reassigned) > 0:
            reassigned.sort(key=operator.itemgetter(0),reverse=True)
            self.logger.warn("Reassigned %d categories%s",len(reassigned),"".join(
                ["\n  Proficiency=%.2f%%: Actual [%s](%d) = Predicted [%s](%d)"
                 % (p,ac,an,pc,pn)
                 for (p,ac,an,pc,pn) in reassigned
                 if (p >= 10 and an >= 5 and pn >= 5)]))
        actual_entropy = sum(actual_entropy)
        return (taxonomy,indexes,
                0 if actual_entropy == 0 else mutual_information / actual_entropy)

    def get_assignment (self):
        if not self.reassign:
            return None
        if self.assignment_cache is None:
            self.assignment_cache = self.compute_assignment()
        return self.assignment_cache

    def proficiency_assigned (self):
        assignment = self.get_assignment()
        return assignment and assignment[2]

    def matches_reassigned (self):
        tip = self.get_assignment()
        if tip is None:
            return None
        taxonomy,indexes,_proficiency_assigned = tip
        matched_ass = 0
        matched_old = 0
        reassigned = 0
        for row, col in indexes:
            matched_ass += self.tp.get((taxonomy[row],taxonomy[col]),0)
            matched_old += self.tp.get((taxonomy[row],taxonomy[row]),0)
            if row != col:
                reassigned += 1
        assert matched_old == sum(self.match.itervalues())
        return matched_ass,reassigned

    def proficiency_raw (self):
        mutual_information = 0.0
        actual_entropy = 0.0
        for (c,an) in self.actual.iteritems():
            ae = util.bin_entropy(self.observations,an)
            actual_entropy += ae
            mi = util.bin_mutual_info(
                self.observations,self.predicted.get(c,0),an,self.tp.get((c,c),0))
            mutual_information += mi
        if actual_entropy == 0:
            return 0
        return mutual_information / actual_entropy

    def all_metrics (self):
        f1,p,r = self.f1score()
        return dict([("proficiency",self.proficiency_raw()),
                     ("f1score",f1),("precision",p),("recall",r)])

    def __str__ (self):
        tip = self.get_assignment()
        m = sum(self.match.itervalues())
        a = sum(self.actual.itervalues())
        p = sum(self.predicted.itervalues())
        taxonomy = self.taxonomy()
        aec = runstat.NumStat("ex/cat",values=self.actual.itervalues(),integer=True)
        am = len(taxonomy.difference(self.actual.iterkeys()))
        for _ in xrange(am):
            aec.add(0)
        pec = runstat.NumStat("ex/cat",values=self.predicted.itervalues(),integer=True)
        pm = len(taxonomy.difference(self.predicted.iterkeys()))
        for _ in xrange(pm):
            pec.add(0)
        toSt = "{0:.2f}".format
        def metrics (m,a,p,pro):
            return "Pre={pre:.2%} Rec={rec:.2%} (F1={f1:.2%}) Pro={pro:.2%}".format(
                pre=safe_div(m,p),rec=safe_div(m,a),f1=safe_div(2*m,a+p),pro=pro)
        if tip:
            matched_ass,reassigned = self.matches_reassigned()
            metrics_reassigned = "\n [Reassigned %d: %s]" % (
                reassigned,metrics(matched_ass,a,p,tip[2]))
        else:
            metrics_reassigned = ""
        return ("MuLabCat({n:s}) [{e:s}]\n [t={t:,d};m={m:,d}/{M:,d};a={a:,d}/{A:,d};"
                "p={p:,d}/{P:,d}]: {metrics:s}{reassigned:s}"
                "\n    {ace:s} {aec:s}\n {pce:s} {pec:s}{rp:s}".format(
                    n=self.title,e=progress.elapsed(self.start),
                    t=len(taxonomy),m=len(self.match),M=m,
                    a=len(self.actual),A=a,p=len(self.predicted),P=p,
                    metrics=metrics(m,a,p,self.proficiency_raw()),
                    reassigned=metrics_reassigned,
                    ace=self.ace.__str__(toSt),aec=aec.__str__(toSt),
                    pce=self.pce.__str__(toSt),pec=pec.__str__(toSt),
                    rp=RandomPair.stat2string(self.halves)))

    fieldnames = [
        "title","observations","match.len","match.sum",
        "actual.len","actual.sum","predicted.len","predicted.sum",
        "precision","recall","f1",
        "proficiency.raw","proficiency.assigned",
        "match.sum.assigned","taxonomy","reassigned",
    ]
    def get_dict (self):
        m = sum(self.match.itervalues())
        a = sum(self.actual.itervalues())
        p = sum(self.predicted.itervalues())
        tip = self.get_assignment()
        if tip:
            taxonomy,_indexes,proficiency_assigned = tip
            matched_ass,reassigned = self.matches_reassigned()
        else:
            taxonomy = self.taxonomy()
            proficiency_assigned = None
            matched_ass = None
            reassigned = None
        return {
            "title": self.title,
            "observations": self.observations,
            "match.len": len(self.match),
            "match.sum": m,
            "actual.len": len(self.actual),
            "actual.sum": a,
            "predicted.len": len(self.predicted),
            "predicted.sum": p,
            "precision": safe_div(m,p),
            "recall": safe_div(m,a),
            "f1": safe_div(2*m,a+p),
            "proficiency.raw": self.proficiency_raw(),
            "proficiency.assigned": proficiency_assigned,
            "match.sum.assigned": matched_ass,
            "taxonomy": len(taxonomy),
            "reassigned": reassigned,
            "actual.cat/ex": self.ace.as_dict(),
            "predicted.cat/ex": self.pce.as_dict(),
        }

    def save (self, baseout):
        if self.isempty():
            self.logger.error("NOT saving empty MuLabCat to %s",baseout)
            return
        di = self.get_dict()
        with open(baseout+".csv","w") as o:
            w = csv.DictWriter(o, fieldnames=MuLabCat.fieldnames)
            w.writeheader()
            w.writerow(di)
        util.wrote(baseout+".csv",logger=self.logger)
        with open(baseout+".json","w") as j:
            di['TimeStamp'] = datetime.datetime.now().strftime("%F %T")
            di['Elapsed'] = time.time()-self.start
            json.dump(di,j,sort_keys=True,indent=2,separators=(',',':'))
        util.wrote(baseout+".json",logger=self.logger)

    @staticmethod
    def erd(actual,predicted,delimiter='\t',idcol=0,catcol=2,NumRP=0,logger=None):
        """Score one file against another.
        File format: ...<id>...<category>...
        """
        logger = logger or MuLabCat.logger
        mlc = MuLabCat(util.title_from_2paths(actual,predicted),
                       reassign=False,NumRP=NumRP,logger=logger)
        adict = util.read_multimap(actual,delimiter,col1=idcol,col2=catcol,
                                   logger=logger)
        pdict = util.read_multimap(predicted,delimiter,col1=idcol,col2=catcol,
                                   logger=logger)
        for obj,acat in adict.iteritems():
            mlc.add(acat,pdict.get(obj,frozenset()))
        for obj,pcat in pdict.iteritems():
            if obj not in adict:
                mlc.add(frozenset(),pcat)
        return mlc

    @staticmethod
    def score (actual,predicted,delimiter='\t',abeg=1,pbeg=1,NumRP=0,logger=None):
        """Score one file against another.
        File format: <query string>[<delimiter><category>]+
        The queries in both files must be identical.
        abeg and pbeg are the columns where actual and pedicted categories start.
        """
        logger = logger or MuLabCat.logger
        mlc = MuLabCat(util.title_from_2paths(actual,predicted),NumRP=NumRP,
                       logger=logger)
        util.reading(actual,logger)
        util.reading(predicted,logger)
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
    def random_stats (actual, repeat=1000, delimiter='\t', abeg=1, logger=None):
        "Score a file against shuffled self."
        logger = logger or MuLabCat.logger
        util.reading(actual,logger)
        nca = runstat.NumStat("cat/ex",integer=True)
        empty = frozenset([''])
        data = list()
        with open(actual) as af:
            for sa in csv.reader(af,delimiter=delimiter):
                sa = frozenset(sa[abeg:])-empty
                data.append(sa)
                nca.add(len(sa))
        logger.info("random_stats({r:,d}): Read {l:,d} lines: {o:s}".format(
            r=repeat,l=len(data),o=nca.__str__("{0:.2f}".format)))
        prre = runstat.NumStat("prec/recall")
        prof = runstat.NumStat("proficiency")
        for _ in xrange(repeat):
            indexes = range(len(data))
            random.shuffle(indexes)
            mlc = MuLabCat('shuffle+'+actual,reassign=False,logger=logger)
            for i in xrange(len(data)):
                mlc.add(data[i],data[indexes[i]])
            # logger.info("%s",mlc)
            f1,p,r = mlc.f1score()
            assert f1 == p and f1 == r
            prre.add(p)
            prof.add(mlc.proficiency_raw())
        logger.info("%d runs:\n %s\n %s",repeat,
                    prof.__str__("{0:.2%}".format),
                    prre.__str__("{0:.2%}".format))

    @staticmethod
    def category_set (taxonomy_size, max_cat = 4):
        return frozenset(random.sample(range(taxonomy_size),
                                       random.randint(1,max_cat)))

    @staticmethod
    def test (sample_size, taxonomy_size):
        print "\n *** MuLabCat(sample_size=%d,taxonomy_size=%d)" % (sample_size,taxonomy_size)
        MuLabCat.logger.info("test\n")
        mlc = MuLabCat("MLC.Perfect")
        for _ in xrange(sample_size):
            s = MuLabCat.category_set(taxonomy_size)
            mlc.add(s,s)
        MuLabCat.logger.info("%s\n",mlc)
        mlc = MuLabCat("MLC.Mislabled")
        relabling = range(taxonomy_size)
        random.shuffle(relabling)
        MuLabCat.logger.info("Relabling: %s\n",relabling)
        for _ in xrange(sample_size):
            s = MuLabCat.category_set(taxonomy_size)
            mlc.add(s,frozenset([relabling[i] for i in s]))
        MuLabCat.logger.info("%s\n",mlc)
        mlc = MuLabCat("MLC.Random")
        for _ in xrange(sample_size):
            mlc.add(MuLabCat.category_set(taxonomy_size),
                    MuLabCat.category_set(taxonomy_size))
        MuLabCat.logger.info("%s\n",mlc)
        mlc = MuLabCat("MLC.NotOne")
        for _ in xrange(sample_size):
            mlc.add(frozenset(range(taxonomy_size))-
                    frozenset([random.randint(1,taxonomy_size)-1]),
                    frozenset(range(taxonomy_size))-
                    frozenset([random.randint(1,taxonomy_size)-1]))
        MuLabCat.logger.info("%s\n",mlc)
        mlc = MuLabCat("MLC.TotalRecall")
        taxonomy = xrange(taxonomy_size)
        asize = taxonomy_size / 5
        for _ in xrange(sample_size):
            actual = frozenset(random.sample(taxonomy,asize))
            mlc.add(actual,actual | frozenset(random.sample(taxonomy,3*asize)))
        MuLabCat.logger.info("%s\n",mlc)
        mlc = MuLabCat("MLC.FullPrecision")
        taxonomy = xrange(taxonomy_size)
        asize = taxonomy_size / 2
        for _ in xrange(sample_size):
            actual = frozenset(random.sample(taxonomy,asize))
            mlc.add(actual,actual - frozenset(random.sample(actual,asize/2)))
        MuLabCat.logger.info("%s\n",mlc)
        # proficiency > f1 for mlc_0: many empty actual == empty predicted
        def small_sample (size, prob, repeat):
            # sample from range(size) at most repeat times
            # exiting early with probability 1-prob
            ret = set()
            for _ in xrange(repeat):
                if random.random() > prob:
                    return ret
                ret.add(random.randint(1,size))
            return ret
        def test_large_taxonomy (large_taxonomy_size,
                                 aprob, arepeat, pprob, prepreat):
            MuLabCat.logger.info("taxonomy: %d; actual: %g * %d; predicted: %g * %d",
                            large_taxonomy_size, aprob, arepeat, pprob, prepreat)
            mlc_0 = MuLabCat("MLC.Empty_As_Is")
            mlc_U = MuLabCat("MLC.Empty_Nocat")
            for _ in xrange(sample_size):
                actual = small_sample(large_taxonomy_size,aprob,arepeat)
                predicted = small_sample(large_taxonomy_size,pprob,prepreat)
                if actual:
                    if predicted:
                        predicted.remove(random.choice(list(predicted)))
                    predicted.add(random.choice(list(actual)))
                mlc_0.add(actual,predicted)
                # replace empty categorizations with special "Uncategorized"
                mlc_U.add(actual or {0},predicted or {0})
            MuLabCat.logger.info("test_large_taxonomy\n%s\n%s\n",mlc_0,mlc_U)
            for _ in xrange(sample_size):
                mlc_U.add({0},{0})
                mlc_0.add(set(),set())
            MuLabCat.logger.info("extra empties\n%s\n%s\n",mlc_0,mlc_U)
        test_large_taxonomy(taxonomy_size*10, 0.1, 2, 0.2, 3)
        test_large_taxonomy(taxonomy_size*10, 0.1, 3, 0.15, 4)
        for prob in [0.1, 0.2]:
            for repeat in [2, 3]:
                test_large_taxonomy(taxonomy_size*10, prob, repeat, prob, repeat)

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

    def sort (self):
        self.observations.sort(key=operator.itemgetter(1),reverse=True)

    def lq (self, thresholds, logger, title):
        if self.observations == []:
            return (float('NaN'), 0, 0, [])
        brc,brw = self.baseRate()
        if thresholds is None:
            thresholds = [brc] if brc == brw else [brc,brw]
        self.sort()
        targetLevel = 0
        cph = 0
        totalWeight = 0
        for isTarget, _score, weight in self.observations:
            cph += targetLevel * weight
            if isTarget:
                targetLevel += weight
                cph += weight * (weight + 1) / 2
            totalWeight += weight
        logger.debug("%s: cph=%g targetLevel=%g totalWeight=%g",
                     title,cph,targetLevel,totalWeight)
        if targetLevel == 0:
            return (float('NaN'), brc, brw, [])
        lq = ((2.0*cph - targetLevel) / (targetLevel * totalWeight) - 1) / (1 - brw)
        mxl = []
        total = self.len()
        for threshold in thresholds:
            mx = ConfusionMX("{h:s} at {d:.2%}".format(h=title,d=threshold),
                             logger=logger)
            mxl.append(mx)
            count = 0
            for isTarget, _score, weight in self.observations:
                count += 1
                mx.add(isTarget, total * threshold >= count, weight)
        return (lq, brc, brw, mxl)

    def __repr__ (self):
        self.sort()
        return repr(self.observations)

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
        borders = [int(i*n/float(nbin))+1 for i in xrange(1,nbin)]
        self.cuts = [-(observations[i-1][1]+observations[i][1])*0.5 for i in borders]
        self.bins = ([LqObExact.sum(observations,end=borders[0])]+
                     [LqObExact.sum(observations,beg=borders[i-1],end=borders[i])
                      for i in xrange(1,nbin-1)]+
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
        # cph = sum([(stw - tw/2)*ws for stw,(tw,ws) in
        #            zip(util.cumsum([tw for _tc, _co, tw, _ws in self.bins]),
        #                [(tw,ws) for _tc, _co, tw, ws in self.bins])])
        targetLevel = 0
        cph = 0
        for _tc, _co, tw, ws in self.bins:
            cph += ws * (targetLevel + (tw + 1) / 2)
            targetLevel += tw
        cph = float(cph) / (sws*stw)
        logger.debug("%s: cph=%g targetLevel=%g totalWeight=%g",
                     title,cph,stw,sws)
        lq = (2*cph - 1) / (1 - brw)
        mxl = []
        for threshold in thresholds:
            mx = ConfusionMX("{h:s} at {d:.2%}".format(
                h=title,d=float(cumco[threshold])/sco),logger=logger)
            mxl.append(mx)
            for i in xrange(len(self.bins)):
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

    def __init__ (self, title, truth = True, logger = None,
                  conversion_threshold = default_conversion_threshold):
        self.title = title
        self.truth = truth
        self.observations = LqObExact()
        self.conversion_threshold = conversion_threshold
        self.logger = logger or LiftQuality.logger

    def add (self, actual, score, weight = 1):
        if (self.conversion_threshold is not None and
            self.observations.len() == self.conversion_threshold):
            self.observations = LqObBinned(self.observations.observations,
                                           self.conversion_threshold/10)
        self.observations.add(actual == self.truth, score, weight)

    def len (self): return self.observations.len()
    def size (self): return self.observations.size()
    def baseRate (self): return self.observations.baseRate()
    def bestLift (self): return self.observations.bestLift()
    def lq (self, thresholds = None):
        return self.observations.lq(thresholds,self.logger,self.title)

    def __str__ (self):
        lq, _brc, brw, mxl = self.lq()
        summary = "LiftQuality({h:s}/{l:,d} {b:.2%}): Lq={q:.2%}".format(
            h=self.title,l=self.len(),b=brw,q=lq)
        if mxl == []:
            return summary
        return summary + "\n " + "\n ".join(str(mx) for mx in mxl)

    def __repr__ (self):
        return "LiftQuality[%s](%r)" % (
            self.title, self.observations
            if self.observations.len() < 20 else self.observations.__class__)

    @staticmethod
    def test (scale):
        print "\n *** LiftQuality(scale=%d)" % (scale)
        def show (l):
            print l
            print l.bestLift()
            print repr(l)
        def test_weights (size):
            lq = LiftQuality("base")
            lqr2 = LiftQuality("r2")
            lqd2 = LiftQuality("d2")
            lqd2r2 = LiftQuality("d2r2")
            lqd2r4 = LiftQuality("d2r4")
            lqm2 = LiftQuality("m2")
            lqrr = LiftQuality("RandomRepeat")
            lqrw = LiftQuality("RandomWeight")
            def observe (actual, score):
                lq.add(actual, score = score)
                lqd2.add(actual, score = score, weight = 0.5)
                lqm2.add(actual, score = score, weight = 2)
                for _ in xrange(2):
                    lqr2.add(actual, score = score)
                    lqd2r2.add(actual, score = score, weight = 0.5)
                for _ in xrange(4):
                    lqd2r4.add(actual, score = score, weight = 0.5)
                weight = random.randint(1,4)
                for _ in xrange(weight):
                    lqrr.add(actual, score = score)
                lqrw.add(actual, score = score, weight = weight)
            for _ in xrange(size):
                if random.randint(1,4) == 1:
                    observe(True,score = random.uniform(1,3))
                else:
                    observe(False,score = random.uniform(0,2))
            if lq.baseRate() == 0:
                observe(True,score = random.uniform(1,3))
            if lq.baseRate() == 1:
                observe(False,score = random.uniform(0,2))
            show(lq)
            show(lqr2)
            show(lqd2)
            show(lqd2r2)
            show(lqd2r4)
            show(lqm2)
            show(lqrr)
            show(lqrw)
            if size >= LiftQuality.default_conversion_threshold:
                tolerance = 0.001
            else: tolerance = None
            assert same_lq(lq,lqr2,tolerance)
            assert same_lq(lq,lqd2,tolerance)
            assert same_lq(lq,lqd2r2,tolerance)
            assert same_lq(lq,lqd2r4,tolerance)
            assert same_lq(lq,lqm2,tolerance)
            assert same_lq(lqrr,lqrw,tolerance)
        test_weights(10)
        # test_weights(scale)
        test_weights(10*scale)
        lq = LiftQuality("random")
        for _ in xrange(10*scale):
            lq.add(random.randint(1,4) == 1, random.random())
        show(lq)
        lq = LiftQuality("perfect")
        for _ in xrange(10*scale):
            if random.randint(1,4) == 1:
                lq.add(True, random.uniform(2,3))
            else:
                lq.add(False, random.uniform(0,1))
        show(lq)
        lq = LiftQuality("inverted")
        for _ in xrange(10*scale):
            if random.randint(1,4) == 1:
                lq.add(True, random.uniform(0,1))
            else:
                lq.add(False, random.uniform(2,3))
        show(lq)

def same_lq (lq1, lq2, tolerance = None):
    if tolerance is None:
        tolerance = sys.float_info.epsilon * (lq1.size() + lq2.size())
    lq_1,_brc1,brw1,_ = lq1.lq()
    lq_2,_brc2,brw2,_ = lq2.lq()
    # print "tol=%g; lq: %f %f %g; brc: %f %f %g; brw: %f %f %g" % (
    #     tolerance, lq_1, lq_2, lq_1-lq_2, brc1, brc2, brc1-brc2,
    #     brw1, brw2, brw1-brw2)
    # when testing weights, brc is always different!
    return (abs(lq_1 - lq_2) < tolerance and
            # abs(brc1 - brc2) < tolerance and
            abs(brw1 - brw2) < tolerance)

def check_merge_mismatch (o1, o2, attributes):
    if type(o1) != type(o2):
        raise ValueError ("merge: type mismatch",type(o1),type(o2),o1,o2)
    for a in attributes:
        if getattr(o1,a) != getattr(o2,a):
            raise ValueError ("merge: attribute mismatch",a,
                              getattr(o1,a),getattr(o2,a),o1,o2)

def merge_maybe_none (o1, o2):
    if o1 and o2:
        o1.merge(o2)
        return
    if o1 is None and o2 is None:
        return
    raise ValueError ("merge_maybe_none",o1,o2)

class LqStreaming (object):
    "Assume that the observations are pre-sorted by score, high to low."
    logger = util.get_logger("LqStreaming")

    def __init__ (self, title, truth = True, reverse = False):
        self.title = title
        self.truth = truth
        self.reverse = reverse
        self.targetWeight = 0
        self.targetCount = 0
        self.cph = 0
        self.totalWeight = 0
        self.totalCount = 0

    def isgood (self):
        return self.targetWeight > 0 and self.targetWeight < self.totalWeight

    def add (self, actual, weight = 1):
        self.totalCount += 1
        self.totalWeight += weight
        self.cph += self.targetWeight * weight
        if actual == self.truth:
            self.targetCount += 1
            self.targetWeight += weight
            self.cph += weight * (weight + 1) / 2

    def len (self):
        return self.totalCount

    def size (self):
        return self.totalWeight

    def baseRate (self):
        return (float(self.targetCount)/self.totalCount,
                float(self.targetWeight)/self.totalWeight)

    def lq (self, _thresholds = None):
        if self.totalWeight == []:
            return (float('NaN'), 0, 0, [])
        brc,brw = self.baseRate()
        if self.targetWeight == 0:
            return (float('NaN'), brc, brw, [])
        LqStreaming.logger.debug("%s: cph=%g targetWeight=%g totalWeight=%g",
                                 self.title,self.cph,self.targetWeight,
                                 self.totalWeight)
        lq = ((2.0*self.cph - self.targetWeight) / (
            self.targetWeight * self.totalWeight) - 1) / (1 - brw)
        return (-lq if self.reverse else lq, brc, brw, [])

    def __str__ (self):
        lq, brc, _brw, _ = self.lq()
        return "LqStreaming({h:s}:{t:,d}/{l:,d}={b:.2%}): Lq={q:.2%}".format(
            h=self.title,t=self.targetCount,l=self.totalCount,b=brc,q=lq)

    # this assumes reverse=False (i.e., decreasing score order)
    def merge (self, lqs):
        check_merge_mismatch(self,lqs,["title","truth","reverse"])
        self.cph += self.targetWeight * lqs.totalWeight + lqs.cph
        self.targetWeight += lqs.targetWeight
        self.targetCount += lqs.targetCount
        self.totalWeight += lqs.totalWeight
        self.totalCount += lqs.totalCount

    @staticmethod
    def test (sample_size):
        print "\n *** LqStreaming.test(%d)" % (sample_size)
        lqSW = LqStreaming("lqSW")
        lq0W = LiftQuality("lq0W")
        lqSR = LqStreaming("lqSR")
        lq0R = LiftQuality("lq0R")
        lqH1 = LqStreaming("Half1")
        lqH2 = LqStreaming("Half2")
        for i in xrange(sample_size):
            a = random.randint(1,4) == 1
            w = random.randint(3,8)
            lqSW.add(a,weight=w)
            lq0W.add(a,-i,weight=w)
            for _ in xrange(w):
                lqSR.add(a)
                lq0R.add(a,-i)
            if 2 * i < sample_size:
                lqH1.add(a,weight=w)
            else:
                lqH2.add(a,weight=w)
        print lqSW, lqSW.__dict__
        print lq0W
        print lqSR, lqSR.__dict__
        print lq0R
        print lqH1, lqH1.__dict__
        print lqH2, lqH2.__dict__
        # pacify check_merge_mismatch
        lqH2.title = lqH1.title
        lqH1.merge(lqH2)
        print lqH1, lqH1.__dict__
        assert same_lq(lqSW,lq0W)
        assert same_lq(lqSR,lq0R)
        assert same_lq(lqSW,lqSR)
        assert same_lq(lq0W,lq0R)
        assert same_lq(lq0W,lqH1)
        lqH1.title = lqSW.title
        assert lqSW.__dict__ == lqH1.__dict__

class PosNegStat (object):
    def __init__ (self, allt, post, negt):
        self.all = runstat.NumStat(allt)
        self.pos = runstat.NumStat(post)
        self.neg = runstat.NumStat(negt)

    def add_pos (self, num, weight):
        self.all.add(num,weight)
        self.pos.add(num,weight)

    def add_neg (self, num, weight):
        self.all.add(num,weight)
        self.neg.add(num,weight)

    def __str__ (self, toSt = str):
        return "  %s\n  %s\n  %s\n" % (
            self.all.__str__(toSt),self.pos.__str__(toSt),self.neg.__str__(toSt))

    def as_dict (self):
        return {
            "all": self.all.as_dict(),
            "pos": self.pos.as_dict(),
            "neg": self.neg.as_dict(),
        }

    def merge (self, pns):
        self.all.merge(pns.all)
        self.pos.merge(pns.pos)
        self.neg.merge(pns.neg)

class ScoringRule (object):
    "http://en.wikipedia.org/wiki/Scoring_rule"
    def __init__ (self, title, truth = True, stat = True):
        self.title = title
        self.truth = truth
        self.scores_stat = stat and PosNegStat("scores","sc_pos","sc_neg")
        self.logloss_stat = stat and PosNegStat("logloss","lol_pos","lol_neg")
        self.observations = 0
        self.positives = 0
        self.scores_sum_pos = 0.0
        self.scores_sum_neg = 0.0
        self.logloss_sum_pos = 0.0
        self.logloss_sum_neg = 0.0
        self.brier_sum = 0.0
        self.fp = 0             # actual=0, score=1
        self.fn = 0             # actual=1, score=0

    def isgood (self):
        return self.positives > 0 and self.positives < self.observations

    def merge (self, sr):
        check_merge_mismatch(self,sr,["title","truth"])
        merge_maybe_none(self.scores_stat,sr.scores_stat)
        merge_maybe_none(self.logloss_stat,sr.logloss_stat)
        self.observations += sr.observations
        self.positives += sr.positives
        self.scores_sum_pos += sr.scores_sum_pos
        self.scores_sum_neg += sr.scores_sum_neg
        self.logloss_sum_pos += sr.logloss_sum_pos
        self.logloss_sum_neg += sr.logloss_sum_neg
        self.brier_sum += sr.brier_sum
        self.fp += sr.fp
        self.fn += sr.fn

    def stats_dict (self):
        if self.scores_stat:
            return {
                "scores": self.scores_stat.as_dict(),
                "logloss": self.logloss_stat.as_dict(),
            }
        else:
            return None

    def add (self, actual, score, weight = 1):
        if 0 > score or score > 1 or weight <= 0:
            raise ValueError("ScoringRule.add",self,actual,score,weight)
        self.observations += weight
        if actual == self.truth:
            self.positives += weight
            if self.scores_stat:
                self.scores_stat.add_pos(score,weight)
            self.scores_sum_pos += weight * score
            brier = (score - 1) * (score - 1)
            lolo = - math.log(max(sys.float_info.epsilon,score))
            if self.logloss_stat:
                self.logloss_stat.add_pos(lolo,weight)
            self.logloss_sum_pos += weight * lolo
            if score == 0:
                self.fn += weight
        else:
            if self.scores_stat:
                self.scores_stat.add_neg(score,weight)
            self.scores_sum_neg += weight * score
            brier = score * score
            lolo = - math.log(max(sys.float_info.epsilon,1-score))
            if self.logloss_stat:
                self.logloss_stat.add_neg(lolo,weight)
            self.logloss_sum_neg += weight * lolo
            if score == 1:
                self.fp += weight
        self.brier_sum += weight * brier

    def base_rate (self):
        if self.observations == 0:
            return float("nan")
        return float(self.positives) / self.observations

    def brier (self):
        if self.observations == 0:
            return float("nan")
        return self.brier_sum / self.observations

    def brier_normalized (self, p = None):
        if p is None:
            p = self.base_rate()
        if math.isnan(p) or p == 1 or p == 0:
            return ""
        return " ({b:.2%})".format(b=1 - self.brier() / (p * (1 - p)))

    def log_loss (self):
        if self.observations == 0:
            return float("nan")
        return (self.logloss_sum_pos + self.logloss_sum_neg) / self.observations

    def mean_score (self):
        if self.observations == 0:
            return float("nan")
        return (self.scores_sum_neg + self.scores_sum_pos) / self.observations

    def log_loss_normalized (self, p = None):
        if p is None:
            p = self.base_rate()
        if math.isnan(p) or p == 1 or p == 0:
            return float("nan")
        return 1 + self.log_loss() / (p*math.log(p)+(1-p)*math.log(1-p))

    def log_loss_normalized_s (self, p = None):
        l = self.log_loss_normalized(p=p)
        if math.isnan(l):
            return ""
        return " ({l:.2%})".format(l=l)

    def pos_neg_sum (self, pos, neg, toSt):
        if self.observations == 0:
            return "[no observations]"
        tot = pos + neg
        if tot == 0:
            return "[pos+neg=0]"
        return "{t:s} = pos {p:s}({pp:.4%}) + neg {n:s}({np:.4%})".format(
            t=toSt(tot/self.observations),p=toSt(pos/self.observations),
            pp=pos/tot,n=toSt(neg/self.observations),np=neg/tot)

    def overpredict (self):
        total_scores = self.scores_sum_neg + self.scores_sum_pos
        if self.positives == 0:
            if total_scores == 0:
                return 0
            return float("+inf")
        return total_scores / self.positives - 1

    def __str__ (self):
        p = self.base_rate()
        return ("ScoringRule({T:s}:{P:,.0f}/{N:,.0f}={br:.2%}[err={e:.2%}]"
                "{fp:s}{fn:s}) Log={L:6f}{Ln} Brier={B:6f}{Bn}\n{ScSt:s}{LLSt:s}"
                "  Score {scores:s}\n  LogLoss {logloss:s}".format(
                    T=self.title,P=self.positives,N=self.observations,br=p,
                    e=self.overpredict(),
                    fp="" if self.fp==0 else "/fp={fp:,.0f}".format(fp=self.fp),
                    fn="" if self.fn==0 else "/fn={fn:,.0f}".format(fn=self.fn),
                    L=self.log_loss(),Ln=self.log_loss_normalized_s(p),
                    B=self.brier(),Bn=self.brier_normalized(p),
                    ScSt=(self.scores_stat.__str__("{0:.2%}".format)
                          if self.scores_stat else ""),
                    LLSt=str(self.logloss_stat) if self.logloss_stat else "",
                    scores=self.pos_neg_sum(
                        self.scores_sum_pos,self.scores_sum_neg,"{:.3%}".format),
                    logloss=self.pos_neg_sum(
                        self.logloss_sum_pos,self.logloss_sum_neg,"{:f}".format)))

    @staticmethod
    def test (mean_ratio,base_rate,length,alpha_pos=2,alpha_neg=2):
        print "\n *** ScoringRule(mean_ratio=%d,base_rate=%f,length=%d,alpha: %d/%d)" % (mean_ratio,base_rate,length,alpha_pos,alpha_neg)
        assert mean_ratio > 1
        mean_neg = base_rate / (mean_ratio * base_rate + 1 - base_rate)
        beta_neg = alpha_neg * (1-mean_neg) / mean_neg
        mean_pos = min(1-base_rate,mean_ratio * mean_neg)
        beta_pos = alpha_pos * (1-mean_pos) / mean_pos
        sr = ScoringRule("mr={mr:g}".format(mr=mean_ratio))
        lq = LiftQuality(sr.title)
        for _ in xrange(int(round(base_rate*length))):
            score = random.betavariate(alpha_pos,beta_pos)
            sr.add(True,score)
            lq.add(True,score)
        for _ in xrange(int(round(length*(1-base_rate)))):
            score = random.betavariate(alpha_neg,beta_neg)
            sr.add(False,score)
            lq.add(False,score)
        print sr
        print lq

def test (what):
    known = frozenset(["mlc","cmx","lq","sr"])
    if len(what) == 0 or "mlc" in what:
        MuLabCat.test(10000, 10)
    if len(what) == 0 or "cmx" in what:
        ConfusionMX.test(10000,2)
        ConfusionMX.test(10000,5)
    if len(what) == 0 or "lq" in what:
        LiftQuality.test(LiftQuality.default_conversion_threshold)
        LqStreaming.test(100)
    if len(what) == 0 or "sr" in what:
        ScoringRule.test(2,0.1,100000)
        ScoringRule.test(3,0.01,100000)
        ScoringRule.test(4,0.001,100000)
    if not what.issubset(known):
        raise ValueError("unknown class",what-known,known)

class LqSr (object):
    "Sorted stream, scored using both LqStreaming & ScoringRule - single."
    def __init__ (self, title, truth=True, reverse=False, calibrated=True, stat=True):
        self.lq = LqStreaming(title,truth=truth,reverse=reverse)
        self.sr = ScoringRule(title,truth=truth,stat=stat)
        self.calibrated = calibrated

    def isempty (self):
        return not (self.lq.isgood() and self.sr.isgood())

    def add (self, target, score, weight = 1):
        self.lq.add(target,weight)
        if not self.calibrated:
            score = 1/(1+math.exp(-score))
        self.sr.add(target,score,weight)

    def merge (self, lqsr):
        check_merge_mismatch(self,lqsr,["calibrated"])
        self.lq.merge(lqsr.lq)
        self.sr.merge(lqsr.sr)

    def get_res (self, group, bad_order_count):
        return {
            'group':group,
            'lq':self.lq.lq()[0] if bad_order_count == 0 and self.lq.isgood() else None,
            'observations':self.lq.totalWeight,
            'positives':self.lq.targetWeight,
            'log_loss':self.sr.log_loss(),
            'brier':self.sr.brier(),
            'mean_prediction':self.sr.mean_score(),
            'fp':self.sr.fp,
            'fn':self.sr.fn,
            # 'stats':self.sr.stats_dict(), <-- each group will get a stat!
        }

class StreamingLqSr (object):
    "Sorted stream, scored using both LqStreaming & ScoringRule - grouped."
    def __init__ (self, title, baseout, runid=None, groups=None,
                  truth=True, reverse=False, calibrated=True):
        self.lqsr = LqSr(title,truth=truth,reverse=reverse,calibrated=calibrated)
        self.last_score = float("-Inf") if reverse else float("Inf")
        self.reverse = reverse
        self.bad_order_count = 0
        self.record = 0
        self.lqsrd = dict()
        self.baseout = baseout
        self.runid = runid
        self.groups = groups
        self.start = time.time()

    def merge (self,slqsr):
        check_merge_mismatch(self,slqsr,["baseout","runid","reverse","groups"])
        if (self.last_score > slqsr.last_score if self.reverse else
            self.last_score < slqsr.last_score):
            raise ValueError("StreamingLqSr.merge: last_score",self.reverse,
                             self.last_score,slqsr.last_score,self,slqsr)
        self.lqsr.merge(slqsr.lqsr)
        self.bad_order_count += slqsr.bad_order_count
        self.record += slqsr.record
        lqsrd2 = slqsr.lqsrd.copy()
        for gr,lqsr in self.lqsrd.iteritems():
            try:
                lqsr.merge(lqsrd2.pop(gr))
            except KeyError:
                pass              # ignore missing groups!
        self.lqsrd.update(lqsrd2) # add possible new groups

    def save (self, logger = util.get_logger("StreamingLqSr")):
        if self.lqsr.isempty():
            logger.error("NOT saving empty LqSr to %s",self.baseout)
            return
        self.lqsrd["total"] = self.lqsr
        resl = sorted([lqsr.get_res(group,self.bad_order_count)
                       for group,lqsr in self.lqsrd.iteritems()],
                      key = operator.itemgetter('observations'),
                      reverse=True)
        logger.info("Writing total%s to %s.{json,csv}",
                    " and {g:,d} groups".format(g=len(self.lqsrd))
                    if self.lqsrd else "", self.baseout)
        with open(self.baseout+".csv","w") as o:
            w = csv.DictWriter(o, fieldnames=[
                'group',
                'observations', 'positives',
                'lq',
                'log_loss', 'brier', 'mean_prediction',
                'fp', 'fn',
            ]) # add [[extrasaction='ignore']] if get_res() includes stats
            w.writeheader()
            for res in resl:
                w.writerow(res)
        util.wrote(self.baseout+".csv",logger=logger)
        with open(self.baseout+".json","w") as j:
            json.dump({
                'TimeStamp':datetime.datetime.now().strftime("%F %T"),
                'Elapsed':time.time()-self.start,
                'RunID':self.runid, 'Name':self.lqsr.lq.title,
                'Groups':self.groups,
                'Stats': self.lqsr.sr.stats_dict(),
                'groups':resl,
            },j,sort_keys=True,indent=2,separators=(',',':'))
        util.wrote(self.baseout+".json",logger=logger)

    def add (self, target, score, weight = 1, group = None):
        self.record += 1
        if (score < self.last_score if self.reverse else score > self.last_score):
            self.bad_order_count += 1
            if self.bad_order_count <= 10:
                sys.stderr.write("record %s: BAD SCORE ORDER: %g %s %g" % (
                    self.record,score,'<' if self.reverse else '>',self.last_score))
        self.last_score = score
        self.lqsr.add(target,score,weight)
        if group:
            try:
                lqsr = self.lqsrd[group]
            except KeyError:
                lqsr = self.lqsrd[group] = LqSr(
                    group,reverse=self.reverse,calibrated=self.lqsr.calibrated,
                    truth=self.lqsr.lq.truth,stat=None) # no per-group pos/neg stats
            lqsr.add(target,score,weight)

    def report (self, logger = util.get_logger("StreamingLqSr")):
        if self.bad_order_count > 0:
            logger.error("The LiftQuality below is invalid: %d mis-ordered scores",self.bad_order_count)
        logger.info("Done {t:s} ({r:,d} records) [{e:s}]\n{lq:s}\n{sr:s}".format(
            t=self.lqsr.lq.title,r=self.record,e=progress.elapsed(self.start),
            lq=self.lqsr.lq,sr=self.lqsr.sr))
        if self.lqsrd:
            lq_ns = runstat.NumStat("LiftQuality")
            ll_ns = runstat.NumStat("LogLossNorm")
            lq_pos = 0
            ll_pos = 0
            for lqsr in self.lqsrd.itervalues():
                ll = lqsr.sr.log_loss_normalized()
                if not math.isnan(ll):
                    ll_ns.add(ll)
                    if ll > 0:
                        ll_pos += 1
                    lq = lqsr.lq.lq()[0]
                    lq_ns.add(lq)
                    if lq > 0:
                        lq_pos += 1
            if lq_ns.count and ll_ns.count:
                logger.info("Positive LQ: %d/%d (%.2f%%); LL: %d/%d (%.2f%%)\n %s\n %s",
                            lq_pos,lq_ns.count,100.0*lq_pos/lq_ns.count,
                            ll_pos,ll_ns.count,100.0*ll_pos/ll_ns.count,
                            lq_ns.__str__("{0:.2%}".format),
                            ll_ns.__str__("{0:.2%}".format))
            else:
                logger.error("No groups with positive observations")
        if self.baseout:
            self.save(logger=logger)

    @staticmethod
    def row2tuple (row, ropa):
        return (ropa.get(row,ropa.tpos) == ropa.truth,
                [float(ropa.get(row,spos)) for spos in ropa.sposl],
                float(ropa.get(row,ropa.wpos)) if ropa.wpos else 1,
                ropa.get_groups(row))

class RowParser (object):
    def __init__ (self, file_format, evaluator, reader, parser, getter,
                  tpos, truth, sposl, wpos, gpos):
        self.file_format = file_format
        self.evaluator = evaluator
        self.reader = reader
        self.parser = parser
        self.tpos = tpos
        self.truth = truth
        self.sposl = sposl
        self.wpos = wpos
        self.gpos = gpos
        self.get = getter

    @staticmethod
    def getA (row, index):
        return row[index]

    @staticmethod
    def getD (row, index):
        return row.get(index)

    def get_groups (self, rec):
        if not self.gpos:
            return None
        return "-".join([str(self.get(rec,g)) for g in self.gpos])

    def iterate (self, istream):
        for row in self.reader(istream) if self.reader else istream:
            try:
                yield self.evaluator.row2tuple(
                    (self.parser(row) if self.parser else row), self)
            except Exception as ex:
                yield (row, ex)

def score_stream (istream,iname,file_format,tpos,truth,sposcal,reverse,
                  wpos=None,gpos=None,out=None,runid=None,max_lines=None,
                  logger=util.get_logger("PredEvalStream")):
    if not sposcal:
        raise ValueError("missing score column(s)")
    if isinstance(sposcal,str):
        logger.warn("sposcal should be a list, not a string %s",sposcal)
        sposcal = [sposcal]
    if isinstance(gpos,str):
        logger.warn("gpos should be a list, not a string %s",gpos)
        gpos = [gpos]
    # evaluator: LqSr or ConfusionMX
    if truth:                   # present ==> LqSr
        evaluator_t = StreamingLqSr
        if isinstance(truth,str):
            try:
                import ast
                truth = ast.literal_eval(truth)
            except Exception as ex:
                logger.warn("Truth value [%s] is literal (%s)",truth,ex)
        logger.info("Evaluating using LiftQuality & ScoringRule, truth=[%s]",truth)
    else:
        evaluator_t = StreamingCMX
        logger.info("Evaluating using ConfusionMX")
    # file_format: VW, CSV or JSON
    if file_format == "vw":
        reader = None
        parser = parse_vw_line
        getter = RowParser.getD
    elif file_format == "json":
        reader = None
        parser = json.loads
        getter = RowParser.getD
    elif file_format == "csv":
        sep = ","
    elif file_format == "tsv":
        sep = "\t"
        file_format = "csv"
    elif file_format.startswith("sep="):
        sep = file_format[4:]
        file_format = "csv"
    else:
        raise ValueError("Bad -format",file_format,
                         ["json","vw","tsv","csv","sep=?"])
    if file_format == "csv":
        file_format, tpos, wpos, gpos = (
            "csv",int(tpos), wpos and int(wpos),[int(g) for g in gpos])
        reader = lambda (fd): csv.reader(fd,delimiter=sep)
        getter = RowParser.getA
        parser = None
    logger.info("Parsing %s as %s",iname,file_format)
    if gpos:
        logger.info("Grouping by %s",gpos)
    # score positions
    evaluators = list()
    sposl = list()
    for sc in sposcal:
        if sc.startswith("raw:"):
            if evaluator_t == StreamingCMX:
                raise ValueError("ConfusionMX does not accept [raw:]",sposcal)
            spos = sc[4:]
            calibrated = False
        else:
            spos = sc
            calibrated = True
        sposl.append(int(spos) if file_format == "csv" else spos)
        if len(sposcal) == 1:
            baseout = out
            ename = iname
        else:
            baseout = out and "%s-%s" % (out,spos)
            ename = "%s-%s" % (iname,spos)
        evaluators.append(
            StreamingLqSr(ename,reverse=reverse,calibrated=calibrated,truth=truth,
                          baseout=baseout,runid=runid,groups=gpos)
            if evaluator_t == StreamingLqSr else
            StreamingCMX(ename,logger=logger,baseout=baseout,runid=runid,groups=gpos))
    ropa = RowParser(file_format=file_format,evaluator=evaluator_t,reader=reader,getter=getter,
                     parser=parser,tpos=tpos,truth=truth,sposl=sposl,wpos=wpos,gpos=gpos)
    lines_bad = 0
    lines_total = 0
    for rec in ropa.iterate(istream):
        lines_total += 1
        if max_lines and lines_total > max_lines:
            break
        if len(rec) != 4:
            lines_bad += 1
            if lines_bad <= 10:
                logger.error("line %d: BAD RECORD: [%s]",lines_total,rec)
            continue
        actual, scores, weight, groups = rec
        for evaluator,score in zip(evaluators,scores):
            evaluator.add(actual,score,weight,groups)
    if lines_bad:
        logger.error("{b:,d} bad lines ({p:.2%} out of {t:,d})".format(
            b=lines_bad,p=float(lines_bad)/lines_total,t=lines_total))
    for evaluator in evaluators:
        evaluator.report(logger=logger)

def main ():
    ap = argparse.ArgumentParser(description='Evaluate Predictions')
    sp = ap.add_subparsers(help='classes to use',dest='cmd')
    cmx = sp.add_parser('ConfusionMX',help='Confusion Matrix')
    mlc = sp.add_parser('MuLabCat',help='Multi Label Categorization')
    lsc = sp.add_parser('Streaming',help='LqStreaming + ScoringRule or ConfusionMX')
    tst = sp.add_parser('test',help='internal tests')
    cmx.add_argument('-predictions',help='path to VW predictions file')
    cmx.add_argument('-base',help='base character for OCR (a or 0 or ...)')
    mlc.add_argument('-format',help='input file format',choices=['erd', 'one'])
    mlc.add_argument('-actual',help='path to the actual (annotations)')
    mlc.add_argument('-predicted',help='path to the predicted')
    mlc.add_argument('-abeg',type=int,default=1,
                     help='the column in actual where the categories start')
    mlc.add_argument('-pbeg',type=int,default=1,
                     help='the column in predicted where the categories start')
    mlc.add_argument('-out',help='output file base (json & csv)')
    mlc.add_argument('-NumRP',type=int,default=0,help='the Number of RandomPairs for testing numeric stability')
    cmx.add_argument('-NumRP',type=int,default=0,help='the Number of RandomPairs for testing numeric stability')
    lsc.add_argument('-format',required=True,
                     help='the file format (json, vw, sep=,)')
    lsc.add_argument('-spos',help='the position of the Score field, may be repeated; if uncalibrated, prepend "raw:" to pass the score to logit(x)=1/(1+exp(-x))',action="append")
    lsc.add_argument('-tpos',help='the position of the Target field')
    lsc.add_argument('-wpos',default=None,
                     help='the position of the Weight field')
    lsc.add_argument('-gpos',action='append',default=[],
                     help='the position of the Group field')
    lsc.add_argument('-truth',help='the Truth value (LqSr when present, ConfusionMX otherwise)')
    lsc.add_argument('-reverse',default=False,action='store_true',
                     help='the stream is in reversed order')
    lsc.add_argument('-input',type=argparse.FileType('r'),default=sys.stdin,
                     help='the stream to read from')
    lsc.add_argument('-input_name',help='the input stream name')
    lsc.add_argument('-id',help='run id, e.g., git hash')
    lsc.add_argument('-max_lines',type=int,help='upper limit on line count')
    lsc.add_argument('-out',help='output file base (json & csv)')
    tst.add_argument('what',nargs='*',help='which classes to test')
    args = ap.parse_args()
    if args.cmd == 'ConfusionMX':
        if args.predictions is not None:
            print ConfusionMX.score_vw_oaa(args.predictions, base=args.base,
                                           NumRP=args.NumRP)
        else: # Parse output from $(make -C vw/demo/mnist raw png)
            ConfusionMX.vw_demos(os.path.expandvars("$HOME/src/sds-vw/demo"),
                                 base=args.base,NumRP=args.NumRP)
    elif args.cmd == 'MuLabCat':
        if args.actual is not None and args.predicted is not None:
            mlc = (MuLabCat.erd(args.actual,args.predicted,NumRP=args.NumRP)
                   if args.format == 'erd' else
                   MuLabCat.score(args.actual,args.predicted,
                                  abeg=args.abeg,pbeg=args.pbeg,
                                  NumRP=args.NumRP))
            print mlc.__str__()
            out = args.out
            if out:
                mlc.save(out)
        else:                   # pre-supplied data
            data_dir = "../data"
            # Queries annotated by 3 different people for KDD Cup 2005
            l1 = os.path.join(data_dir,"labeler1.txt")
            l2 = os.path.join(data_dir,"labeler2.txt")
            l3 = os.path.join(data_dir,"labeler3.txt")
            MuLabCat.random_stats(l1)
            MuLabCat.random_stats(l2)
            MuLabCat.random_stats(l3)
            print MuLabCat.score(l1,l2,NumRP=args.NumRP)
            print MuLabCat.score(l2,l3,NumRP=args.NumRP)
            print MuLabCat.score(l3,l1,NumRP=args.NumRP)
            # Hand-annotated 10k queries (Magnetic)
            ann = os.path.join(data_dir,"magnetic_annotated.txt")
            res = os.path.join(data_dir,"magnetic_results.txt")
            res5 = os.path.join(data_dir,"magnetic_results_upto5.txt")
            MuLabCat.random_stats(ann,abeg=2)
            MuLabCat.random_stats(res)
            MuLabCat.random_stats(res5)
            print MuLabCat.score(ann,res,abeg=2,NumRP=args.NumRP)
            print MuLabCat.score(ann,res5,abeg=2,NumRP=args.NumRP)
            # Slovak queries in ERD format
            trecA = os.path.join(data_dir,"Trec_beta_annotations.txt")
            trecR = os.path.join(data_dir,"Trec_beta_results.txt")
            print MuLabCat.erd(trecA,trecR,NumRP=args.NumRP)
            qskA = os.path.join(data_dir,"sk_annotation.txt")
            qskR = os.path.join(data_dir,"sk_results.txt")
            qskRA = os.path.join(data_dir,"sk_results_ascii.txt")
            print MuLabCat.erd(qskA,qskR,NumRP=args.NumRP)
            print MuLabCat.erd(qskA,qskRA,NumRP=args.NumRP)
    elif args.cmd == 'Streaming':
        if score_stream(istream=args.input,file_format=args.format,tpos=args.tpos,
                        iname=args.input_name or args.input.name,
                        truth=args.truth,sposcal=args.spos,wpos=args.wpos,
                        gpos=args.gpos,out=args.out,reverse=args.reverse,
                        runid=args.id,max_lines=args.max_lines) > 0:
            sys.exit(1)
    elif args.cmd == 'test':
        test(set(args.what))
    else:
        assert(False)           # NOT REACHED

if __name__ == '__main__':
    main()
