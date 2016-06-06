# misc small utilities

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

import sys
import collections
import csv
import os
import math
import re
import logging
import random

def weighted_sample(x, weights, n=1):
    "Returns a weighted sample of length n with replacement. Weight do not need to sum to 1."
    length = len(x)
    assert length == len(weights) > 0
    assert n >= 1
    totalweight = sum(weights)
    cumulative_sum = 0
    sample = []
    r = [random.random() * totalweight for i in xrange(n)]
    for i in xrange(length):
        cumulative_sum += weights[i]
        for j in xrange(n):
            if r[j] < cumulative_sum:
                sample.append(x[i])
                r[j] = totalweight + 1 #make sure that it won't be triggered again
                if len(sample) >= n:
                    break
    return sample

def sample_file (inf, outf, rates, pos, logger=None, separator = "\t"):
    """Sample lines in INF according to RATES at position POS and write OUTF."""
    reading(inf,logger)
    written = collections.defaultdict(int)
    wasread = collections.defaultdict(int)
    with open(inf) as i:
        with open(outf,"w") as o:
            for l in i:
                v = l.strip().split(separator)[pos]
                wasread[v] += 1
                try:
                    r = rates[v]
                except KeyError:
                    r = rates[v] = 1
                    warn("Unexpected value [%s] in [%s], set rate=1" % (v,l.strip()), logger)
                if r == 1 or random.random() <= r:
                    o.write(l)
                    written[v] += 1
    wrote(outf,logger)
    info("Read {i:,d} lines: {c:s}".format(
        i=sum(wasread.itervalues()),c=counter2string(wasread)),logger)
    info("Wrote {i:,d} lines: {c:s}".format(
        i=sum(written.itervalues()),c=counter2string(written)),logger)

def get_logger (name, level = logging.INFO):
    console = logging.StreamHandler() # stderr
    console.setFormatter(logging.Formatter(
        fmt='%(asctime)s %(levelname)s %(name)s/%(module)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'))
    # add the handler to the root logger
    logger = logging.getLogger(name)
    logger.addHandler(console)
    logger.setLevel(level)
    return logger

def debug (s,logger = None):
    if logger is None:
        print "DEBUG " + s
    elif isinstance(logger,logging.Logger):
        logger.debug(s)
    else:
        pass

def info (s,logger = None):
    if logger is None:
        print s
    elif isinstance(logger,logging.Logger):
        logger.info(s)
    else:
        pass

def warn (s,logger = None):
    if logger is None:
        print "WARNING " + s
    elif isinstance(logger,logging.Logger):
        logger.warn(s)
    else:
        pass

# http://stackoverflow.com/questions/497885/python-element-wise-tuple-operations-like-sum
def tuple_sum (a, b):
    return tuple(map(sum, zip(a, b)))

# http://stackoverflow.com/questions/480214/how-do-you-remove-duplicates-from-a-list-in-python-whilst-preserving-order
def dedup (seq):
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]

# http://stackoverflow.com/questions/15889131/how-to-find-the-cumulative-sum-of-numbers-in-a-list
def accumu (seq):
    total = 0
    for x in seq:
        total += x
        yield total

def cumsum (seq): return list(accumu(seq))

# similar to the system function, but
# - does not support negative step
# - does support dates and such, as long as they support "+" and "<"
# - stop is included in the range
def myrange (start, stop, step):
    while start <= stop:
        yield start
        start += step

import ast
import pprint
class HumanReadable (object):
    # http://stackoverflow.com/questions/28055565/how-to-serialize-a-python-dict-to-text-in-a-human-readable-way
    @staticmethod
    def save (x, fname):
        with open(fname, 'w') as f:
            pprint.PrettyPrinter(stream=f).pprint(x)

    @staticmethod
    def load (fname):
        with open(fname, 'r') as f:
            return ast.literal_eval(f.read())

# execfile('util.py'); test()
def test ():
    counter = collections.Counter(['a','b','a','c','a','b'])
    PrintCounter().out(counter,'test1')
    PrintCounter(max_row=2,min_omit=0,min_row=0).out(counter,'test2')
    PrintCounter.csv(counter,'test3',sys.stdout)
    PrintCounter.csv(counter,'test4',"foo-")
    os.remove("foo-test4.csv")
    counter[u'a\u0437'] = 3
    counter[7] = 5
    print counter
    PrintCounter.csv(counter,'test5',"foo-")
    os.remove("foo-test5.csv")
    print asBigNumberBin(123456789)
    print asBigNumberDec(123456789)
    print "bin_entropy"
    for x in range(10):
        print bin_entropy(10,x)
    print "bin_mutual_info"
    print bin_mutual_info(200,100,100,50)
    for x in range(10):
        print bin_mutual_info(200,20,20+0.8*x,(200-x)*0.1)
    x1 = dict([(a,2*a) for a in range(10)])
    x1[(1,2,3)] = 6
    x1 = [x1] + [(x1,x1)]
    HumanReadable.save(x1, "tmp")
    x2 = HumanReadable.load("tmp")
    os.remove("tmp")
    if x1 != x2:
        raise Exception("HumanReadable",x1,x2)
    print x1

def default_None (x, d): return d if x is None else x

def empty2none (v): return (None if v == '' else v)

# http://stackoverflow.com/questions/29127801/cross-version-portability-in-python
if hasattr(1, 'bit_length'):
    def bitlen (x): return x.bit_length()
else:
    def bitlen (x): return len(bin(x))-2

# http://en.wikipedia.org/wiki/Binary_prefix
binaryPrefixes = ['K','M','G','T','P','E','Z','Y']

asBigNumberBinCuts = [(10*(y+1),binaryPrefixes[y]) for y in range(len(binaryPrefixes))][::-1]
def asBigNumberBin (v):         # valid toString argument
    l = bitlen(v)
    for b,p in asBigNumberBinCuts:
        if l >= b:
            return "%.1f%si" % ((v >> (b-10)) / 1024.0, p)
    return str(v)

asBigNumberDecCuts = [(10.0**(3*(y+1)),binaryPrefixes[y]) for y in range(len(binaryPrefixes))][::-1]
def asBigNumberDec (v):         # valid toString argument
    for c,p in asBigNumberDecCuts:
        if v >= c:
            return "%.1f%s" % (v / c, p)
    return str(v)

def nicenum (s):                # nice number presentation
    try:
        return "{n:,d}".format(n=int(s))
    except ValueError:
        return s

# not needed in python3
def ensure_dir (path, logger = None):
    if path == "":              # current directory is presumed to exist
        return
    try:
        os.makedirs(path)
        info("Created [%s]" % (path),logger)
    except OSError:
        if os.path.isdir(path):
            debug("Path [%s] already exists" % (path),logger)
        else:
            raise

class DirLock (object):
    def __init__ (self, path):
        ensure_dir(path)
        self.dir = path
        self.lock = os.path.join(path,"locked")

    def __enter__ (self):
        if os.path.exists(self.lock):
            with open(self.lock) as l:
                raise ValueError("directory is in use",self.dir,l.read())
        with open(self.lock,"w") as l:
            l.write("pid=%s logname=%s" % (os.getpid(),os.getenv("LOGNAME")))
        return self.dir

    def __exit__ (self, _exc_type, _exc_value, _traceback):
        os.unlink(self.lock)

# turn exceptions into None
def catching_exceptions (logger,function,arguments):
    try:
        return function(*arguments)
    except Exception as e:
        logger.error("%s: %s",function.__name__,e)
        return None

# http://nullege.com/codes/search/pyutil.strutil.commonsuffix
def commonsuffix(l):
    cp = []
    for i in range(min([len(element) for element in l])):
        c = l[0][-i-1]
        for s in l[1:]:
            if s[-i-1] != c:
                cp.reverse()
                return ''.join(cp)
        cp.append(c)
    cp.reverse()
    return ''.join(cp)

def title_from_2paths (first, second):
    cp = os.path.commonprefix([first,second])
    cs = commonsuffix([first,second])
    return "%s(%s|%s)%s" % (
        cp,first[len(cp):len(first)-len(cs)],
        second[len(cp):len(second)-len(cs)],cs)

def canonicalize_domain (domain):
    if domain is None or domain == '':
        return None
    if re.match(r'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+(:[0-9]+)?$',domain):
        return 'dotted.quad'
    domain = re.sub(r'(:[0-9]+|[.:]+)$','',domain.lower()) # strip port & downcase
    tld = re.sub(r'^.*\.([a-z]*)$',r'\1',domain)
    if len(tld) > 2: mindot = 1 # .com .info, .travel, .kitchen &c
    elif len(tld) == 2: mindot = 2 # gov.us com.cn &c
    else:
        # logger.info("weird domain [[%s]]",domain)
        return domain
    while domain.count('.') > mindot:
        domain1 = re.sub(r'^(pub|web|www*)?-?[0-9]*\.','',domain)
        if domain1 == domain:
            return domain
        else:
            domain = domain1
    return domain

def url2host (url):
    if url is None or url == '':
        return None
    if re.match(r'https?://',url):
        return re.sub(r'^https?://([^/]*)(/.*)?',r'\1',url)
    return 'bad.url'

def url2domain (url):
    return canonicalize_domain(url2host(url))

def sigmoid (v):
    return 1/(1+math.exp(-v))
def antisigmoid(v):
    return -math.log(1/v - 1)

def bin_entropy (total, first):
    "Return the total entropy in nats."
    if total < 0 or first < 0 or first > total:
        raise ValueError("util.bin_entropy",total,first)
    if total == 0 or first == 0 or first == total:
        return 0
    second = total - first
    return math.log(total) - (
        first * math.log(first) + second * math.log(second)) / total

def bin_mutual_info (total, actual, predicted, tp):
    "Return the mutual information in nats."
    fn = actual - tp
    fp = predicted - tp
    tn = total - actual - predicted + tp
    if (total < 0 or actual > total or actual < 0 or predicted > total
        or predicted < 0 or tp < 0 or fn < 0 or fp < 0 or tn < 0):
        raise ValueError("util.bin_mutual_info",total, actual, predicted, tp)
    if total == 0 or actual == 0 or actual == total or predicted == 0 or predicted == total:
        return 0
    mi = 0
    total = float(total)
    if tp > 0:
        mi += tp * math.log(total * tp / (actual * predicted))
    if fn > 0:
        mi += fn * math.log(total * fn / (actual * (total-predicted)))
    if fp > 0:
        mi += fp * math.log(total * fp / ((total-actual) * predicted))
    if tn > 0:
        mi += tn * math.log(total * tn / ((total-actual) * (total-predicted)))
    return mi / total

def dict_entropy (counts, missing = None, scaledto1 = False):
    "Return total entropy and the entropy with missing dropped."
    n = sum(counts.itervalues())
    if n == 0 or len(counts) <= 1:
        return (0,None)
    s = sum(c*math.log(c,2) for c in counts.itervalues())
    entropy_total = math.log(n,2) - s / n
    if missing in counts:
        if len(counts) == 2:
            entropy_present = 0
        else:
            nonN = counts[missing]
            n -= nonN
            entropy_present = math.log(n,2) - (s - nonN * math.log(nonN,2)) / n
    else: entropy_present = None
    if scaledto1:
        if entropy_total is not None:
            entropy_total /= math.log(len(counts), 2)
        if entropy_present is not None:
            entropy_present /= math.log(len(counts), 2)
    return (entropy_total,entropy_present)

def dict__str__ (counts, missing = None):
    "Return a short string describing the counter dictionary."
    entropy_total,entropy_present = dict_entropy(counts,missing)
    return "len={l:,d}; sum={s:,d}; entropy={e:g}{p:s}".format(
        l=len(counts),s=sum(counts.itervalues()),e=entropy_total,
        p=("" if entropy_present is None else
           "/{p:g}".format(p=entropy_present)))

def title2missing (title):
    return tuple([None] * len(title)) if isinstance(title,tuple) else None

def title2string(title, sep='-'):
    return sep.join(str(o) for o in title) if not isinstance(title,str) else title

# http://stackoverflow.com/questions/613183/python-sort-a-dictionary-by-value
# http://stackoverflow.com/questions/28839182/sorting-dictionary-by-value-and-lexicographical
def counter2pairs (counter):
    # count: reverse, value: lexicographical
    return sorted(counter.iteritems(), key=lambda (k,v): (-v,k))

def dict_drop_rare (counter, min_count):
    return dict((k,v) for (k,v) in counter.iteritems() if v >= min_count)

def counter_aggregate (dicts):
    ret = dict()
    for di in dicts:
        for what,count in di.iteritems():
            ret[what] = ret.get(what,0) + count
    return ret

def counter2string (counter, sep="; ", maxlen=None):
    total = sum(counter.itervalues())
    pairs = counter2pairs(counter)
    if maxlen and maxlen < len(pairs):
        suffix = "...%d omitted" % (len(pairs) - maxlen)
        pairs = pairs[:maxlen]
    else:
        suffix = ""
    return sep.join("[{k:s}: {v:,d} ({p:.2%})]".format(
        k=str(k),v=v,p=float(v)/total) for (k,v) in pairs)+suffix

class PrintCounter (object):
    min_count_default = 0        # omit if count less that this OR
    max_row_default = sys.maxint # ... already have this many rows OR
    min_percent_default = 0.     # ... percent less that this
    min_row_default = 10 # ... but ONLY IF already printed at least this much
    min_omit_default = 10      # ... AND do NOT omit less than this much
    header_default = '==='     # printed before the counter title
    prefix_default = None      # printed before the list
    suffix_default = None      # printed after the list

    def __init__ (self, *args, **kwargs):
        # args -- tuple of anonymous arguments
        # kwargs -- dictionary of named arguments
        if len(args) > 0:
            if len(kwargs) == 0:
                kwargs = args[0]
            else: raise Exception("PrintCounter: cannot mix anonymous & named")
        self.min_count = default_None(kwargs.get('pc_min_count'), PrintCounter.min_count_default)
        self.max_row = default_None(kwargs.get('pc_max_row'), PrintCounter.max_row_default)
        self.min_percent = default_None(kwargs.get('pc_min_percent'), PrintCounter.min_percent_default)
        self.min_row = default_None(kwargs.get('pc_min_row'), PrintCounter.min_row_default)
        self.min_omit = default_None(kwargs.get('pc_min_omit'), PrintCounter.min_omit_default)
        self.header = default_None(kwargs.get('pc_header'), PrintCounter.header_default)
        self.prefix = default_None(kwargs.get('pc_prefix'), PrintCounter.prefix_default)
        self.suffix = default_None(kwargs.get('pc_suffix'), PrintCounter.suffix_default)
        self.total = dict()     # fill it outside for cross-percentages

    @staticmethod
    def add_arguments (parser):
        parser.add_argument('-pc-min_count', type=int, help='for PrintCounter')
        parser.add_argument('-pc-max_row', type=int, default=100, help='for PrintCounter')
        parser.add_argument('-pc-min_percent', type=float, default=1.0, help='for PrintCounter')
        parser.add_argument('-pc-min_row', type=int, help='for PrintCounter')
        parser.add_argument('-pc-min_omit', type=int, help='for PrintCounter')
        parser.add_argument('-pc-header', help='for PrintCounter')
        parser.add_argument('-pc-prefix', help='for PrintCounter')
        parser.add_argument('-pc-suffix', help='for PrintCounter')

    def out (self, counter, title, missing = None):
        if missing is None:
            missing = title2missing(title)
        title = title2string(title)
        total = sum(counter.itervalues())
        num_rows = len(counter)
        if total == num_rows:
            print "{h:s} {t:s} {n:,d} items: {i:s}".format(
                h=self.header,t=title,n=num_rows,i=counter.keys())
            return False
        small5 = dict_drop_rare(counter,5)
        if len(small5) == len(counter) or len(small5) < 2:
            print "{h:s} {t:s} ({a:s})".format(
                h=self.header,t=title,a=dict__str__(counter,missing))
        else:
            print "{h:s} {t:s} ({a:s})/({s:s})".format(
                h=self.header,t=title,a=dict__str__(counter,missing),
                s=dict__str__(small5,missing))
        row = 0
        left = 1
        if not self.prefix is None:
            print self.prefix
        def as_ascii (o):
            if isinstance(o,str):
                return o
            if isinstance(o,unicode):
                return o.encode('utf-8')
            return str(o)
        for obj, count in counter2pairs(counter):
            percent = float(count) / total
            row += 1
            omit = num_rows - row + 1
            if ((count < self.min_count or row > self.max_row
                 or 100 * percent < self.min_percent)
                and omit > self.min_omit and row > self.min_row):
                print " - omitted {o:,d} rows ({l:.2%})".format(o=omit,l=left)
                if not self.suffix is None:
                    print self.suffix
                return True     # truncated
            left -= percent
            xp = ("" if obj not in self.total else
                  " ({p:.2%})".format(p=float(count)/self.total[obj]))
            if isinstance(obj,tuple):
                print " {r:5d} {o:s} {c:12,d} {p:6.2%}{xp:s}".format(
                    r=row, o=" ".join(as_ascii(o).rjust(30) for o in obj),
                    c=count, p=percent, xp = xp)
            else:
                print " {r:5d} {o:30s} {c:12,d} {p:6.2%}{xp:s}".format(
                    r=row, o=as_ascii(obj), c=count, p=percent, xp=xp)
        if not self.suffix is None:
            print self.suffix
        return False            # no truncation

    @staticmethod
    def csv (counter, title, destination, logger=None, smallest = 0):
        if isinstance(destination, str):
            if isinstance(title,tuple):
                destination += "-".join(str(o) for o in title) + ".csv"
            else:
                destination += title + ".csv"
            info("Writing {r:,d} rows to [{d:s}]".format(
                r=len(counter),d=destination),logger)
            with open(destination,"wb") as dest:
                PrintCounter.csv(counter, title, dest, smallest=smallest)
            wrote(destination,logger=logger)
        else:
            writer = csv.writer(destination)
            if isinstance(title,tuple):
                writer.writerow(list(title)+["count"])
                for observation,count in counter2pairs(counter):
                    if count >= smallest:
                        writer.writerow([unicode(x).encode('utf-8')
                                         for x in observation]+[count])
            else:
                writer.writerow([title,"count"])
                # writer.writerows(counter2pairs(counter))
                for observation,count in counter2pairs(counter):
                    if count >= smallest:
                        writer.writerow([unicode(observation).encode('utf-8'),count])
            # chances are, write() above will write a better message than this
            #if isinstance(destination,file) and os.path.isfile(destination.name):
            #    wrote(destination.name,logger=logger)

# http://stackoverflow.com/questions/390250/elegant-ways-to-support-equivalence-equality-in-python-classes
class CommonMixin(object):
    def __eq__(self, other):
        return type(other) is type(self) and self.__dict__ == other.__dict__
    def __ne__(self, other):
        return not self.__eq__(other)
    def __str__(self):
        return str(self.__dict__)

def wilson (success, total):
    "Return the center and the half-length of the Wilson confidence interval"
    z = 1.96
    p = float(success) / total
    scale = 1 / (1 + z*z / total)
    center = ( p + z*z / (2 * total) ) * scale
    halfwidth = z * math.sqrt( p*(1-p) / total + z*z/(4*total*total) ) * scale
    return (center, halfwidth)

# pass an empty collections.defaultdict(int) as types
# and it will be filled with type counts
# NB: this will double count objects which appear multiple times in containers
def sizeof (obj, types = None):
    ret = sys.getsizeof(obj)
    if types is not None:
        types[type(obj).__name__] += 1
    if (isinstance(obj,list) or isinstance(obj,tuple) or
        isinstance(obj,set) or isinstance(obj,frozenset)):
        for x in obj:
            ret += sizeof(x, types = types)
        return ret
    if isinstance(obj,dict):
        for k,v in obj.iteritems():
            ret += sizeof(k, types = types) + sizeof(v, types = types)
        return ret
    return ret

def bytes2string (s):
    if bitlen(s) > 10:
        return "{b:,d} bytes ({a:s}B)".format(b=s,a=asBigNumberBin(s))
    else:
        return "{b:,d} bytes".format(b=s)

def filesize2string (f):
    return bytes2string(os.path.getsize(f))

def reading (f,logger = None):
    info("Reading {s:s} from [{f:s}]".format(s=filesize2string(f),f=f),logger)

def wrote (f,logger = None):
    info("Wrote {s:s} into [{f:s}]".format(s=filesize2string(f),f=f),logger)

def enum (name, values):
    return type(name, (), dict(zip(values,values)))

def enum_get (cl, val):
    ret = cl.__dict__.get(val)
    if val == ret: return ret
    raise ValueError("enum_get: Bad value for Enum",cl.__name__,val)

def read_multimap (inf, delimiter, col1, col2, logger = None,
                   keyproc = None, valproc = None):
    'Read a multi-map from a TSV/CSV stream with 2 columns.'
    if isinstance(inf,str):
        reading(inf,logger=logger)
        with open(inf) as ins:
            return read_multimap(ins,delimiter,col1,col2,logger=logger,
                                 keyproc=keyproc,valproc=valproc)
    ret = dict()
    lines = 0
    for row in csv.reader(inf,delimiter=delimiter,escapechar='\\'):
        if len(row) <= max(col1,col2):
            warn("Bad line %s, aborting" % (row),logger)
            break
        lines += 1
        key = row[col1].strip()
        if keyproc is not None:
            key = keyproc(key)
        val = row[col2].strip()
        if valproc is not None:
            val = valproc(val)
        try:
            s = ret[key]
        except KeyError:
            s = ret[key] = set()
        if val in s:
            warn("Duplicate value [%s] for key [%s]" % (val,key),logger)
        s.add(val)
    info("Read {l:,d} lines with {k:,d} keys and {v:,d} values".format(
        l=lines,k=len(ret),v=sum([len(s) for s in ret.itervalues()])),logger)
    return ret

if __name__ == '__main__':
    test()
