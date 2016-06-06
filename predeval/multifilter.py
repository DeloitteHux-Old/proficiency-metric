# Remove a regexp from a set of parallel files.
# Usage:
#  python multifilter.py -rex '\tNegative' ../data/magnetic_annotated.txt ../data/magnetic_results*

import argparse
import re
import tempfile
import os
import util
logger = util.get_logger("MF")

def finish (i, o, p):
    i.close()
    os.close(o)
    os.rename(p,i.name)
    util.wrote(i.name,logger)

def run (rex, main, others):
    util.reading(main.name,logger)
    mainH,mainP = tempfile.mkstemp(dir=os.path.dirname(main.name))
    othersIO = list()
    for o in others:
        util.reading(o.name,logger)
        h,p = tempfile.mkstemp(dir=os.path.dirname(o.name))
        othersIO.append((o,h,p))
    # read the files in parallel
    dropped = 0
    lines = 0
    for line in main:
        lines += 1
        prefix = line.split('\t')[0]
        keep = rex.search(line) is None
        if keep:
            os.write(mainH,line)
        else:
            dropped += 1
        for i,o,_p in othersIO:
            line1 = i.readline()
            prefix1 = line1.split('\t')[0].rstrip()
            if prefix1 != prefix:
                raise Exception("prefix mismatch",prefix,prefix,i.name)
            if keep:
                os.write(o,line1)
    for i in others:
        line1 = i.readline()
        if line1 != '':
            raise Exception('uneven files',line1,i.name)
    logger.info("Dropped {:,d} lines out of {:,d}".format(dropped,lines))
    # close streams and rename
    finish(main,mainH,mainP)
    for i,o,p in othersIO:
        finish(i,o,p)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Filter several parallel files')
    ap.add_argument('-rex',help='the regular expression to filter on',
                    required=True)
    ap.add_argument('main',help='the main file searched in',
                    type=argparse.FileType('r'))
    ap.add_argument('others',help='other filtered files',nargs='*',
                    type=argparse.FileType('r'))
    args = ap.parse_args()
    run(re.compile(args.rex),args.main,args.others)
