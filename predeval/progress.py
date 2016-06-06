# progress reporting

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

import argparse
import time
import datetime
import re
import util

def difftime2string (x):
    ax = abs(x)
    if ax < 1: return "%.2fms" % (x*1000.0)
    if ax < 100: return "%.2fsec" % (x)
    if ax < 6000: return "%.2fmin" % (x/60.0)
    if ax < 108000: return "%.2fhrs" % (x/3600.0)
    if ax < 400*24*3600: return "%.2fdays" % (x/(24*3600.0))
    return "%.2fyrs" % (x/(365.25*24*3600))

def elapsed (start):
    return difftime2string(time.time()-start)

def processed (start,count,unit):
    spent = time.time() - start
    return "%d new %ss in %s%s" % (
        count,unit,difftime2string(spent),
        (" (%s/%s)" % (difftime2string(spent/count),unit)) if count else "")

def timing (func, logger = None):
    start = time.time()
    ret = func()
    util.info("Ran %s in %s" % (func, elapsed(start)),logger=logger)
    return ret

difftime_rex = re.compile('^(-)?([0-9.]+)(ms|sec|min|hrs|days|yrs)$')
def parse_difftime (s):
    if s is None:
        return None
    if isinstance(s,int):
        return s
    if not isinstance(s,str):
        raise TypeError("parse_difftime",s)
    m = difftime_rex.match(s)
    if m is None:
        raise ValueError("parse_difftime",s)
    sign,num,units = m.groups()
    num = float(num) * (1 if sign is None else -1)
    if units == "ms": return num / 1000.0
    if units == "sec": return num
    if units == "min": return num * 60
    if units == "hrs": return num * 3600
    if units == "days": return num * 3600 * 24
    if units == "yrs": return num * 3600 * 24 * 365.25
    raise ValueError("parse_difftime",s,units)

def parse_ymdh (s):
    return datetime.datetime.strptime(s,"%Y/%m/%d/%H")

def time2string (t = None):
    return time.strftime("%F %T",time.localtime(t))

def test ():
    print difftime2string(100)
    print parse_difftime("-45min")
    print time2string()

class Done (Exception):
    pass

class Progress (object):
    @staticmethod
    def get_parser (max_ticks = None, tick_report = None,
                    max_time = None, time_report = None,
                    flow_report = None):
        aparse = argparse.ArgumentParser(add_help=False)
        aparse.add_argument('-max-ticks',type=int, default=max_ticks,
                            help='Iterate at most time many times')
        aparse.add_argument('-tick-report',type=int, default=tick_report, metavar='N',
                            help='Report progress every N ticks')
        aparse.add_argument('-max-time',default=max_time,
                            help='Iterate for at most this long (e.g., 4hrs)')
        aparse.add_argument('-time-report',type=int, default=time_report, metavar='S',
                            help='Report progress every S seconds')
        aparse.add_argument('-flow-report', default=flow_report,
                            help='Report progress based on data flow time interval, e.g., every 20min of data')
        return aparse

    def __init__ (self, logger, status, opts, max_possible = None):
        self.logger = logger
        self.status = status
        self.start = time.time()
        self.ticks = 0
        self.last_report_ticks = self.ticks
        self.last_report_time = self.start
        self.max_ticks = min(opts.max_ticks or max_possible,
                             max_possible or opts.max_ticks)
        self.tick_report = opts.tick_report
        self.max_time = parse_difftime(opts.max_time)
        self.time_report = opts.time_report
        try:
            self.date_beg = opts.beg
            self.date_end = opts.end
            self.flow_beg = datetime.datetime.combine(opts.beg, datetime.time.min)
            self.flow_end = datetime.datetime.combine(opts.end, datetime.time.max)
        except AttributeError:
            self.date_beg = self.date_end = self.flow_beg = self.flow_end = None
        self.flow_now = self.flow_beg
        self.flow_report = None if opts.flow_report is None else parse_difftime(opts.flow_report)
        self.last_report_flow = self.flow_now

    def completed_ticks (self):
        if self.max_ticks is None:
            return None
        return float(self.ticks) / self.max_ticks
    def completed_flow (self):
        if self.flow_now is None:
            return None
        return (float((self.flow_now - self.flow_beg).total_seconds()) /
                (self.flow_end - self.flow_beg).total_seconds())
    def completed (self):
        completed_ticks = self.completed_ticks()
        completed_flow = self.completed_flow()
        if completed_flow:
            if completed_ticks:
                return (completed_flow + completed_ticks) / 2
            return completed_flow
        if completed_ticks:
            return completed_ticks
        return None

    def __str__ (self):
        return ("<start=" + time2string(self.start) +
                ('' if self.max_ticks is None else
                 " max_ticks={m:,d}".format(m=self.max_ticks)) +
                ('' if self.max_time is None else
                 " max_time=" + difftime2string(self.max_time)) +
                ('' if self.tick_report is None else
                 " tick_report={t:,d}".format(t=self.tick_report)) +
                ('' if self.time_report is None else
                 " time_report=" + difftime2string(self.time_report)) +
                ('' if self.flow_report is None else
                 " flow_report=" + difftime2string(self.flow_report)) +
                " ticks={t:,d}>".format(t=self.ticks))

    # return (remaining-time, expected-time-at-end)
    def eta (self):
        completed = self.completed()
        if completed is None:
            if self.max_time is None:
                return (None, None)
            end = self.start + self.max_time
            return (end - time.time(), end)
        now = time.time()
        remains = (now - self.start) * (1-completed) / completed
        if self.max_time is None:
            return (remains, now + remains)
        end = self.start + self.max_time
        return (min(remains, end - now), min(now + remains, end))

    # flow_now is the timestamp of the current record
    def tick (self, flow_now = None):
        now = time.time()
        if ((self.max_ticks is not None and self.ticks == self.max_ticks) or
            (self.max_time is not None and now > self.start + self.max_time)):
            raise Done()
        self.ticks += 1
        if flow_now is not None:
            self.flow_now = flow_now
        if ((self.tick_report is not None and
             self.ticks - self.last_report_ticks >= self.tick_report) or
            (self.flow_report is not None and self.flow_now is not None and
             ((self.flow_now - self.last_report_flow).total_seconds()
              >= self.flow_report)) or
            (self.time_report is not None and
             now - self.last_report_time >= self.time_report)):
            self.logger.info("%s",self.report())
            self.last_report_time = now
            self.last_report_ticks = self.ticks
            self.last_report_flow = self.flow_now

    def report (self):
        remains, eta = self.eta()
        s = "" if self.flow_now is None else self.flow_now.strftime(
            "%Y-%m-%d %H:%M:%S ")
        s += "" if self.status is None else self.status()
        if remains is None or remains <= 0:
            return s + "{t:,d}".format(t=self.ticks)
        return s + "{t:,d} ({c:.2%}) ETA: {e:s} ({r:s})".format(
            t=self.ticks,c=self.completed() or 0,e=time2string(eta),
            r=difftime2string(remains))

    @staticmethod
    def test ():
        p = Progress(None, None, Progress.get_parser().parse_args())
        p.max_ticks = 1000
        p.ticks = 100
        p.start -= 100
        print p
        print p.report()
        p.tick()
        print p
        print p.report()

if __name__ == '__main__':
    test()
    Progress.test()
