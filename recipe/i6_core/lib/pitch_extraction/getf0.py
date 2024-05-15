#!/usr/bin/python3
# -*- coding: utf-8 -*-

import gzip
import sys

# import _tkinter
from tkinter import *
from tkSnack import *


def getF0(audioFile, pitchFile, frameadd=0):
    root = Tkinter.Tk()
    initializeSnack(root)
    s = Sound(load=audioFile)

    f0 = s.pitch("esps")
    # f0 = s.pitch(method='ESPS')
    with gzip.open(pitchFile, "wt") as out:
        for line in f0:
            # pitch pv rms acpeak
            out.write("%.3f %i %.3f %.6f\n" % (line[0], line[1], line[2], line[3]))

        #### add frames if necessary
        for count in range(frameadd):
            out.write("%.3f %i %.3f %.6f\n" % (line[0], line[1], line[2], line[3]))


if __name__ == "__main__":
    import os.path
    from optparse import OptionParser

    optparser = OptionParser(usage="usage: %prog [OPTION] <ref-file>")
    optparser.add_option(
        "-o", "--output", dest="out", default="-", help="output", metavar="FILE"
    )
    optparser.add_option(
        "-p", "--pad", dest="padframes", default="0", help="output", metavar="NUMBER"
    )

    opts, args = optparser.parse_args()
    assert len(args) == 1
    getF0(args[0], opts.out, int(opts.padframes))
