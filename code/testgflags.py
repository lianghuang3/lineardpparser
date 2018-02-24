import gflags as flags
FLAGS=flags.FLAGS

import sys

flags.DEFINE_boolean("profile", False, "profile perceptron training")

argv = FLAGS(sys.argv)

print FLAGS.profile

for flag in FLAGS:
    print FLAGS[flag]
