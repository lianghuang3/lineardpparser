import os, resource

def cpu():
    return (resource.getrusage(resource.RUSAGE_SELF).ru_utime+
            resource.getrusage(resource.RUSAGE_SELF).ru_stime)

# from http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/286222


_proc_status = '/proc/%d/status' % os.getpid()

_scale = {'kB': 1024.0, 'mB': 1024.0*1024.0,
          'KB': 1024.0, 'MB': 1024.0*1024.0}

scale = {'B': 1.0,
		 'kB': 1024.0, 'mB': 1024.0*1024.0,
         'KB': 1024.0, 'MB': 1024.0*1024.0,
		 'gB': 1024*1024*1024.0, 'GB': 1024*1024*1024.0}

def human(v):
    base = "%3.1lf B" % v
    for a in ['GB', 'MB', 'kB']:
        if v > scale[a]:
            return "%3.1lf %s\t(%s)" % (v/scale[a], a, base)
    return base 

def _VmB(VmKey):
    '''Private.
    '''
    global _proc_status, _scale
     # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
     # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
     # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]

def memory(since=0.0):
    '''Return memory usage in bytes.
    '''
    return _VmB('VmSize:') - since

def resident(since=0.0):
    '''Return resident memory usage in bytes.
    '''
    return _VmB('VmRSS:') - since


def stacksize(since=0.0):
    '''Return stack size in bytes.
    '''
    return _VmB('VmStk:') - since
