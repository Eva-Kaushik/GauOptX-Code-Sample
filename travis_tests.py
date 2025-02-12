import matplotlib
matplotlib.use('agg')

import nose, warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
nose.main('GauOptX', defaultTest='GauOptX/testing/', argv=[''])