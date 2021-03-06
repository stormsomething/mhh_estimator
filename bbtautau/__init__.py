import logging

log = logging.getLogger()
log.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# create formatter
formatter = logging.Formatter('%(name)s:%(levelname)s: %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
log.addHandler(ch)
