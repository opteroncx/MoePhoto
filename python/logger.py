import json
import logging
import logging.config
import time
import os

class JSONFormatter(logging.Formatter):
  converter = time.gmtime
  default_msec_format = '%s.%03dZ'
  default_time_format = '%Y-%m-%dT%H:%M:%S'

  def formatException(self, exc_info):
    result = super(JSONFormatter, self).formatException(exc_info)
    return repr(result)

  def format(self, r):
    super(JSONFormatter, self).format(r)
    time = super(JSONFormatter, self).formatTime(r)
    res = dict(time=time
      , level=r.levelname, message=r.msg, file=r.module
      , line=r.lineno)
    if r.stack_info:
      res['stack'] = r.stack_info
    if r.exc_text:
      res['exception'] = r.exc_text
    return json.dumps(res, ensure_ascii=False, separators=(',', ':'))

class LocalFormatter(logging.Formatter):
  converter = time.gmtime
  default_msec_format = '%s.%03d'

  def format(self, r):
    message = super(LocalFormatter, self).format(r)
    time = super(LocalFormatter, self).formatTime(r)
    if r.levelno >= logging.WARNING:
      return '{}|{}|{:8}|{}:{}|{}'.format(
        time, r.name, r.levelname, r.module, r.lineno, message)
    else:
      return '{}|{}|{}'.format(time, r.name, message)

def initLogging(logFile=None):
  LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
      'JSON': {
        '()': JSONFormatter
      },
      'local': {
        '()': LocalFormatter
      }
    },
    'handlers': {
      'console': {
        'class': 'logging.StreamHandler',
        'formatter': 'local',
      }
    },
    'root': {
      'level': 'INFO',
      'handlers': ['console']
    }
  }
  if logFile:
    if not os.path.exists(os.path.dirname(logFile)):
      os.makedirs(os.path.dirname(logFile))

    LOGGING['handlers']['logFile'] = {
      'class': 'logging.handlers.RotatingFileHandler',
      'filename': logFile,
      'mode': 'w',
      'backupCount': 1,
      'maxBytes': 1 << 24,
      'encoding': 'utf-8',
      'formatter': 'JSON'
    }
    LOGGING['root']['handlers'].append('logFile')
  logging.config.dictConfig(LOGGING)
  return logging