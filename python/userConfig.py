import json
from os.path import exists
from defaultConfig import defaultConfig
VERSION = '4.6'
configPath = '.user/config.json'
manifestPath = 'manifest.json'

def compareVersion(a, b):
  for v0, v1 in zip(a.split('.'), b.split('.')):
    n0 = int(v0)
    n1 = int(v1)
    if n0 < n1:
      return -1
    elif n0 > n1:
      return 1
  if len(a) < len(b):
    return -1
  elif len(a) > len(b):
    return 1
  else:
    return 0

def setConfig(config, version=VERSION):
  for key in defaultConfig:
    config[key] = defaultConfig[key][0]
  if exists(manifestPath):
    with open(manifestPath,'r',encoding='utf-8') as manifest:
      config['version'] = json.load(manifest)['version']
  if exists(configPath):
    with open(configPath, 'r', encoding='utf-8') as fp:
      try:
        userConfig = json.load(fp)
      except:
        raise UserWarning('Loading user config failed, fallback to defaults.')
      c = compareVersion(version, userConfig['version'])
      del userConfig['version']
      if c > 0:
        raise UserWarning('User config is too old and not supported by current version.')
      for key in userConfig:
        config[key] = userConfig[key][0]