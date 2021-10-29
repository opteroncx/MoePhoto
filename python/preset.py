import os
import json
import time
from flask import request, safe_join
from userConfig import compareVersion, VERSION
version = VERSION

cache = {}

getBrief = lambda item: dict(name=item['name'], notes=item.get('notes', []))

def loadPreset(path):
  def f(filename, raw=False):
    if not filename.endswith('.json'):
      return
    name = filename.rpartition('.')[0]
    filename = safe_join(path, filename)
    if not os.path.exists(filename):
      return
    mtime = cache[name][0] if name in cache else 0
    st_mtime = os.stat(filename).st_mtime
    if mtime < st_mtime:
      try:
        with open(filename, 'r', encoding='utf-8') as fp:
          text = fp.read()
          item = json.loads(text)
          name = item['name'] # should equals name
          if compareVersion(version, item['version']) < 0:
            return 'Incompatible version' if raw else None
          cache[name] = (st_mtime, text, getBrief(item))
      except Exception as e:
        return str(e) if raw else None
    return cache[name][1] if raw else cache[name][2]
  return f

def savePreset(path):
  def f(data):
    if not os.path.exists(path):
      os.mkdir(path)
    brief = getBrief(json.loads(data))
    name = brief['name']
    with open(safe_join(path, name + '.json'), 'w', encoding='utf-8') as fp:
      fp.write(data)
      cache[name] = (time.time(), data, brief)
    return name
  return f

def initPreset(config):
  global version
  if 'version' in config:
    version = config['version']

def preset():
  try:
    pType = request.values['path']
    if not pType in {'video', 'image'}:
      return '', 403
    path = '.user/preset_' + pType
    name = request.values['name'] if 'name' in request.values else None
    data = request.values['data'] if 'data' in request.values else None
    if data:
      return savePreset(path)(data), 200
    else:
      if name:
        res = cache[name][1] if name in cache else loadPreset(path)(name + '.json', True)
        if res:
          return res, 200
        else:
          return '', 404
      else:
        if os.path.exists(path):
          res = [*filter(None, map(loadPreset(path), os.listdir(path)))]
          return json.dumps(res , ensure_ascii=False, separators=(',', ':')), 200
        else:
          return '[]', 200
  except Exception:
    return '', 403