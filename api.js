// https://cnodejs.org/topic/4ffed8544764b729026b1da3
const http = require('http')
const fs = require('fs')
const path = require('path')

const BOUNDARYPREFIX = 'nbglme'
const max = 9007199254740992
const getBoundary = () => BOUNDARYPREFIX + (Math.random() * max).toString(36)
const mixin = (...source) => Object.assign({}, ...source)
const imageMimes = {
  '.bmp': 'image/bmp',
  '.png': 'image/png',
  '.gif': 'image/gif',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.tif': 'image/tiff',
  '.tiff': 'image/tiff',
  '.webp': 'image/webp'
}
const videoMimes = {
  '.flv': 'video/x-flv',
  '.mp4': 'video/mp4',
  '.ts': 'video/MP2T',
  '.3gp': 'video/3gpp',
  '.ogm': 'video/ogg',
  '.mov': 'video/quicktime',
  '.avi': 'video/x-msvideo',
  '.wmv': 'video/x-ms-wmv',
  '.webm': 'video/webm',
  '.mkv': 'application/octet-stream',
  '.mpg': 'video/mpeg',
  '.mpeg': 'video/mpeg'
}
const mimes = mixin(imageMimes, videoMimes)

const mkpic = dir => (name, fn) => {
  let ext = path.extname(name)
  let mime = mimes[ext]
  if (!mime) throw new Error(`Invalid File Format`)
  let filename = path.join(dir, name)

  return fs.readFile(filename, (err, data) =>
    fn(
      err,
      [
        'Content-Transfer-Encoding: binary',
        `Content-Disposition: form-data; name="file"; filename="${name}"`,
        `Content-Type: ${mime}`,
        '',
        ''
      ].join('\r\n'),
      data
    )
  )
}

const mkfield = (field, value) => `Content-Disposition: form-data; name="${field}"\r\n\r\n${value}`

const requestPromise = options => {
  let _resolve, _reject
  let p = new Promise((resolve, reject) => {
    ;[_resolve, _reject] = [resolve, reject]
  })
  let req = http.request(options, _resolve)
  req.on('error', _reject)
  return [p, req]
}

const responsePromise = res => {
  let _resolve, _reject
  let p = new Promise((resolve, reject) => {
    ;[_resolve, _reject] = [resolve, reject]
  })
  const { statusCode } = res
  if (statusCode !== 200) {
    res.resume()
    throw res.statusMessage
  }
  res.setEncoding('utf8')
  let rawData = ''
  res.on('data', chunk => (rawData += chunk))
  res.on('end', _ => {
    try {
      data = rawData.length ? JSON.parse(rawData) : void 0
      return _resolve(data)
    } catch (e) {
      _reject(e)
    }
  })
  res.on('error', _reject)
  return p
}

const genRequest = options => (dir, media, param = {}, data = ['']) => {
  for (let k in param) data.push(mkfield(k, param[k]))
  data.push('')
  let boundary = getBoundary()
  let padBoundary = '--' + boundary
  let makeMedia = mkpic(dir)
  let f = (resolve, reject) => (err, head, mediaData) => {
    if (err) return reject(err)
    data[data.length - 1] = head
    let body0 = data.join(`\r\n${padBoundary}\r\n`)
    let [p, req] = requestPromise(options)
    req.setHeader('Content-Type', `multipart/form-data; boundary=${boundary}`)
    let tail = `\r\n${padBoundary}--`
    let buf = Buffer.concat([Buffer.from(body0, 'utf8'), mediaData, Buffer.from(tail, 'utf8')])
    req.setHeader('Content-Length', buf.length)
    req.write(buf)
    return resolve([p, req])
  }
  return new Promise((resolve, reject) => makeMedia(media, f(resolve, reject)))
}

const getRequest = options => {
  let [p, req] = requestPromise(options)
  req.end()
  return p.then(responsePromise)
}

const queryResult = (options, callback) => _ => getRequest(options).then(callback)

const throwError = err => {
  throw err
}

const batchRequest = (options, callback, onError = throwError) =>
  async function * (dir, files, getParam) {
    let connection = Promise.resolve()
    let r = genRequest(options)

    let i = 0
    for await (let fp of files) {
      try {
        let [[p, req]] = await Promise.all([r(dir, fp, getParam(i)), connection])
        req.end()
        connection = p.then(responsePromise).catch(err => {
          throw err
        })
        yield connection.then(callback)
      } catch (e) {
        connection = Promise.resolve()
        yield Promise.reject([e, fp, i]).catch(onError)
      }
      i += 1
    }
  }

const traverse = async function * (dir = '', pathName = '') {
  let files = []
  let dirs = []
  await fs.promises.readdir(path.join(dir, pathName), { withFileTypes: true }).then(arr => {
    files = arr.filter(fp => fp.isFile())
    dirs = arr.filter(fp => fp.isDirectory())
  })
  yield * files.map(fp => Promise.resolve(path.join(pathName, fp.name)))
  for (let dir_ of dirs) {
    yield * traverse(dir, path.join(pathName, dir_.name))
  }
}

const genParam = (steps, start, total) => {
  let param = { steps: JSON.stringify(steps) }
  return i => {
    param.gone = i + 1
    param.eta = i ? (((total - i) * (Date.now() - start)) / i) * 1e-3 : 10
    return param
  }
}

const log = res => {
  console.log(res)
  return res
}

const error = res => {
  console.error(res)
  return res
}

const processType = (opt, type, single = false) => {
  let r = type
    ? [videoMimes, 'video', '%2Fvideo_enhance']
    : [imageMimes, 'image', '%2F' + (single ? 'image_enhance' : 'batch_enhance')]
  return r.concat([mixin(opt, { timeout: 500, path: `/msg?session=0&path=${r[2]}` })])
}

const getPreset = (opt, p, preset) =>
  getRequest(mixin(opt, { timeout: 2000, path: `/preset?path=${p}&name=${preset}` }))

const API = (url = 'localhost', port = 2333) => {
  let o = { host: url, port }
  const process = async (filepath, preset) => {
    let { dir, base, ext } = path.parse(filepath)
    let type = ext in videoMimes
    let [, p, , optMsg] = processType(o, type, true)
    let response = getPreset(o, p, preset)
    let optPost = mixin(o, { path: `/${p}_enhance?session=0`, method: 'POST' })
    let steps = (await response).steps
    let [q, req] = await genRequest(optPost)(dir, base, { steps: JSON.stringify(steps) })
    req.end()
    return q.then(responsePromise).then(_ => getRequest(optMsg))
  }
  const processFolder = async (dir, preset, type = 0, callback = log) => {
    let files = []

    let results = []

    let [m, p, fakePath, optMsg] = processType(o, type)
    let response = getPreset(o, p, preset)
    for await (const name of traverse(dir)) if (path.extname(name).toLowerCase() in m) files.push(name)
    if (!files.length) return results
    let steps = (await response).steps
    let optPost = mixin(o, {
      path: `/${p}_enhance?session=0&total=${files.length}&path=${fakePath}`,
      method: 'POST'
    })
    let br = batchRequest(optPost, queryResult(optMsg, callback), error)
    let getParam = genParam(steps, Date.now(), files.length)
    for await (let res of br(dir, files, getParam)) results.push(res)
    return results
  }
  return { process, processFolder }
}
exports.MoePhoto = API
