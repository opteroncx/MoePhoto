const defaultOpt = {
  cache: 'no-store'
}
const throwError = e =>
  setTimeout(() => {
    throw e
  }, 0)
const formatData = res => {
  let data = res.text().then(str => (str ? JSON.parse(str) : void 0))
  if (!res.ok) {
    return data.then(data => {
      throw [data, res.statusText]
    })
  }
  return data
}
const encodeParam = pair => pair.map(encodeURIComponent).join('=')
const encodeParams = o =>
  Object.entries(o)
    .map(encodeParam)
    .join('&')
export const newMessager = (url, session, opt = {}) => {
  var m = { url, session, xhr: null, status: 0 }

  var listeners = {}

  var opt = Object.assign({}, defaultOpt, opt)
  const getUrl = data => `${m.url}?${encodeParams(Object.assign({ session: m.session }, data))}`
  const onError = a => m.fire({ type: 'error', data: a[0], error: a[1] })
  const pend = res => {
    m.status
      ? (m.xhr = fetch(getUrl(), opt)
        .then(formatData)
        .catch(onError)
        .then(data => m.fire({ type: 'message', data })))
      : (m.xhr = null)
    return res
  }
  m.on = (type, listener) => {
    type = String(type)
    var ls, i
    if (listeners[type] == null) {
      ls = []
      listeners[type] = ls
    }
    ls = listeners[type]
    for (i = ls.length; i-- && ls[i] !== listener;);
    if (i < 0) ls.push(listener)
    return m
  }
  m.on('message', pend)
  m.on('open', pend)
  m.removeEventListener = (type, listener) => {
    type = String(type)
    var ls = listeners[type]
    if (!ls || !ls.length) return m
    listeners[type] = ls.filter(l => l !== listener)
    if (!listeners[type].length) delete listeners[type]
    return m
  }
  m.fire = function (event) {
    event.target = m
    var ls = listeners[event.type]
    if (!ls || !ls.length) return
    return ls.map(listener => {
      try {
        if (typeof listener.handleEvent === 'function') {
          return listener.handleEvent(event)
        } else {
          return listener.call(this, event)
        }
      } catch (e) {
        throwError(e)
      }
    })
  }
  m.open = data => {
    if (!m.status) {
      m.status = 1
      m.xhr = fetch(getUrl(data), opt)
        .then(formatData)
        .catch(onError)
        .then(data => m.fire({ type: 'open', data }))
    }
    return m.xhr
  }
  m.abort = _ => (m.status = 0)
  return m
}
export const getSession = _ => {
  var cookie = window.document.cookie
  let start = cookie.indexOf('session=') + 8
  let end = cookie.indexOf(';', start)
  if (end < 0) end = cookie.length
  let res = unescape(cookie.substring(start, end))
  ;+res || console.error(`No session found in cookie ${cookie}`)
  return res
}
