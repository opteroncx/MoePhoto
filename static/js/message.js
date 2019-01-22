(function (global) {
  "use strict"
  const $ = global.$
  const doc = global.document
  const defaultOpt = {
    cache: false
  }
  const throwError = e => setTimeout(() => {throw e}, 0)
  const formatData = data => data && typeof data === 'string' ? JSON.parse(data) : data
  const newMessager = (url, session, opt = {}) => {
    var m = { url, session, xhr: null, status: 0 }, listeners = {},
    opt = Object.assign({}, defaultOpt, opt)
    const getOpt = data => {
      var res = Object.assign({}, opt)
      if (data == null) {
        data = { session: m.session }
      }
      if (typeof data === 'object') {
        data.session = m.session
        res.url = m.url
      } else {
        res.url = `${m.url}?session?=${m.session}`
      }
      res.data = data
      return res
    }
    const onError = (xhr, _, error) => m.fire({ type: 'error', data: formatData(xhr.responseJSON), error })
    const pend = res => {
      if (m.status) {
        m.xhr = $.ajax(getOpt())
        .then((data => m.fire({ type: 'message', data: formatData(data) })), onError)
      } else m.xhr = null
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
      if ((!ls) || (!ls.length)) return m
      listeners[type] = ls.filter(l => l !== listener)
      if (!listeners[type].length) delete listeners[type]
      return m
    }
    m.fire = function (event) {
      event.target = m
      var ls = listeners[event.type]
      if ((!ls) || (!ls.length)) return
      return ls.map(listener => {
        try {
          if (typeof listener.handleEvent === "function") {
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
        m.xhr = $.ajax(getOpt(data))
        .then((data => m.fire({ type: 'open', data: formatData(data) })), onError)
      }
      return m.xhr
    }
    m.abort = _ => m.status = 0
    return m
  }
  const getSession = _ => {
    var cookie = doc.cookie
    let start = cookie.indexOf('session=') + 8
    let end = cookie.indexOf(';', start)
    if (end < 0) end = cookie.length
    return unescape(cookie.substring(start, end))
  }
  (factory => {
    if (typeof module === "object" && typeof module.exports === "object") {
      let v = factory(exports)
      module.exports = v
    } else if (typeof define === "function" && define.amd) {
      define(["exports"], factory)
    } else {
      factory(global)
    }
  })(exports => {
    exports.getSession = getSession
    exports.newMessager = newMessager
  })
})(typeof window !== 'undefined' ? window : typeof self !== 'undefined' ? self : this)