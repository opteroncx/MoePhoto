(function (global) {
  "use strict"
  const bindProgress = $ele => {
    var intervalId = 0, remain = 0, bar = $ele.find('.progress-bar'),
      msgBox = $ele.find('.message'), timeBox = $ele.find('.time'), progress
    const timeFormatter = time => `，预计还需要${time.toFixed(2)}秒`
    const elapse = _ => {
      bar[0].value += 1
      remain -= 1
      if (remain < 1) remain = 1
      timeBox.text(timeFormatter(remain))
    }
    const show = _ => {
        intervalId && clearInterval(intervalId)
        bar[0].value = 0
        intervalId = setInterval(elapse, 1000)
        timeBox.show()
        bar.show()
        return progress
      }
    const hide = _ => {
      intervalId && clearInterval(intervalId)
      remain = 0
      timeBox.hide()
      bar.hide()
      return progress
    }
    const setMessage = data => {
      if (typeof data === 'string') {
        msgBox.html(data)
      } else if (data && data.result) {
        msgBox.html(data.result)
      }
      return progress
    }
    const setStatus = eta => {
      intervalId || show()
      bar[0].max = eta + +bar[0].value
      remain = eta
      timeBox.text(timeFormatter(remain))
      return progress
    }
    return progress = { show, hide, setMessage, setStatus }
  }
  const reconnectPeriod = 5
  const bindMessager = (path, $ele, idleFunc, session) => {
    session || (session = global.getSession())
    const progress = bindProgress($ele)
    const onMessage = event => {
      if (!event.data) {
        progress.hide().setMessage('空闲中')
        messager.abort()
        return typeof idleFunc === 'function' ? idleFunc(): void 0
      } else {
        let data = event.data
        if (data) progress.setMessage(data)
        if (data.eta) {
          let eta = +data.eta
          progress.setStatus(eta)
        }
      }
    }
    const messager = global.newMessager(path, session)
    .on('message', onMessage)
    .on('open', onMessage)
    .on('error', event => {
      console.error(event)
      let eta = 0
      if (event.data) {
        messager.abort()
        progress.setMessage(event.data)
        eta = +event.data.eta
      } else eta = reconnectPeriod
      if (eta) {
        eta += Math.random()
        eta = Math.min(Math.max(eta, .1), 2147483)
        progress.setStatus(eta).setMessage('忙碌中').show()
        setTimeout(messager.open, eta * 1000)
      } else {
        progress.hide()
      }
    })
    messager.open()
    progress.final = msg => {
      messager.abort()
      progress.hide().setMessage(msg)
      messager.xhr.then(_ => progress.setMessage(msg))
      return progress
    }
    progress.begin = msg => {
      setTimeout(messager.open, 200)
      return progress.show().setMessage(msg)
    }
    progress.messager = messager
    return progress
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
    exports.bindMessager = bindMessager
  })
})(typeof window !== 'undefined' ? window : typeof self !== 'undefined' ? self : this)