import $ from 'jquery'
import { getResource, getSession } from './common.js'
import app from './app.js'
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
    intervalId = 0
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
const bindMessager = ($ele, messager) => {
  const progress = bindProgress($ele)
  const onMessage = event => {
    if (!event.data) {
      progress.hide().setMessage('空闲中')
      progress.status || messager.abort()
    } else {
      let data = event.data
      if (data) {
        progress.setMessage(data)
        if (data.eta) progress.setStatus(+data.eta)
      }
    }
  }
  messager.on('message', onMessage)
    .on('open', onMessage)
  progress.final = msg => {
    progress.status = 0
    messager.abort()
    progress.hide().setMessage(msg)
    messager.xhr && messager.xhr.then(_ => progress.setMessage(msg))
    return progress
  }
  progress.begin = msg => {
    progress.status = 1
    return progress.show().setMessage(msg)
  }
  return progress
}
const setup = opt => {
  var stopButton = $('#StopButton').hide(), runButton = $('#RunButton'), total = 0
  const setPreview = opt.outputImg ? (_ => {
    let idle = true
    opt.outputImg.on('load', _ => idle = true)
    return path => {
      if (idle) {
        idle = false
        opt.outputImg.attr('src', path)
      }
    }
  })() : _ => _
  if (!opt.session) opt.session = getSession()
  const onErrorMsg = gone => '忙碌中' + (gone == null ? '' : `，已经过${gone}秒`)
  if (!opt.onProgress) opt.onProgress = onErrorMsg
  const onMessage = e => {
    if (!e.data) return
    let data = e.data
    data.preview ? setPreview(getResource(data.preview)) : 0
    data.total ? total = data.total : 0
    data.gone ? progress.setMessage(opt.onProgress(data.gone, total, data)) : 0
  }
  const messager = app.setup(opt)
  messager.on('message', onMessage).on('open', onMessage)
  const progress = bindMessager(opt.progress, messager)
  opt.onErrorMsg = data => progress.setMessage(onErrorMsg(data.gone, total, data))
  opt.setStatus = progress.setStatus
  opt.setMessage = progress.setMessage
  let beforeSend = opt.beforeSend
  opt.beforeSend = data => {
    runButton.hide()
    stopButton.attr('disabled', false).show()
    progress.begin('正在处理您的任务')
    beforeSend && beforeSend(data)
  }
  let success = opt.success
  opt.success = result => {
    runButton.show()
    stopButton.hide()
    success && success(result, progress)
  }
  opt.error = msg => {
    progress.final(msg)
    runButton.show()
    stopButton.hide()
  }
  stopButton.click(_ => {
    $.ajax({
      url: `/stop?session=${opt.session}`,
      beforeSend: _ => {
        stopButton.attr('disabled', true)
        progress.hide().setMessage('等待保存已处理部分')
      },
      error: (xhr, status, error) => {
        console.error(xhr, status, error)
        progress.final('出错啦')
      }
    })
  })
  return progress
}
const exportApp = { getSession, getResource, setup }
window.app = exportApp
export default exportApp