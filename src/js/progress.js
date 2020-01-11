import $ from 'jquery'
import { getResource, getSession, texts } from './common.js'
import { setup } from './app.js'
const bindProgress = ($ele, context) => {
  var intervalId = 0,
    remain = 0,
    nowStage = 0,
    bar = $ele.find('.progress-bar'),
    statusBox = $ele.find('.status'),
    msgBox = $ele.find('.message'),
    timeBox = $ele.find('.time'),
    $steps = $('#steps'),
    progress
  const elapse = _ => {
    bar[0].value += 1
    remain -= 1
    if (remain < 1) remain = 1
    timeBox.text(texts.timeFormatter(remain))
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
    if (typeof data === 'string') msgBox.html(data)
    return progress
  }
  const setStatus = str => statusBox.html(str) && progress
  const setTime = eta => {
    intervalId || show()
    if (eta >= 0) {
      bar[0].max = eta + +bar[0].value
      remain = eta
      timeBox.text(texts.timeFormatter(remain))
    } else timeBox.text('')
    return progress
  }
  const setStage = data => {
    let stage = (data && data.stage) || 0
    if (stage !== nowStage) {
      let ss = $steps.find('.step')
      ss.removeClass('running')
      !context.changed && stage && ss[stage].classList.add('running')
      nowStage = stage
    }
    return progress
  }
  return (progress = { show, hide, setMessage, setStatus, setTime, setStage })
}
const bindMessager = ($ele, messager, context) => {
  const progress = bindProgress($ele, context)
  const onMessage = event => {
    if (!event.data) {
      progress.hide().setStatus(texts.idle)
      progress.status || messager.abort()
    } else {
      let data = event.data
      data && data.eta && progress.setTime(+data.eta)
    }
  }
  messager.on('message', onMessage).on('open', onMessage)
  progress.final = msg => {
    progress.status = 0
    messager.abort()
    progress.hide().setMessage(msg)
    messager.xhr && messager.xhr.then(_ => progress.setMessage(msg))
    return progress
  }
  progress.begin = msg => {
    progress.status = 1
    context && (context.changed = false)
    return progress.show().setMessage(msg)
  }
  progress.beforeSend = messager.beforeSend
  progress.on = messager.on
  return progress
}
const setupProgress = opt => {
  var stopButton = $('#StopButton').hide(),
    runButton = $('#RunButton'),
    total = 0
  const setPreview = opt.outputImg
    ? (_ => {
        let idle = true
        opt.outputImg.on('load', _ => (idle = true))
        return path => {
          if (idle) {
            idle = false
            opt.outputImg.attr('src', path)
          }
        }
      })()
    : _ => _
  if (!opt.session) opt.session = getSession()
  if (!opt.onProgress) opt.onProgress = texts.onBusy
  const onMessage = e => {
    if (!e.data) return
    let data = e.data
    if (data.eta) {
      runButton.hide()
      stopButton.attr('disabled', false).show()
    }
    data.preview ? setPreview(getResource(data.preview)) : 0
    data.total ? (total = data.total) : 0
    data.gone
      ? progress.setStatus(opt.onProgress(data.gone, total, data))
      : data.eta
      ? progress.setStatus(texts.onBusy(null))
      : 0
    data.stage ? progress.setStage(data) : 0
  }
  const messager = setup(opt)
  messager.on('message', onMessage).on('open', onMessage)
  const progress = bindMessager(opt.progress, messager, opt.context)
  opt.onErrorMsg = data =>
    progress.setStatus(texts.onBusy(data.gone, total, data))
  opt.setStatus = progress.setTime
  opt.setMessage = progress.setMessage
  let beforeSend = opt.beforeSend
  opt.beforeSend = data => {
    runButton.hide()
    stopButton.attr('disabled', false).show()
    progress.begin(texts.running)
    beforeSend && beforeSend(data)
  }
  let success = opt.success,
    error = opt.error
  opt.success = result => {
    runButton.show()
    stopButton.hide()
    progress.setStage()
    success && success(result, progress)
  }
  opt.error = (msg, xhr) => {
    progress.final(msg)
    runButton.show()
    stopButton.hide()
    progress.setStage()
    error && error(xhr)
  }
  stopButton.click(_ => {
    $.ajax({
      url: `/stop?session=${opt.session}`,
      beforeSend: _ => {
        stopButton.attr('disabled', true)
        progress.hide().setMessage(texts.stopping)
      },
      error: (xhr, status, error) => {
        console.error(xhr, status, error)
        progress.final(texts.errorMsg)
      }
    })
  })
  return progress
}

export { setupProgress as setup }
