import { newMessager, texts } from './common.js'
import $ from 'jquery'
const reconnectPeriod = 5
const setup = opt => {
  var options = $('#options')
  if (opt.inputImg && opt.inputImg.length) {
    const readURL = function () {
      if (this.files && this.files[0]) {
        var reader = new FileReader()
        reader.onload = e => opt.inputImg.attr('src', e.target.result)
        reader.readAsDataURL(this.files[0])
      }
    }
    options.on('change', '.imgInp', readURL)
  }
  if (opt.dropZone && opt.dropZone.length) {
    var dropZone = opt.dropZone[0]
    dropZone.addEventListener(
      'dragover',
      e => {
        e.stopPropagation()
        e.preventDefault()
        e.dataTransfer.dropEffect = 'copy'
      },
      false
    )
    dropZone.addEventListener(
      'drop',
      e => {
        e.stopPropagation()
        e.preventDefault()
        let imgInp = options.find('.imgInp')
        imgInp[0].files = e.dataTransfer.files
        imgInp.trigger('change')
      },
      false
    )
  }
  var downloader = $('#downloader')

  var loading = $('#FG')

  var runButton = $('#RunButton')

  var intervalId

  var running = 0
  loading.hide()
  downloader.hide()

  const messager = newMessager('/msg', opt.session)
  const onMessage = event => {
    if (!event.data) {
      messager.abort()
      running && openMessager() && (running = 0)
      runButton.attr('disabled', false)
    } else {
      clearInterval(intervalId)
      let result = event.data.result
      if (result === 'Fail') onError(0, 400, event.data.exception)
      else if (result != null) onSuccess(event.data)
      else running || ((running = 1) && runButton.attr('disabled', true))
    }
  }
  messager
    .on('message', onMessage)
    .on('open', onMessage)
    .on('error', event => {
      console.error(event)
      running = 0
      clearInterval(intervalId)
      runButton.attr('disabled', true)
      let eta = 0
      if (event.data) {
        messager.abort()
        eta = +event.data.eta
        opt.onErrorMsg && opt.onErrorMsg(0, eta, event.data)
      } else eta = reconnectPeriod
      if (eta) {
        eta += Math.random()
        eta = Math.min(Math.max(eta, 0.1), 2147483)
        setTimeout(openMessager, eta * 1000)
      }
      return opt.setStatus && opt.setStatus(eta)
    })
  const openMessager = _ => messager.open({ path: opt.path })
  openMessager()

  const onSuccess = result => {
    console.log(result)
    running = 0
    clearInterval(intervalId)
    loading.hide()
    downloader.show()
    runButton.attr('disabled', false)
    opt.success && opt.success(result.result)
  }

  const onError = (xhr, status, error) => {
    console.error(xhr, status, error)
    running = 0
    clearInterval(intervalId)
    loading.hide()
    opt.error ? opt.error(texts.errorMsg, xhr) : alert(texts.errorMsg)
  }

  const beforeSend = (messager.beforeSend = _ => {
    running = 1
    loading.show()
    intervalId = setInterval(openMessager, 200)
    openMessager()
    return messager
  })

  if (opt.session) {
    runButton.bind('click', _ => {
      var fdata = new FormData()
      opt.beforeSend && opt.beforeSend(fdata)
      if (!(opt.noCheckFile || fdata.get('file').size)) {
        return opt.setMessage ? opt.setMessage(texts.noFileMsg) : alert(texts.noFileMsg)
      }
      $.post({
        url: `${opt.path}?session=${opt.session}`,
        data: fdata,
        contentType: false,
        processData: false,
        beforeSend: _ => beforeSend() && runButton.attr('disabled', true)
      })
    })
  } else {
    let errorMsg = 'Session invailid, please try clear browser data and refresh this page.'
    opt.error ? opt.error(errorMsg) : alert(errorMsg)
  }
  return messager
}

export { setup, texts }
