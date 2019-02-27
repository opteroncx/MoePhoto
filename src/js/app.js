import { getResource, getSession, newMessager, texts } from './common.js'
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
    dropZone.addEventListener('dragover', e => {
      e.stopPropagation()
      e.preventDefault()
      e.dataTransfer.dropEffect = 'copy'
    }, false)
    dropZone.addEventListener('drop', e => {
      e.stopPropagation()
      e.preventDefault()
      options.find('.imgInp')[0].files = e.dataTransfer.files
    }, false)
  }
  var downloader = $('#downloader'), loading = $('#FG'), runButton = $('#RunButton'), intervalId
  loading.hide()
  downloader.hide()

  const messager = newMessager('/msg', opt.session)
    .on('message', event => {
      if (!event.data) {
        messager.abort()
        runButton.attr('disabled', false)
      } else {
        clearInterval(intervalId)
        let result = event.data.result
        if (result === 'Fail')
          onError(0, 400, event.data.exception)
        else if (result)
          onSuccess(result)
        else
          runButton.attr('disabled', true)
      }
    }).on('error', event => {
      console.error(event)
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
        eta = Math.min(Math.max(eta, .1), 2147483)
        setTimeout(openMessager, eta * 1000)
      }
      return opt.setStatus && opt.setStatus(eta)
    })
  const openMessager = _ => messager.open({ path: opt.path })
  openMessager()

  const onSuccess = result => {
    console.log(result)
    clearInterval(intervalId)
    loading.hide()
    downloader.show()
    runButton.attr('disabled', false)
    opt.success && opt.success(result.result)
  }

  const onError = (xhr, status, error) => {
    console.error(xhr, status, error)
    clearInterval(intervalId)
    loading.hide()
    opt.error ? opt.error(texts.errorMsg) : alert(texts.errorMsg)
  }

  if (opt.session) {
    runButton.bind('click', function () {
      var fdata = new FormData()
      opt.beforeSend && opt.beforeSend(fdata)
      if (!fdata.get('file').size) return opt.setMessage ? opt.setMessage(texts.noFileMsg) : alert(texts.noFileMsg)
      $.ajax({
        url: `${opt.path}?session=${opt.session}`,
        type: "POST",
        data: fdata,
        cache: false,
        contentType: false,
        processData: false,
        async: true,
        dataType: 'json',
        beforeSend: _ => {
          loading.show()
          runButton.attr('disabled', true)
          intervalId = setInterval(openMessager, 200)
        },
        success: onSuccess,
        error: onError
      })
    })
  } else {
    let errorMsg = 'Session invailid, please try clear browser data and refresh this page.'
    opt.error ? opt.error(errorMsg) : alert(errorMsg)
  }
  return messager
}
const app = { getSession, getResource, setup, texts }
if (window.app)
  Object.assign(window.app, app)
else
  window.app = app
export default app