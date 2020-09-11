import { newMessager, texts } from './common.js'
import $ from 'jquery'
const reconnectPeriod = 5
const setComparison = opt =>
  opt.outputImg &&
  opt.inputImg &&
  opt.inputImg.length &&
  $('.twentytwenty-container').twentytwenty({
    no_overlay: true
  })
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
  setComparison(opt)
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

  var running = 0
  loading.hide()
  downloader.hide()

  const messager = newMessager('/msg', opt.session)
  const onMessage = event => {
    if (event.data) {
      let result = event.data.result
      if (result === 'Fail' && !onError(0, 400, event.data.exception)) return
      if (result != null) onSuccess(event.data)
      else running || ((running = 1) && runButton.attr('disabled', true))
    } else {
      messager.abort()
      running && setTimeout(_ => running && openMessager(), 200)
    }
  }
  messager
    .on('message', onMessage)
    .on('open', onMessage)
    .on('error', event => {
      console.error(event)
      endSession(1)
      runButton.attr('disabled', true)
      let eta = 0
      if (event.data) {
        eta = +event.data.eta
        opt.onErrorMsg && opt.onErrorMsg(0, eta, event.data)
      } else eta = reconnectPeriod
      if (eta) {
        console.log('error', eta)
        eta += Math.random()
        eta = Math.min(Math.max(eta, 0.1), 2147483)
        setTimeout(openMessager, eta * 1000)
      }
      return opt.setStatus && opt.setStatus(eta)
    })
  const openMessager = _ => {
    return messager.open({ path: opt.path })
  }
  openMessager()

  const onSuccess = result => {
    console.log(result)
    endSession()
    downloader.show()
    runButton.attr('disabled', false)
    opt.success && opt.success(result.result)
  }

  const endSession = retry => {
    running = 0
    retry || loading.hide()
    messager.abort()
  }

  const onError = (xhr, status, error) => {
    console.error(xhr, status, error)
    if (opt.ignoreError) return 1
    endSession()
    opt.error ? opt.error(texts.errorMsg, xhr) : alert(texts.errorMsg)
  }

  const beforeSend = (messager.beforeSend = _ => {
    running = 1
    loading.show()
    openMessager()
    return messager
  })

  if (opt.session) {
    runButton.bind('click', _ => {
      var fdata = new FormData()
      opt.beforeSend && opt.beforeSend(fdata)
      if (
        !(
          opt.noCheckFile ||
          fdata.noCheckFile ||
          (fdata.get('file') && fdata.get('file').size)
        )
      ) {
        return opt.setMessage
          ? opt.setMessage(texts.noFileMsg)
          : alert(texts.noFileMsg)
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
    let errorMsg =
      'Session invailid, please try clear browser data and refresh this page.'
    opt.error ? opt.error(errorMsg) : alert(errorMsg)
  }
  return messager
}

export { setup, texts }
