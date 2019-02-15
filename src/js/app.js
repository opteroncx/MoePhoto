import { getResource, getSession, newMessager } from './common.js'
import $ from 'jquery'
const reconnectPeriod = 5
const setup = opt => {
  var imgInp = $("#imgInp")
  if (opt.inputImg && opt.inputImg.length) {
    const readURL = function () {
      if (this.files && this.files[0]) {
        var reader = new FileReader()
        reader.onload = e => opt.inputImg.attr('src', e.target.result)
        reader.readAsDataURL(this.files[0])
      }
    }
    imgInp.change(readURL)
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
      imgInp[0].files = e.dataTransfer.files
    }, false)
  }
  var downloader = $('#downloader'), loading = $('#FG'), runButton = $('#RunButton'),
    imgUpload = document.querySelector("#imgUpload"), intervalId, inTaskFlag = 0
  loading.hide()
  downloader.hide()

  const messager = newMessager('/msg', opt.session)
    .on('message', event => {
      if (!event.data) {
        messager.abort()
        runButton.attr('disabled', false)
      } else {
        clearInterval(intervalId)
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

  if (opt.session) {
    const noFileMsg = '缺少输入文件', errorMsg = '出错啦'
    runButton.bind('click', function () {
      var fdata = new FormData(imgUpload)
      if (!fdata.get('file').size) return opt.setMessage ? opt.setMessage(noFileMsg) : alert(noFileMsg)
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
          opt.beforeSend && opt.beforeSend(fdata)
        },
        success: result => {
          console.log(result)
          clearInterval(intervalId)
          loading.hide()
          downloader.show()
          runButton.attr('disabled', false)
          opt.success && opt.success(result.result)
        },
        error: (xhr, status, error) => {
          console.error(xhr, status, error)
          clearInterval(intervalId)
          loading.hide()
          opt.error ? opt.error(errorMsg) : alert(errorMsg)
        }
      })
    })
  }
  return messager
}
const app = { getSession, getResource, setup }
window.app = app
export default app