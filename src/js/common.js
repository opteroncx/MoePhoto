import('bootstrap/dist/css/bootstrap.min.css')
import('../css/style.css')
import('../css/font-awesome.css')
import('../css/loader.css')
import('../css/component.css')
import('../css/Yanone Kaffeesatz-200,300,400,700.css')
import('../css/Roboto-400,100,100italic,300,300italic,400italic,500,500italic,700,700italic,900,900italic.css')
import $ from 'jquery'
window.$ = $
import 'bootstrap'
import { jarallax } from 'jarallax'
import './SmoothScroll.min.js'
import './jquery.totemticker.js'
import './move-top.min.js'
import './easing.js'
import './numscroller-1.0.js'
import './custom-file-input.js'
import { getSession, newMessager } from './message.js'
jarallax($('.jarallax'), {
  speed: 0.5,
  imgWidth: 1366,
  imgHeight: 768
})
$(_ => {
  $('#vertical-ticker').totemticker({
    row_height: '100px',
    next: '#ticker-next',
    previous: '#ticker-previous',
    stop: '#stop',
    start: '#start',
    mousestop: true
  })
})
$(document).ready($ => {
  $(".scroll ").click(function (event) {
    event.preventDefault();
    $('html,body').animate({
      scrollTop: $(this.hash).offset().top
    }, 1000)
  })
  $().UItoTop({
    easingType: 'easeOutQuart'
  })
})
const getResource = path => [path, '?', (new Date()).getTime()].join('')
const texts = {
  step: '步骤',
  add: '点击添加...',
  delete: '删除',
  labelSplitter: '：',
  pixel: '像素',
  fps: '帧每秒',
  noFileMsg: '缺少输入文件',
  errorMsg: '出错啦',
  idle: '空闲中',
  finish: '完成啦',
  running: '正在处理您的任务',
  processing: '处理中',
  stopping: '等待保存已处理部分',
  onBusy: gone => '忙碌中' + (gone == null ? '' : `，已经过${gone}秒`),
  timeFormatter: time => `，预计还需要${time.toFixed(2)}秒`
}
const appendText = key => text => text + texts[key]
const [setLanguage, registryLanguageListener] = (_ => {
  const listeners = []
  return [language => {
    for (let key of language)
      key in texts && (texts[key] = language[key])
    listeners.forEach(language)
  },
  listener => {
    listener in listeners || listeners.push(listener)
  }]
})()
export { getResource, getSession, newMessager, appendText, texts, setLanguage, registryLanguageListener }