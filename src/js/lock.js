import $ from 'jquery'
import { appendText, texts } from './common.js'
import { addPanel, initListeners, submit, context } from './steps.js'
import { setup } from './progress.js'

const path = '/lockInterface'
const lockPanel = {
  text: '锁定设置',
  description: '把后台锁起来一段时间内不让用，其他访问会看到“忙碌中”',
  position: 0,
  args: {
    duration: {
      type: 'number',
      text: '持续时间',
      value: 10,
      view: appendText('second'),
      classes: ['input-number'],
      notes: ['只能趁没人在用的空隙提交', '计时精度为整数秒']
    }
  }
}
addPanel('lock', lockPanel)
$(document).ready(_ => {
  initListeners()
  context.setFeatures(['lock'])
  const progress = setup({
    onProgress: texts.lockRunning,
    progress: $('#progress'),
    beforeSend: submit,
    path,
    noCheckFile: 1,
    success: _ => progress.final(''),
    error: (_, xhr) => {
      let busy = xhr ? xhr.responseJSON ? xhr.responseJSON.eta == null ? 0 : 1 : 0 : 0
      if (busy) {
        progress.setStatus(+xhr.responseJSON.eta)
      } else {
        console.error(xhr)
      }
    }
  })
})