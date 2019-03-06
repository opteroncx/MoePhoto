import $ from 'jquery'
import { texts, getSession } from './common.js'
import { setup } from './progress.js'

const path = '/systemInfo'

$(document).ready(_ => {
  var session = getSession(), GPUfrees = $('.freeGPUMemory'), logButton = $('#logButton')
  const progress = setup({
    session,
    progress: $('#progress'),
    path,
    success: freeMems => progress.final('') && GPUfrees.each((i, elem) => elem.innerHTML = freeMems[i])
  })
  session && $.ajax({
    url: path,
    data: { session },
    cache: false,
    beforeSend: _ => progress.begin(texts.running).beforeSend(),
    error: (xhr, status, error) => {
      let busy = xhr ? xhr.responseJSON ? xhr.responseJSON.eta == null ? 0 : 1 : 0 : 0
      if (busy) {
        progress.setStatus(+xhr.responseJSON.eta)
        GPUfrees.text(texts.needRefresh)
      } else {
        console.error(xhr, status, error)
        progress.final(texts.errorMsg)
      }
    }
  })
  logButton.attr('href', '#')
  logButton.click(_ => {
    logButton.attr('disabled', true)
    $.get('/log')
      .then(data => {
        data = data.split('\n').filter(line => line.length)
          .map(JSON.parse).map(ev => {
            ev.time = new Date(ev.time)
            return ev
          })
        if (data.length) {
          progress.final(texts.logWritten)
          data.forEach(item => console.log(item))
        } else progress.final(texts.noMoreLog)
      })
      .catch(console.error.bind(console))
      .then(_ => logButton.attr('disabled', false))
  })
})