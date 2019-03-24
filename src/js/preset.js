import { texts, VERSION } from './common.js'
import { context, serializeSteps } from './steps.js'
const fetchPath = '/preset?path='
const presetNotesName = 'preset-notes'
const presetNotesEditor = {
  type: 'textarea',
  text: '说明和描述',
  name: presetNotesName,
  notes: [
    '一行一个'
  ]
}
const presetSelectName = 'preset-selector', presetListName = 'preset-list'
const genPresetArgs = path => {
  var url = `${fetchPath}${path}`
  var cache = {}
  var loadPresetArg = {
    type: 'select',
    text: '预置'
  }
  var savePresetArg = {
    type: 'list',
    text: '预置'
  }
  var applyPresetButton = {
    type: 'button',
    value: '应用预置',
    click: () => {
      var name = document.getElementsByName(presetSelectName)[0].value
      fetch(`${url}&name=${name}`)
        .then(response => cache[name] = response.json())
        .then(data => data.steps)
        .then(context.loadOptions)
    },
    disabled: true
  }
  var savePresetButton = {
    type: 'button',
    value: '保存预置',
    click: ev => {
      var name = document.getElementById(presetListName).value
      if (!name) return
      ev.target.value = texts.running
      var notesText = document.getElementsByName(presetNotesName)[0].value
      var notes = notesText ? notesText.split('\n') : void 0
      var data = new FormData()
      data.set('path', path)
      data.set('name', name)
      data.set('data', JSON.stringify({
        version: VERSION,
        name,
        notes,
        path,
        steps: serializeSteps(2)
      }, null, 2))
      fetch(url, {
        method: 'POST',
        body: data
      }).then(() => ev.target.value = savePresetButton.value)
    },
    disabled: true
  }
  return [loadPresetArg, savePresetArg, applyPresetButton, savePresetButton]
}
export { genPresetArgs, presetNotesEditor }