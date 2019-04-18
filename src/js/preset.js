import { texts, VERSION } from './common.js'
import { context, serializeSteps, getOptions } from './steps.js'
const fetchPath = '/preset?path='
const presetNotesName = 'preset-notes'
const presetNotesEditor = {
  type: 'textarea',
  text: '说明和描述',
  ignore: true,
  name: presetNotesName,
  value: '',
  notes: [
    '一行一个说明'
  ],
  classes: ['input-text']
}
const None = () => void 0
const getByName = name => document.getElementsByName(name)[0]
const getById = id => document.getElementById(id)
const fillSelect = s => names => s.options = names.map(name => ({ value: name }))
const fillEmptyOption = (s, text = texts.empty) => s.options = [{ value: text, disabled: true }]
const fillHtml = (s, ele) => ele.innerHTML = getOptions(s)
const genPresetArgs = (path, presetSelectName = 'preset', presetListName = 'preset') => {
  var url = `${fetchPath}${path}`
  var cache = {}, fetching = 0, names = []
  const saveNames = data => names = data.map(item => (cache[item.name] = item) && item.name)
  const fetchNames = (target) => {
    fetching = 1
    cache = {}
    return fetch(url)
      .then(response => response.json())
      .then(saveNames)
      .then(fillSelect(target))
      .then(() => fetching = 2)
      .catch(e => {
        console.error(e)
        fillEmptyOption(target, texts.errorMsg)
        fetching = 0
        throw e
      })
  }
  var loadPresetArg = {
    type: 'select',
    text: '使用预置设定',
    classes: ['input-text', 'input-short'],
    ignore: true,
    binds: ['apply'],
    focus: () => {
      let ele = getByName(presetSelectName)
      if (fetching > 1 && !loadPresetArg.options) {
        names.length ? fillSelect(loadPresetArg)(names) : fillEmptyOption(loadPresetArg)
        fillHtml(loadPresetArg, ele)
        ele.selectedIndex = -1
      }
      if (fetching) return
      fillEmptyOption(loadPresetArg, texts.fetching + '...')
      fillHtml(loadPresetArg, ele)
      return fetchNames(loadPresetArg)
      .then(() => {
        names.length || fillEmptyOption(loadPresetArg)
        fillHtml(loadPresetArg, ele)
        ele.selectedIndex = -1
      })
      .catch(() => fillHtml(loadPresetArg, ele))
      .finally(None)
    },
    change: () => {
      let name = getByName(presetSelectName).value, t
      applyPresetButton.disabled = !name
      name && (loadPresetArg.notes = cache[name].notes) && (t = loadPresetArg.notes) && (presetNotesEditor.value = t.join('\n'))
    }
  }
  var savePresetArg = {
    type: 'list',
    name: presetListName,
    text: '保存设定为预置',
    value: '',
    classes: ['input-text', 'input-short'],
    ignore: true,
    binds: ['savePreset'],
    focus: () => {
      let ele = getById(`list-${presetListName}`)
      if (fetching > 1 && !savePresetArg.options) {
        fillSelect(savePresetArg)(names)
        fillHtml(savePresetArg, ele)
      }
      if (fetching) return
      let input = getByName(presetListName)
      return fetchNames(savePresetArg)
      .then(() => fillHtml(savePresetArg, ele))
      .then(() => {
        console.log(ele)
      })
      .catch(e => input.value = e)
      .finally(None)
    },
    change: () => {
      let name = getByName(presetListName).value, t
      savePresetArg.value = name
      savePresetButton.disabled = !name
      name && (t = cache[name]) && (t = t.notes) && (presetNotesEditor.value = t.join('\n'))
    }
  }
  var applyPresetButton = {
    type: 'button',
    value: '应用',
    ignore: true,
    text: '',
    click: () => {
      var name = getByName(presetSelectName).value
      fetch(`${url}&name=${name}`)
        .then(response => response.json())
        .then(data => cache[name] = data)
        .then(data => data.steps)
        .then(context.loadOptions)
        .catch(console.error.bind(console))
    },
    disabled: true
  }
  var savePresetButton = {
    type: 'button',
    ignore: true,
    value: '保存',
    text: '',
    click: ev => {
      var name = getByName(presetListName).value
      if (!name) return
      ev.target.value = texts.running
      var notesText = getByName(presetNotesName).value
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
      }).catch(console.error.bind(console))
      .finally(() => ev.target.value = savePresetButton.value)
    },
    disabled: true
  }
  return [loadPresetArg, savePresetArg, applyPresetButton, savePresetButton]
}
export { genPresetArgs, presetNotesEditor }