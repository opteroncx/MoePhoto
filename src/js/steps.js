import $ from 'jquery'
import { texts, registryLanguageListener } from './common.js'

const panels = {
  index: {
    text: '请选择一项功能',
    draggable: 1
  }
}

const isType = type => x => type === typeof x
const setSubText = target => (item, key) =>
  isType('object')(target[key]) && (isType('object')(item) || isType('string')(item))
    ? setPanelTexts(target[key], item) : target[key] = item
const setPanelTexts = (target, t) => {
  Array.isArray(t) ? t.forEach(setSubText(target))
    : isType('string')(t) ? (target.text = t)
      : (_ => { for (let key in t) setSubText(target)(t[key], key) })()
}
registryLanguageListener(lang => setPanelTexts(panels, lang))

const eventTypes = ['change', 'click', 'focus', 'blur', 'hover']
const optionType = { radio: 1, checkbox: 1 }
const bindChild = panel => child => {
  child = panel.args[child]
  child.slave = true
  return child
}
const addPanel = (panelName, panel) => {
  panel.name = panelName
  panel.initOpt = {}
  let listeners = {}
  eventTypes.forEach(type => listeners[type] = [])
  for (let argName in panel.args) {
    let arg = panel.args[argName]
    let selector = `input[name=${argName}]`, bindFlag = 0
    arg.name = argName
    if (optionType[arg.type])
      arg.values.forEach(v => {
        v.name = argName
        v.type = arg.type
        v.binds = v.binds && v.binds.map(bindChild(panel))
        bindFlag |= !!v.binds
        v.checked ? panel.initOpt[argName] = v.value : 0
      })
    else
      panel.initOpt[argName] = arg.value
    let changeOpt = arg.type === 'checkbox' ? (ev, opt) => {
      opt[argName] || (opt[argName] = {})
      opt[argName][ev.target.value] = ev.target.checked
    } : arg.type === 'file' ? (ev, opt) => opt[argName] = ev.target.files
        : (ev, opt) => opt[argName] = ev.target.value,
      _c = arg.change ? arg.change.bind(arg) : _ => 0
    arg.change = (ev, opt) => {
      changeOpt(ev, opt)
      context.refreshCurrentStep()
      return _c(ev, opt) || bindFlag
    }
    eventTypes.filter(type => arg[type]).forEach(type => listeners[type].push({
      selector,
      f: ev => arg[type](ev, context.getOpt()) && context.refreshPanel()
    }))
  }
  panel.listeners = listeners
  panels[panelName] = panel
}

const reduceApply = (...arr) => arr.reduce((pre, cur) => cur(pre))
const endPoint = _ => ''
const getAttributes = next => (item, opt) => (item.attributes ? item.attributes.join(' ') : '') + next(item, opt)
const getValue = (item, opt) => opt[item.name] != null ? opt[item.name] : item.value
const getClassAttr = classes => classes && classes.length ? ` class="${classes.join(' ')}"` : ''
const getValueLabel = next => (item, opt) => item.text ? `<span class="opValue">${item.text}${next(item, opt)}</span>` : ''
const getOptionLabel = next => (item, opt) => next(item, opt) + (item.text ? `<span class="opValue">${item.text}</span>` : '')
const getDisabled = next => (item, opt) => (item.disabled ? ' disabled' : '') + next(item, opt)
const isChecked = opt => item => (!opt[item.name] && item.checked) || opt[item.name] === item.value || opt[item.name][item.value]
const getChecked = next => (item, opt) => (isChecked(opt)(item) ? ' checked' : '') + next(item, opt)
const getInputTag = (next, getValue = item => item.value) => (item, opt) =>
  `<input type="${item.type}" name="${item.name}" value="${getValue(item, opt)}"
    ${getClassAttr(item.classes)}${next(item, opt)}>`
const getInputView = next => (item, opt) => item.view ? item.view(next(item, opt)) : next(item, opt)
const getInputText = getInputView(getInputTag(getDisabled(getAttributes(endPoint)), getValue))
const getBindHTML = (parent, opt) => item => {
  var _d = item.disabled, res
  item.disabled = !isChecked(opt)(parent)
  res = getArgHTML(item, opt, false)
  item.disabled = _d
  return res
}
const getInputCheckBinds = next => (item, opt) => next(item, opt) +
  (item.binds ? item.binds.map(getBindHTML(item, opt)).join('') : '')
const getInputCheckOption = reduceApply(endPoint, getAttributes, getDisabled, getChecked,
  getInputTag, getInputView, getOptionLabel, getInputCheckBinds)
const getNoteHTML = text => `<li>${text}</li>`
const getNotes = item =>
  item.notes && item.notes.length ?
    ['<ul class="visible-md visible-lg description">', ...item.notes.map(getNoteHTML), '</ul>'].join('') : ''
const getArgHTML = (item, opt, hr = true) =>
  (hr ? '<hr>' : '') +
  `<span class="argName${hr ? ' col-sm-2' : ''}">${item.text}</span>` +
  elementTypeMapping[item.type](item, opt) +
  getNotes(item, opt)
const getCheckedItem = (item, opt) => item.values.filter(isChecked(opt))
const flatArray = arrs => arrs.reduce((res, arr) => res.concat(arr))
const getInputCheck = (item, opt) =>
  item.values.map(v => getInputCheckOption(v, opt))
    .concat(flatArray(getCheckedItem(item, opt).map(getNotes)))
    .join('')
const elementTypeMapping = {
  radio: getInputCheck,
  checkbox: getInputCheck,
  number: getInputText,
  text: getInputText,
  file: getInputText
}
const getArgsHTML = (args, opt) => {
  var res = []
  for (let key in args)
    if (!args[key].slave)
      res.push(getArgHTML(args[key], opt))
  return res.join('')
}
const getPanelTitle = (pos, text) =>
  pos != null ? `<header>${texts.step}<span class="order">${context.getCorrectedPos(pos)}</span>
    <span class="op">${text}</span></header>`
    : `<header>${$('#options header').html()}</header>`
const getPanelView = next => (panel, opt, pos) => getPanelTitle(pos, panel.text) + next(panel, opt)
const getPanelInnerView = next => (panel, opt) => '<hr>' + getDiv(x => x)(panel.description) + next(panel.args, opt)
const getIndexItemHTML = name => `<a class="btn btn-lg" name="${name}">${panels[name].text}</a>`
const getDiv = next => (item, id) =>
  ['<div', id ? ` id="${id}"` : '', ' class="visible-md visible-lg">', next(item), '</div>'].join('')
const getIndexHTML = _ =>
  ['<hr><div class="btn-group-justified">',
    ...context.getFeatures().map(getIndexItemHTML),
    '</div>', getDiv(endPoint)(0, 'description')].join('')
const getIndexStepHTML = pos =>
  `<div class="step add" data-position=${pos} draggable><header><p>${texts.add}</p></header><div class="visible-lg"></div></div>`
const getIndicatorHTML = (pos, condition) => condition ? `<hr class="drag-indicator" data-position=${pos}>` : ''
const getSelected = selected => selected ? ' selected' : ''
const getDraggable = step => step.panel.draggable ? ' draggable' : ''
const getDelete = step => step.panel.draggable ? `<a href="#" class="delete">${texts.delete}</a>` : ''
const getStepOpt = step => {
  var panel = step.panel, opt = Object.assign({}, step.opt), args = panel.args, res = []
  for (let key in opt) args[key].type === 'number' && (opt[key] = +opt[key])
  if (panel.view) {
    opt = panel.view(opt)
  } else for (let key in opt)
    if (optionType[args[key].type]) {
      let f = args[key].type === 'radio' ? item => item.value === opt[key] : item => opt[key][item.value]
      opt[key] = args[key].values.filter(f).map(item => item.text).join(',')
    }
  for (let key in opt) res.push(getValueLabel((_, opt) => texts.labelSplitter + opt[key])(args[key], opt))
  return res.join('')
}
const getStepInnerHTML = (step, pos) =>
  `<header>${texts.step}<span class="order">${context.getCorrectedPos(pos)}</span><span class="op">${step.panel.text}</span>
    ${getDelete(step)}</header><div class="configs visible-md visible-lg">${getStepOpt(step)}</div>`
const getPureStepHTML = (step, pos, selected = 0) =>
  `<div class="step${getSelected(selected)}" data-position=${pos}${getDraggable(step)}>${getStepInnerHTML(step, pos)}</div>`
const getStepNIndicatorHTML = pos => (step, i) =>
  getIndicatorHTML(i, step.panel.draggable && (i === 0 || !steps[i - 1].panel.draggable))
  + (step == indexStep ? getIndexStepHTML(i) : getPureStepHTML(step, i, i === pos))
  + getIndicatorHTML(i + 1, step.panel.draggable)

const toggleSelected = ele => {
  $('#steps .step').removeClass('selected')
  ele.classList.add('selected')
}

for (let panelName in panels) addPanel(panelName, panels[panelName])

const indexStep = { panel: panels.index }
const steps = []

const newStep = panel => ({ panel, opt: Object.assign({}, panel.initOpt) })
const context = (steps => {
  var pos = 0, index, addibleFeatures
  const getOpt = _ => steps[pos].opt
  const getCorrectedPos = p => p > index ? p - 1 : p
  const addStep = panelName => {
    steps.splice(index, 0, newStep(panels[panelName]))
    pos = index
    index += 1
    refreshSteps()
  }
  const selectStep = (p, ele) => {
    if (p !== pos) {
      pos = p
      refreshPanel(0)
      toggleSelected(ele)
    }
  }
  const removeStep = pos => {
    steps.splice(pos, 1)
    refreshSteps()
  }
  const compareOp = (a, b) => a.position - b.position
  const pushNewStep = panel => steps.push(newStep(panel))
  const setFeatures = features => {
    addibleFeatures = features.filter(name => panels[name].draggable && name !== 'index')
    let ps = features.map(name => panels[name]),
      tops = ps.filter(panel => panel.position >= 0).sort(compareOp),
      bottoms = ps.filter(panel => panel.position < 0).sort(compareOp)
    tops.forEach(pushNewStep)
    index = steps.length
    features.includes('index') && steps.push(indexStep)
    bottoms.forEach(pushNewStep)
    let listeners = panels.index.listeners
    addibleFeatures.forEach(name => listeners.click.push({
      selector: `a.btn[name=${name}]`,
      f: _ => addStep(name)
    }))
    refreshSteps()
  }
  const getFeatures = _ => addibleFeatures
  const refreshPanel = (refreshStep = 1) => {
    var step = steps[pos], panel = step.panel,
      listeners = panel.listeners
    $('#options').html(getPanelView(panel === panels.index ? getIndexHTML : getPanelInnerView(getArgsHTML))(panel, step.opt, pos))
    refreshStep && context.refreshCurrentStep()
    eventTypes.forEach(type => listeners[type].forEach(o => $(o.selector).on(type, o.f)))
  }
  const refreshCurrentStep = _ => $(`#steps .step[data-position=${pos}]`).html(getStepInnerHTML(steps[pos], pos))
  const refreshSteps = target => {
    target && (pos = target)
    index = steps.indexOf(indexStep)
    index < 0 && (index = steps.length)
    refreshPanel(0)
    $('#steps').html(steps.map(getStepNIndicatorHTML(pos)).join(''))
  }
  return {
    getOpt, getFeatures, getCorrectedPos, setFeatures, selectStep,
    refreshPanel, removeStep, refreshSteps, refreshCurrentStep
  }
})(steps)

const initListeners = _ => {
  $('#steps').on('dragenter', '.drag-indicator', function () {
    this.classList.add('over')
  }).on('dragleave drop', '.drag-indicator', function () {
    this.classList.remove('over')
  }).on('dragstart', '.step', function (ev) {
    this.classList.add('dragging')
    ev.originalEvent.dataTransfer.setData('text', this.getAttribute('data-position'))
  }).on('dragend', '.step', function () {
    this.classList.remove('dragging')
  }).on('drop', '.drag-indicator', function (ev) {
    ev.stopPropagation()
    let posSrc = +ev.originalEvent.dataTransfer.getData('text'), posTarget = +this.getAttribute('data-position')
    if (!(posSrc === posTarget || posSrc === posTarget - 1)) {
      posTarget > posSrc && (posTarget -= 1)
      let src = steps[posSrc]
      steps[posSrc] = steps[posTarget]
      steps[posTarget] = src
      context.refreshSteps(posTarget)
    }
  }).on('dragover', '.drag-indicator', ev => ev.preventDefault())
    .on('click', 'a.delete', function (ev) {
      ev.stopPropagation()
      context.removeStep(+this.parentNode.parentNode.getAttribute('data-position'))
    }).on('click', '.step', function () {
      context.selectStep(+this.getAttribute('data-position'), this)
    })
  $('#options').on('mouseenter', 'a.btn', function () {
    let panelName = this.getAttribute('name')
    panels[panelName].description && $('#description').html(panels[panelName].description)
  }).on('mouseout', 'a.btn', _ => $('#description').html(''))
}

const submit = data => data.set('steps', JSON.stringify(steps
  .filter(step => step !== indexStep)
  .map(step => {
    let opt = Object.assign({}, step.opt), panel = step.panel
    if (panel.submit) {
      for (let key in panel.args) {
        let arg = panel.args[key]
        if (arg.type === 'number' && opt[key] != null)
          opt[key] = +opt[key]
      }
      opt = panel.submit(opt, data)
    }
    opt && (opt.op = panel.op ? panel.op : panel.name)
    return opt
  }).filter(opt => !!opt)))

export { addPanel, initListeners, submit, context }