import $ from 'jquery'
import { texts, registryLanguageListener } from './common.js'

const panels = {
  index: {
    text: '请选择一项功能',
    draggable: 1
  }
}

const isType = type => x => type === typeof x
const None = () => void 0
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
const tags = { select: 'select', textarea: 'textarea' }
const optionType = { radio: 1, checkbox: 1 }
const bindChild = (panel, checked) => child => {
  child = panel.args[child]
  child.slave = true
  child.disabled = !checked
  return child
}
const setBinds = (binds, value) => binds && binds.forEach(child => child.disabled = value)
const addPanel = (panelName, panel) => {
  panel.name = panelName
  panel.initOpt = {}
  let listeners = {}
  eventTypes.forEach(type => listeners[type] = [])
  for (let argName in panel.args) {
    let arg = panel.args[argName]
    arg.name || (arg.name = argName)
    let selector = `${tags[arg.type] ? tags[arg.type] : 'input'}[name=${arg.name}]`
    arg.bindFlag = 0
    if (optionType[arg.type]) {
      arg.values.forEach(v => {
        v.name = argName
        v.type = arg.type
        v.binds = v.binds && v.binds.map(bindChild(panel, v.checked))
        arg.bindFlag |= !!v.binds
        v.checked ? panel.initOpt[argName] = v.value : 0
      })
      arg.dataType || arg.values.some(v => typeof v.value !== 'number') || (arg.dataType = 'number')
    }
    else
      arg.ignore || (panel.initOpt[argName] = arg.value)
    arg.bindFlag |= !!arg.binds
    arg.binds = arg.binds && arg.binds.map(bindChild(panel))
    let changeOpt = arg.ignore ? None : arg.type === 'checkbox' ? (ev, opt) => {
      opt[argName] || (opt[argName] = {})
      opt[argName][ev.target.value] = ev.target.checked
    } : arg.type === 'file' ? (ev, opt) => opt[argName] = ev.target.files
        : (ev, opt) => opt[argName] = ev.target.value,
      _c = arg.change ? arg.change.bind(arg) : _ => 0,
      changeBinds = arg.type === 'checkbox' ? ev =>
        setBinds(arg.values.find(v => v.value === ev.target.value).binds, !ev.target.checked)
        : arg.type === 'radio' ? ev =>
          arg.values.forEach(v => setBinds(v.binds, v.value !== ev.target.value))
          : None
    arg.change = (ev, opt) => {
      changeOpt(ev, opt)
      changeBinds(ev)
      context.refreshCurrentStep()
      return _c(ev, opt) || arg.bindFlag
    }
    eventTypes.filter(type => arg[type]).forEach(type => listeners[type].push({
      selector,
      f: ev => {
        Promise.resolve(arg[type](ev, context.getOpt()))
          .then(flag => flag && context.refreshPanel())
      }
    }))
  }
  panel.listeners = listeners
  panels[panelName] = panel
}

const reduceApply = (...arr) => arr.reduce((pre, cur) => cur(pre))
const endPoint = _ => ''
const getAttributes = next => (item, opt) => (item.attributes ? ' ' + item.attributes.join(' ') : '') + next(item, opt)
const getValue = (item, opt) => opt[item.name] != null ? opt[item.name] : item.value
const getClassAttr = classes => classes && classes.length ? ` class="${classes.join(' ')}"` : ''
const getValueLabel = next => (item, opt) => item.text ? `<span class="opValue">${item.text}${next(item, opt)}</span>` : ''
const getOptionLabel = next => (item, opt) => next(item, opt) + (item.text ? `<span class="opValue">${item.text}</span>` : '')
const getDisabled = next => (item, opt) => (item.disabled ? ' disabled' : '') + next(item, opt)
const isChecked = opt => item => (!opt[item.name] && item.checked) || opt[item.name] === item.value || opt[item.name][item.value]
const getChecked = next => (item, opt) => (isChecked(opt)(item) ? ' checked' : '') + next(item, opt)
const getOptions = item =>
  item.options ? item.options.map(o => `<option value="${o.value}"${getDisabled(endPoint)(o)}>${o.value}</option>`).join('') : ''
const getInputTag = (next, getValue = item => item.value) => (item, opt) =>
  `<input type="${item.type}" name="${item.name}" value="${getValue(item, opt)}"
    ${getClassAttr(item.classes)}${next(item, opt)}>`
const getFileTag = next => (item, opt) =>
  `<input type="file" name="${item.name}" ${getClassAttr(item.classes)}${next(item, opt)}>`
const getSelectTag = next => (item, opt) =>
  `<select id="${item.name}" name="${item.name}" ${getClassAttr(item.classes)}${next(item, opt)}>${getOptions(item)}</select>`
const getTextAreaTag = (next, getValue = item => item.value) => (item, opt) =>
  `<textarea id="${item.name}" name="${item.name}" ${getClassAttr(item.classes)}${next(item, opt)}>${getValue(item, opt)}</textarea>`
const getInputView = next => (item, opt) => item.view ? item.view(next(item, opt)) : next(item, opt)
const getInputText = getInputView(getInputTag(getDisabled(getAttributes(endPoint)), getValue))
const getBindHTML = opt => item => getArgHTML(item, opt, false)
const getInputBinds = next => (item, opt) => next(item, opt) +
  (item.binds ? item.binds.map(getBindHTML(opt)).join('') : '')
const getInputCheckOption = reduceApply(endPoint, getAttributes, getDisabled, getChecked,
  getInputTag, getInputView, getOptionLabel, getInputBinds)
const getListElement = (item, opt) => {
  var listId = `list-${item.name}`, attr = `list="${listId}"`
  item.attributes ? item.attributes.push(attr) : (item.attributes = [attr])
  return [getInputText(item, opt)
    , `<datalist id="${listId}">`
    , getOptions(item)
    , '</datalist>'
    , getInputBinds(endPoint)(item, opt)].join('')
}
const getNoteHTML = text => `<li>${text}</li>`
const getNotes = item =>
  item.notes && item.notes.length ?
    ['<ul class="visible-md visible-lg description">', ...item.notes.map(getNoteHTML), '</ul>'].join('') : ''
const getArgHTML = (item, opt, hr = true) =>
  [(hr ? '<hr>' : ''),
  `<label class="${hr ? 'argName col-sm-2' : 'opValue'}" for="${item.name}">${item.text}</label>`,
  elementTypeMapping[item.type](item, opt),
  hr ? getNotes(item, opt) : ''].join('')
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
  file: getInputView(getFileTag(getDisabled(getAttributes(endPoint)))),
  button: getInputText,
  textarea: getInputView(getTextAreaTag(getDisabled(getAttributes(endPoint)))),
  select: getInputBinds(getInputView(getSelectTag(getDisabled(getAttributes(endPoint))))),
  list: getListElement
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
  if (panel.view) {
    opt = panel.view(opt)
  } else for (let key in opt)
    if (optionType[args[key].type]) {
      let f = args[key].type === 'radio' ? item => item.value.toString() === opt[key].toString()
        : item => opt[key][item.value.toString()]
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

const newStep = (panel, option) => ({ panel, opt: Object.assign({}, panel.initOpt, option) })
const context = (steps => {
  var pos = 0, index, addibleFeatures, tops, bottoms
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
    let ps = features.map(name => panels[name])
    bottoms = ps.filter(panel => panel.position < 0).sort(compareOp)
    tops = ps.filter(panel => panel.position >= 0).sort(compareOp)
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
  const loadStep = (panel, opt, target, file) => {
    let name = panel.op ? panel.op : panel.name
      , flag = opt && opt.op === name ? 1 : 0
    flag || (opt = {})
    file && (opt.file = file)
    delete opt.op
    target.push(newStep(panel, panel.load ? panel.load(opt) : opt))
    return flag
  }
  const loadOptions = options => {
    let files = steps.filter(step => step.panel.position != null)
      .map(step => step.opt ? step.opt.file : void 0)
    steps.splice(0, steps.length)
    var j = 0, k = options.length - 1, bottomSteps = []
    for (let i = 0; i < tops.length; i++)
      j += loadStep(tops[i], options[j], steps, files[i])
    files.splice(0, tops.length)
    for (let i = bottoms.length; i--; k -= loadStep(bottoms[i], options[k], bottomSteps, files[i]));
    for (; j <= k; j++)
      loadStep(panels[options[j].op], options[j], steps)
    steps.push(indexStep)
    for (let i = 0; i < bottomSteps.length; i++)
      steps.push(bottomSteps[i])
    refreshSteps()
  }
  const getFeatures = _ => addibleFeatures
  const refreshPanel = (refreshStep = 1) => {
    var step = steps[pos], panel = step.panel, opt = step.opt, args = panel.args
      , listeners = panel.listeners, options = $('#options')
    options.html(getPanelView(panel === panels.index ? getIndexHTML : getPanelInnerView(getArgsHTML))(panel, opt, pos))
    for (let argName in args)
      args[argName].type === 'file' && (options.find(`input[name=${argName}]`).get(0).files = opt[argName])
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
    refreshPanel, removeStep, refreshSteps, refreshCurrentStep, loadOptions
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

const serializeSteps = (toFile, data = new Map()) => steps
  .filter(step => step !== indexStep)
  .map(step => {
    let opt = Object.assign({}, step.opt), panel = step.panel
    if (panel.submit) {
      for (let key in panel.args) {
        let arg = panel.args[key]
        if ((arg.type === 'number' || arg.dataType === 'number') && opt[key] != null)
          opt[key] = +opt[key]
      }
      opt = panel.submit(opt, data)
    }
    opt && (opt.op = toFile == null && panel.op ? panel.op : panel.name)
    return opt
  }).filter(opt => !!opt)

const submit = data => data.set('steps', JSON.stringify(serializeSteps(null, data)))

export { addPanel, initListeners, submit, serializeSteps, context, getOptions }