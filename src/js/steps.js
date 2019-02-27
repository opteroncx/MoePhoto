import $ from 'jquery'
import { texts, appendText, registryLanguageListener } from './common.js'

const setAll = (arr, key) => values => arr.map((o, i) => o[key] = values[i])
const copyTruly = obj => {
  let res = {}
  for (let key in obj) obj[key] && (res[key] = obj[key])
  return res
}
var SRScaleValues = [
  { value: 2, text: '2倍', checked: 1 },
  { value: 3, text: '3倍' },
  { value: 4, text: '4倍' }
]
const setScaleDisabled = setAll(SRScaleValues, 'disabled')
const scaleModelMapping = {
  a: [0, 0, 0],
  p: [0, 0, 0],
  lite: [0, 1, 0],
  gan: [1, 1, 0]
}
var getResizeView = (by, scale, size) =>
  by === 'scale' ? scale + '倍' : appendText('pixel')(size)
const panels = {
  index: {
    text: '请选择一项功能',
    draggable: 1
  },
  input: {
    text: '输入',
    description: '选择一张你需要放大的图片，开始体验吧！运行完毕请点击保存',
    position: 0,
    submit: (opt, data) => opt.file && opt.file[0] && data.set('file', opt.file[0]) && void 0,
    view: opt => ({ file: opt.file && opt.file[0] ? opt.file[0].name : '请选择' }),
    args: {
      file: {
        type: 'file',
        name: 'file',
        text: '图片',
        classes: ['inputfile-6', 'imgInp']
      }
    }
  },
  inputVideo: {
    text: '输入',
    description: '选择一段需要放大的视频！运行完毕请点击保存',
    position: 0,
    submit: (opt, data) => opt.file && opt.file[0] && data.set('file', opt.file[0]) && void 0,
    view: opt => ({ file: opt.file && opt.file[0] ? opt.file[0].name : '请选择' }),
    args: {
      file: {
        type: 'file',
        name: 'file',
        text: '视频',
        classes: ['inputfile-6'],
        notes: [
          '视频会复制一份上传，存放在程序的upload目录下'
        ]
      }
    }
  },
  inputBatch: {
    text: '批量输入',
    description: '将所有需要放大的图片放置到一个文件夹内，并在下方选择路径',
    position: 0,
    submit: (opt, data) => opt.file && opt.file.length && data.set('file', opt.file) && void 0,
    view: opt => ({ file: opt.file && opt.file.length ? [opt.file[0].name, '等', opt.file.length, '个'].join('') : '请选择' }),
    args: {
      file: {
        type: 'file',
        name: 'file',
        text: '文件',
        classes: ['inputfile-6'],
        attributes: ['webkitdirectory', 'directory']
      }
    }
  },
  SR: {
    text: '超分辨率',
    description: '以2、3、4倍整数比例放大图像',
    draggable: 1,
    args: {
      scale: {
        type: 'radio',
        text: '放大倍数',
        values: SRScaleValues
      },
      model: {
        type: 'radio',
        text: '超分模型',
        change: (_, opt) => {
          setScaleDisabled(scaleModelMapping[opt.model])
          if (SRScaleValues[opt.scale - 2].disabled)
            for (opt.scale = 2; SRScaleValues[opt.scale - 2].disabled; opt.scale++);
          return 1
        },
        values: [
          { value: 'a', text: '动漫', checked: 1 },
          { value: 'p', text: '照片' },
          {
            value: 'lite',
            text: '快速',
            notes: [
              '快速模型仅能放大2倍或4倍，可以在后面添加“缩放”步骤配合使用'
            ]
          },
          {
            value: 'gan',
            text: 'GAN',
            notes: [
              'GAN模型仅适用于RGB图像',
              'GAN模型仅能放大4倍，可以在后面添加“缩放”步骤配合使用'
            ]
          }
        ]
      }
    }
  },
  resize: {
    text: '缩放',
    description: '以插值方法缩放图像，对图像的宽高可以分别设定长度大小或者缩放比例',
    draggable: 1,
    submit: opt => {
      let res = { method: opt.method }
      opt.byW === 'scale' ? res.scaleW = opt.scaleW : res.width = opt.width
      opt.byH === 'scale' ? res.scaleH = opt.scaleH : res.height = opt.height
      return res
    },
    view: opt => {
      let res = { method: panels.resize.args.method.values.find(item => item.value === opt.method).text }
      res.byW = getResizeView(opt.byW, opt.scaleW, opt.width)
      res.byH = getResizeView(opt.byH, opt.scaleH, opt.height)
      return res
    },
    args: {
      method: {
        type: 'radio',
        text: '插值方法',
        values: [
          { value: 'nearest', text: '最近邻' },
          { value: 'bilinear', text: '双线性', checked: 1 }
        ]
      },
      byW: {
        type: 'radio',
        text: '宽度',
        values: [
          { value: 'scale', binds: ['scaleW'] },
          { value: 'pixel', binds: ['width'], checked: 1 }
        ]
      },
      scaleW: {
        type: 'number',
        text: '缩放比例',
        value: 1,
        classes: ['input-number']
      },
      width: {
        type: 'number',
        text: '大小',
        value: 1920,
        view: appendText('pixel'),
        classes: ['input-number']
      },
      byH: {
        type: 'radio',
        text: '高度',
        values: [
          { value: 'scale', binds: ['scaleH'] },
          { value: 'pixel', binds: ['height'], checked: 1 }
        ]
      },
      scaleH: {
        type: 'number',
        text: '缩放比例',
        value: 1,
        classes: ['input-number']
      },
      height: {
        type: 'number',
        text: '大小',
        value: 1080,
        view: appendText('pixel'),
        classes: ['input-number']
      }
    }
  },
  DN: {
    text: '降噪',
    description: '相对较快的轻量级降噪',
    draggable: 1,
    args: {
      model: {
        type: 'radio',
        text: '降噪强度',
        values: [
          { value: 'lite5', text: '弱', checked: 1 },
          { value: 'lite10', text: '中' },
          { value: 'lite15', text: '强' }
        ]
      }
    }
  },
  ednoise: {
    op: 'DN',
    text: '强力降噪',
    description: '这里的降噪非常强，涂抹效果显著，可以试试制作油画风格的照片或者galgame的背景图~',
    draggable: 1,
    args: {
      model: {
        type: 'radio',
        text: '级别',
        values: [
          { value: 'lite5', text: '弱', checked: 1 },
          { value: 'lite10', text: '中' },
          { value: 'lite15', text: '强' }
        ]
      }
    }
  },
  dehaze: {
    text: '去雾',
    description: '去雾AODnet',
    draggable: 1
  },
  decode: {
    text: '输入解码',
    description: '传入ffmpeg的解码参数设定，跟命令行使用是一样的，默认不做额外的解码处理',
    position: 1,
    submit: copyTruly,
    view: copyTruly,
    args: {
      codec: {
        type: 'text',
        text: '解码参数',
        value: '',
        classes: ['input-text'],
        notes: [
          '请不要在这里设置颜色格式',
          '注意无论下一步的开始于设定为多少，解码都是从视频头开始的'
        ]
      },
      width: {
        type: 'number',
        text: '覆盖输入宽度',
        value: 0,
        view: appendText('pixel'),
        classes: ['input-number']
      },
      height: {
        type: 'number',
        text: '覆盖输入高度',
        value: 0,
        view: appendText('pixel'),
        classes: ['input-number'],
        notes: ['我们从输入视频文件里获取画面大小信息，如果这里的解码处理改变了画面大小，请设置上面的两个覆盖值从而告诉后面的处理过程，否则就不要动它们啦']
      }
    }
  },
  range: {
    text: '处理范围',
    description: '如果不需要处理整段视频，可以在这里设置',
    position: 2,
    submit: copyTruly,
    view: copyTruly,
    args: {
      start: {
        type: 'number',
        text: '开始于',
        value: 0,
        view: html => `第${html}帧`,
        classes: ['input-number'],
        notes: [
          '处理范围指的是经过上一步解码处理后的第几帧到第几帧，首帧为第0帧',
          '如果想继续上次中止掉的视频处理，上次完成到第几帧这次这里就填多少，其他配置保持一致，最后的多个输出可以用其他工具直接无损拼接',
          '注意只有视频画面才有帧的概念，若开始大于0，则除画面外的其余轨道将会被忽略，否则其余轨道将会被直接完整复制到输出中'
        ]
      },
      stop: {
        type: 'number',
        text: '结束于',
        value: 0,
        view: html => `第${html}帧`,
        classes: ['input-number'],
        notes: [
          '输出不包括上述编号的帧，即输出帧数为结束减开始，若该值小于等于开始则忽略并处理到视频结尾'
        ]
      }
    }
  },
  encode: {
    text: '输出编码',
    description: '传入ffmpeg的编码参数设定，跟命令行使用是一样的，这里通常指定常见的编码器和颜色格式',
    position: -1,
    submit: copyTruly,
    view: copyTruly,
    args: {
      codec: {
        type: 'text',
        text: '编码参数',
        value: 'libx264 -pix_fmt yuv420p',
        classes: ['input-text'],
        notes: [
          '传入ffmpeg的编码参数设定，默认设定兼容性最好但是质量和效率都普通',
          '一定要以编码器名开头，也一定要指定输出颜色格式（-pix_fmt <格式名>）',
          '输出视频以mkv格式封装，因为这啥都能装，有许多免费工具能够无损转换封装格式，我们就不做这事了'
        ]
      },
      frameRate: {
        type: 'number',
        text: '覆盖输出帧率',
        value: 0,
        view: appendText('fps'),
        classes: ['input-number'],
        notes: [
          '默认的输出帧率就是输入乘上插帧倍数，这样保持时间长度与输入一致，如果您想要看慢动作什么的可以在这里设定',
          '注意我们只处理视频画面部分，设置了这个之后声音字幕什么的通常就对不上了'
        ]
      }
    }
  },
  slomo: {
    text: '插帧',
    description: '以可设定的整倍数填充视频画面帧',
    draggable: 1,
    args: {
      sf: {
        type: 'number',
        text: '倍数',
        value: 1,
        classes: ['input-number'],
        notes: [
          '输出帧数是输入的多少倍，必须是正整数',
          '为了方便精确地拼接输出视频，若之前的视频处理开始于设定大于0且这里设置了大于1的倍数，那么开始帧之前的那一帧会被用作参考帧，输出的头几帧将会是它与开始帧之间的插入帧，但这一参考帧本身将不会被输出'
        ]
      }
    }
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
    var arr = ['index'].concat(features)
    addibleFeatures = features.filter(name => panels[name].draggable)
    let ps = arr.map(name => panels[name]),
      tops = ps.filter(panel => panel.position >= 0).sort(compareOp),
      bottoms = ps.filter(panel => panel.position < 0).sort(compareOp)
    tops.forEach(pushNewStep)
    index = steps.length
    steps.push(indexStep)
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

const app = { addPanel, initListeners, submit, context }
if (window.app)
  Object.assign(window.app, app)
else
  window.app = app
export default app