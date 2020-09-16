import $ from 'jquery'
import { texts } from './common.js'
import { serializeSteps } from './steps.js'
const object = {},
  s = {},
  sf = { '*': {}, '=': {} },
  identityObject = {
    scaleW: 1.0,
    scaleH: 1.0,
    sf: 1.0,
    width: 0,
    height: 0,
    frameRate: 0,
    video: false
  },
  multiples = { scaleW: 'width', scaleH: 'height', sf: 'frameRate' },
  units = {
    width: 'pixel',
    height: 'pixel',
    srcWidth: 'pixel',
    srcHeight: 'pixel',
    frameRate: 'fps',
    fps: 'fps'
  },
  videoOnly = ['frameRate', 'fps'],
  textMapping = { mode: 'imageMode' }
const show = o => {
  let ot = {}
  for (let key in units) o[key] && (ot[key] = o[key] + texts[units[key]])
  for (let key in multiples) {
    let ko = multiples[key]
    o[ko] || (ot[ko] = o[key].toFixed(2) + texts.scale)
  }
  for (let key in textMapping)
    o[key] &&
      (ot[key] =
        o[key] in texts[textMapping[key]]
          ? texts[textMapping[key]][o[key]]
          : o[key])
  o.video ||
    videoOnly.forEach(key => {
      delete ot[key]
    })
  for (let key in ot) $(`#summary-${key}`).html(ot[key])
}
const toKey = (op, arg) => `${op}.${arg}`
const addSummary = (op, arg, option) => (s[toKey(op, arg)] = option)
const assign = k => (o, v) => v == null || (o[k] = +v)
const multiply = k => (o, v) => {
  if (v == null) return
  v = +v
  o[k] *= v
  o[multiples[k]] *= v
}
for (let k in multiples) {
  sf['*'][k] = multiply(k)
  sf['='][multiples[k]] = assign(multiples[k])
}
const initObeject = o => Object.assign(o, identityObject)
const summary = (acc, step) => {
  let op = step.op,
    option
  acc.video |= op === 'decode'
  for (let key in step)
    if ((option = s[toKey(op, key)])) {
      option.keys
        ? option.keys.forEach(k => sf[option.op][k](acc, step[key]))
        : sf[option] && sf[option][key](acc, step[key])
    }
  return acc
}
const onSummaryChange = _ =>
  show(serializeSteps().reduce(summary, initObeject(object)))
const onSummaryMessage = ({ data }) => {
  let change = false
  if (!data) return
  if (data.shape && data.shape.length > 1) {
    identityObject.height = identityObject.srcHeight = data.shape[0]
    identityObject.width = identityObject.srcWidth = data.shape[1]
    change = true
  }
  data.fps &&
    (change = true) &&
    (identityObject.frameRate = identityObject.fps = data.fps)
  data.mode && (change = true) && (identityObject.mode = data.mode)
  change && onSummaryChange()
}
export { onSummaryChange, onSummaryMessage, addSummary }
