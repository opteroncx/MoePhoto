import { texts, urlParams } from './common.js'
import { context } from './steps.js'
const ops = {
  SR: ['model', 'scale'],
  DN: ['model'],
  resize: ['mode'],
  dehaze: [],
  sun: [],
  mddm: [],
  slomo: []
}
const weights = {
  resize: 3e-7,
  SR: 3e-5,
  DN: 2e-5,
  slomo: 15e-6,
  dehaze: 3e-4
}
const names = {}
const diag = { d: new Map() }
const joinByKeys = (o, keys) =>
  keys.length ? ':' + keys.map(key => o[key]).join(', ') : ''
const restrictLength = str => str.slice(0, 32)

const genDiagnoseArgs = () => {
  return [
    {
      type: 'checkbox',
      text: texts.diagnose,
      classes: ['input-text', 'input-short'],
      values: [
        { value: 'bench', text: texts.bench },
        { value: 'clear', text: texts.clearMetric }
      ]
    }
  ]
}

const initDiagnoser = (panels, ele) => {
  if (!ele) return
  for (let k in panels) names[k] = panels[k].text
  panels.demoire.args.model.values.forEach(
    v => (names[v.value] = panels.demoire.text + v.text)
  )
  let by = urlParams.get('by')
  if (by) {
    context.getOpt(0).by = by
    context.refreshSteps()
  }
  diag.t = document.getElementById('benchmark')
  if (!diag.t) {
    let table = document.createElement('table')
    diag.t = document.createElement('tbody')
    diag.t.setAttribute('id', 'benchmark')
    diag.t.hidden = true
    diag.t.innerHTML = `<th>${texts.item}</th><th>${texts.samples}</th><th>${texts.mark}</th>`
    table.appendChild(diag.t)
    ele.appendChild(document.createElement('hr'))
    ele.appendChild(table)
  }
  return ele
}

const newItem = (op, samples, mark) => {
  let item = setItem(document.createElement('tr'), op, samples, mark)
  diag.d.set(op, item)
  return item
}

const setItem = (tr, op, samples, mark) => {
  tr.innerHTML = `<td>${op}</td><td>${samples}</td><td>${mark.toFixed(3)}</td>`
  return tr
}

const showBench = (op, weight, samples) => {
  if (!op || !ops[op.op]) return
  let mark = (weights[op.op] || 1e-3) / weight
  op = restrictLength(names[op.op] + joinByKeys(op, ops[op.op]))
  diag.d.has(op)
    ? setItem(diag.d.get(op), op, samples, mark)
    : diag.t.appendChild(newItem(op, samples, mark))
  diag.t.hidden = false
}

const onDiagnoseMessage = ({ data }) =>
  data && data.op && showBench(data.op, data.weight, data.samples)
export { genDiagnoseArgs, onDiagnoseMessage, initDiagnoser }
