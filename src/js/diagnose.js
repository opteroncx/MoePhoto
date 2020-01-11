import { texts } from './common.js'
const ops = {
  SR: ['model', 'scale'],
  DN: ['model'],
  dehaze: [],
  sun: [],
  encode: ['encodec'],
  slomo: []
}
const weights = { encode: 100 }
const names = {}
const context = { d: new Map() }
const joinByKeys = (o, keys) =>
  keys.length ? ':' + keys.map(key => o[key]).join(', ') : ''
const restrictLength = str => str.slice(0, 32)

const genDiagnoseArgs = () => {
  return [
    {
      type: 'checkbox',
      text: texts.diagnose,
      classes: ['input-text', 'input-short'],
      values: [{ value: 'bench', text: texts.bench }]
    }
  ]
}

const initDiagnoser = (panels, ele) => {
  if (!ele) return
  for (let k in panels) names[k] = panels[k].text
  panels.demoire.args.model.values.forEach(
    v => (names[v.value] = panels.demoire.text + v.text)
  )
  context.t = document.getElementById('benchmark')
  if (!context.t) {
    context.t = document.createElement('table')
    context.t.setAttribute('id', 'benchmark')
    ele.appendChild(context.t)
  }
  context.t.hidden = true
  context.t.innerHTML = `<th><td>${texts.item}</td><td>${texts.samples}</td><td>${texts.mark}</td></th>`
  return ele
}

const newItem = (op, samples, mark) =>
  context.d
    .set(op, setItem(document.createElement('tr'), op, samples, mark))
    .get(op)

const setItem = (tr, op, samples, mark) =>
  (tr.innerHTML = `<td>${op}</td><td>${samples}</td><td>${mark}</td>`) && tr

const showBench = (op, weight, samples) => {
  if (!op || !ops[op.op]) return
  let mark = (weights[op.op] || 1e-3) / weight
  op = restrictLength(names[op.op] + joinByKeys(op, ops[op.op]))
  context.d.has(op)
    ? setItem(context.d.get(op), op, samples, mark)
    : context.t.appendChild(newItem(op, samples, mark))
  context.t.hidden = false
}

const onDiagnoseMessage = ({ data }) =>
  data && data.op && showBench(data.op, data.weight, data.samples)
export { genDiagnoseArgs, onDiagnoseMessage, initDiagnoser }
