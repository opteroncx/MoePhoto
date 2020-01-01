import { texts } from './common.js'
const genDiagnoseArgs = () => {
  return [{
    type: 'select',
    text: texts.finish,
    classes: ['input-text', 'input-short']
  }]
}
export { genDiagnoseArgs }