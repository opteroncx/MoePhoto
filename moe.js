const { MoePhoto } = require('./static/api.js')
const fs = require('fs')
var port = 2333
try {
  port = +JSON.parse(fs.readFileSync('.user/config.json')).port[0]
} catch { }
MoePhoto('localhost', port)
  .process(process.argv[3], process.argv[2]) // filepath, preset
  .then(res => console.log(res))
  .catch(err => console.error(err))
