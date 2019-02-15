import './common.js'
import $ from 'jquery'
/*
support_doc ='<div class="col-md-3 col-xs-6 team-grids"><div class="thumbnail team-agileits">\
  <img src="%s" class="img-responsive" alt="" /><div class="w3agile-caption ">\
  <h4>%s</h4><div class="social-icon social-w3lsicon">\
  <a href="%s" class="social-button drb1">\
	<i class="fa fa-home"></i></a></div></div></div></div>'
support_row_doc = '<div class="team-row-agileinfo">'
def about_supporter():
  info_doc = codecs.open('static/supporter.json',encoding='utf-8').read()
  info_doc = json.loads(info_doc)
  show_doc = support_row_doc
  counter = 0
  for k in info_doc.keys():
    show_doc += support_doc%('static/savatar/'+k+'.jpg',k,info_doc[k])
    counter += 1
    if counter%4 ==0:
      counter = 0
      show_doc += '</div>'
      show_doc += support_row_doc
  return show_doc
*/
const prefix = 'http://may-workshop.com/moephoto/'
const jsonUrl = `${prefix}supporter.json`
const supportView = (img, name, link) => `<div class="col-md-3 col-xs-6 team-grids"><div class="thumbnail team-agileits">
<img src="${prefix}${img}" class="img-responsive" alt="${name}" /><div class="w3agile-caption ">
<h4>${name}</h4><div class="social-icon social-w3lsicon">
<a href="${link}" class="social-button drb1">
<i class="fa fa-home"></i></a></div></div></div></div>`
const supportRowStart = '<div class="team-row-agileinfo">', supportRowEnd = '</div>'
const supportRowView = items => [supportRowStart, ...items.map(item => supportView(item.img, item.name, item.link)), supportRowEnd].join('')
const getView = obj => {
  var items = [], counter = 0, res = [], v
  const pushRow = _ => {
    res.push(supportRowView(items))
    items = []
  }
  for (let k in obj) {
    v = obj[k]
    items.push({ name: k, img: v[1], link: v[0] });
    (++counter % 4 === 0) && pushRow()
  }
  pushRow()
  return res.join('')
}
$(document).ready(_ => {
  const supporters = $('#supporters')
  $.getJSON(jsonUrl)
    .then(getView)
    .then(supporters.html.bind(supporters))
    .catch(console.error.bind(console))
})