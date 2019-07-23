import $ from 'jquery'
import { appendText, texts, getResource } from './common.js'
import { addPanel, initListeners, submit, context } from './steps.js'
import { setup } from './progress.js'
import { genPresetArgs, presetNotesEditor } from './preset.js'

const None = () => void 0
const setAll = (arr, key) => values => arr.map((o, i) => o[key] = values[i])
const copyTruly = obj => {
  let res = {}
  for (let key in obj) obj[key] && (res[key] = obj[key])
  return res
}
var SRScaleValues = [
  { value: 2, text: '2倍', checked: 1 },
  { value: 3, text: '3倍' },
  { value: 4, text: '4倍' },
  { value: 8, text: '8倍', disabled: 1 }
]
const setScaleDisabled = setAll(SRScaleValues, 'disabled')
const scaleModelMapping = {
  a: [0, 0, 0, 1],
  p: [0, 0, 0, 1],
  lite: [0, 1, 0, 0],
  gan: [1, 1, 0, 1]
}
var getResizeView = (by, scale, size) =>
  by === 'scale' ? scale + '倍' : appendText('pixel')(size)
const setFile = opt => ({ file: opt.file && opt.file[0] ? opt.file[0].name : '请选择' })
const submitFile = (opt, data) => opt.file && opt.file[0] && data.set('file', opt.file[0]) && void 0
const [loadImagePreset, saveImagePreset, applyImagePresetButton, saveImagePresetButton] = genPresetArgs('image')
const [loadVideoPreset, saveVideoPreset, applyVideoPresetButton, saveVideoPresetButton] = genPresetArgs('video')
const panels = {
  input: {
    text: '输入',
    description: '选择一张你需要放大的图片，开始体验吧！运行完毕请点击保存',
    position: 0,
    submit: submitFile,
    view: opt => ({ file: opt.file && opt.file[0] ? opt.file[0].name : '请选择' }),
    args: {
      file: {
        type: 'file',
        name: 'file',
        text: '图片',
        classes: ['inputfile-6', 'imgInp'],
        attributes: ['required', 'accept="image/*"']
      },
      preset: loadImagePreset,
      apply: applyImagePresetButton
    }
  },
  inputVideo: {
    text: '输入',
    description: '选择一段需要放大的视频！运行完毕请点击保存',
    position: 0,
    submit: submitFile,
    view: setFile,
    args: {
      file: {
        type: 'file',
        name: 'file',
        text: '视频',
        classes: ['inputfile-6', 'imgInp'],
        attributes: ['required', 'accept="video/*,application/octet-stream"'],
        notes: [
          '视频会复制一份上传，存放在程序的upload目录下'
        ]
      },
      preset: loadVideoPreset,
      apply: applyVideoPresetButton
    }
  },
  inputBatch: {
    text: '批量输入',
    description: '将所有需要放大的图片放置到一个文件夹内，并在下方选择路径',
    position: 0,
    submit: (opt, data) => opt.file && opt.file.length && [...opt.file].forEach(f => data.append && data.append('file', f, f.name)) && void 0,
    view: opt => ({ file: opt.file && opt.file.length ? [opt.file[0].name, '等', opt.file.length, '个'].join('') : '请选择' }),
    args: {
      file: {
        type: 'file',
        name: 'file',
        text: '文件',
        classes: ['inputfile-6'],
        attributes: ['required', 'webkitdirectory', 'directory']
      },
      preset: loadImagePreset,
      apply: applyImagePresetButton
    }
  },
  output: {
    text: '输出',
    description: '保存预置',
    position: -1,
    submit: None,
    view: () => '',
    args: {
      preset: saveImagePreset,
      notes: presetNotesEditor,
      savePreset: saveImagePresetButton
    }
  },
  SR: {
    text: '超分辨率',
    description: '以2、3、4甚至8倍整数比例放大图像',
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
          let i = SRScaleValues.findIndex(item => item.value === opt.scale)
          for (; SRScaleValues[i].disabled; i = (i + 1) % SRScaleValues.length);
          opt.scale = SRScaleValues[i].value
          return 1
        },
        values: [
          { value: 'a', text: '动漫', checked: 1 },
          { value: 'p', text: '照片' },
          {
            value: 'lite',
            text: '快速',
            notes: [
              '快速模型仅能放大2、4、8倍，可以在后面添加“缩放”步骤配合使用'
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
    load: opt => {
      opt.byW = opt.scaleW != null ? 'scale' : 'pixel'
      opt.byH = opt.scaleH != null ? 'scale' : 'pixel'
      return opt
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
        classes: ['input-number'],
        attributes: ['min="0"', 'step="0.1"']
      },
      width: {
        type: 'number',
        text: '大小',
        value: 1920,
        view: appendText('pixel'),
        classes: ['input-number'],
        attributes: ['min="1"', 'step="10"']
      },
      byH: {
        type: 'radio',
        text: '高度',
        values: [
          { value: 'scale', binds: ['scaleH'] },
          { value: 'pixel', binds: ['height'], checked: 1 }
        ],
        notes: ['按比例缩放图像长宽的小数部分四舍五入为整数']
      },
      scaleH: {
        type: 'number',
        text: '缩放比例',
        value: 1,
        classes: ['input-number'],
        attributes: ['min="0"', 'step="0.1"']
      },
      height: {
        type: 'number',
        text: '大小',
        value: 1080,
        view: appendText('pixel'),
        classes: ['input-number'],
        attributes: ['min="1"', 'step="10"']
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
        attributes: ['spellcheck="false"'],
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
        classes: ['input-number'],
        attributes: ['min="1"', 'step="10"']
      },
      height: {
        type: 'number',
        text: '覆盖输入高度',
        value: 0,
        view: appendText('pixel'),
        classes: ['input-number'],
        attributes: ['min="1"', 'step="10"'],
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
        attributes: ['min="0"'],
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
        attributes: ['min="0"'],
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
        value: 'libx264 -crf 17 -bf 10 -refs 12 -coder ac -cmp chroma -profile:v high -level 51 -g 720 -keyint_min 20 -psy 1 -weightb 1 -weightp 2 -mbtree 1 -threads 0 -pix_fmt yuv420p',
        classes: ['input-text'],
        attributes: ['spellcheck="false"'],
        notes: [
          '传入ffmpeg的编码参数设定，默认设定兼容性最好但是效率一般',
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
        attributes: ['min="1"'],
        notes: [
          '默认的输出帧率就是输入乘上插帧倍数，这样保持时间长度与输入一致，如果您想要看慢动作什么的可以在这里设定',
          '注意我们只处理视频画面部分，设置了这个之后声音字幕什么的通常就对不上了'
        ]
      },
      preset: saveVideoPreset,
      notes: presetNotesEditor,
      savePreset: saveVideoPresetButton
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
        attributes: ['min="1"'],
        notes: [
          '输出帧数是输入的多少倍，必须是正整数',
          '为了方便精确地拼接输出视频，若之前的视频处理开始于设定大于0且这里设置了大于1的倍数，那么开始帧之前的那一帧会被用作参考帧，输出的头几帧将会是它与开始帧之间的插入帧，但这一参考帧本身将不会被输出'
        ]
      }
    }
  }
}

const setupMain = opt => {
  opt.progress = $('#progress')
  opt.beforeSend = submit
  for (let key of opt.features) if (key !== 'index') addPanel(key, panels[key])
  setup(opt)
  initListeners()
  context.setFeatures(opt.features)
}
const exportApp = { setup: setupMain, texts, getResource }
if (window.app)
  Object.assign(window.app, exportApp)
else
  window.app = exportApp