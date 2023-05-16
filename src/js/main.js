import $ from 'jquery'
import { appendText, texts, getResource, urlParams } from './common.js'
import { addPanel, initListeners, submit, context } from './steps.js'
import { setup } from './progress.js'
import { genPresetArgs, presetNotesEditor } from './preset.js'
import {
  genDiagnoseArgs,
  onDiagnoseMessage,
  initDiagnoser
} from './diagnose.js'
import { onSummaryMessage } from './summary.js'

const None = () => void 0
const setAll = (arr, key) => values => arr.map((o, i) => (o[key] = values[i]))
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
  gan: [0, 1, 0, 1],
  gana: [1, 1, 0, 1]
}
const MPRNetNote = '来自<a href="https://github.com/swz30/MPRNet">Syed Waqas Zamir</a>'
const NAFNetNote = '来自<a href="https://github.com/megvii-research/NAFNet">旷视研究院</a>'
const AiLUTNote = '来自<a href="https://github.com/ImCharlesY/AdaInt">Yang Canqian</a>'
const ESRGANNote = '来自<a href="https://github.com/xinntao/Real-ESRGAN">Xintao Wang的Real-ESRGAN</a>'
const DehazeModelValues = [
  {
    value: 'dehaze',
    text: '去雾模型',
    hidden: true,
    func: 'dehaze'
  },
  {
    value: 'MPRNet_deblurring', text: 'MPRNet',
    notes: [MPRNetNote],
    func: 'deblur'
  },
  {
    value: 'NAFNet_deblur_32', text: 'NAFNet小',
    notes: [NAFNetNote],
    func: 'deblur'
  },
  {
    value: 'NAFNet_deblur_64', text: 'NAFNet大',
    checked: 1,
    notes: [NAFNetNote, '比小个的慢一点，据说效果也好一点'],
    func: 'deblur'
  },
  {
    value: 'NAFNet_deblur_JPEG_64', text: 'NAFNet_JPEG',
    notes: [NAFNetNote, '这个据说还有修复JPEG的技能'],
    func: 'deblur'
  },
  {
    value: 'MPRNet_deraining', text: 'MPRNet',
    notes: [MPRNetNote],
    hidden: true,
    func: 'derain'
  }
]
const RetouchModelValues = [
  {
    value: 'AiLUT_sRGB_3', text: 'AiLUT小',
    notes: [AiLUTNote],
    func: 'retouch'
  },
  {
    value: 'AiLUT_sRGB_5', text: 'AiLUT大',
    checked: 1,
    notes: [AiLUTNote, '和小个的相比都挺快的，效果也看不出差别来'],
    func: 'retouch'
  },
  {
    value: 'AiLUT_XYZ_3', text: 'AiLUT小',
    notes: [AiLUTNote, '相当不一样的色彩'],
    hidden: true,
    func: 'tone-remapping'
  }
]
const changeFuncs = Values => (_, opt) => {
  let models = []
  for (let item of Values)
    if (item.func === opt.func) {
      item.hidden = false
      models.push(item)
    } else {
      item.hidden = true
      item.checked = false
    }
  let i = models.findIndex(item => !!item.checked)
  if (i < 0) {
    i = models.length - 1
    models[i].checked = 1
  }
  opt.model = models[i].value
  return 1
}
var getResizeView = (by, scale, size) =>
  by === 'scale' ? scale + '倍' : appendText('pixel')(size)
const getFileName = opt => ({
  file: opt.file && opt.file[0] ? opt.file[0].name : texts.noFile
})
const submitFile = (opt, data) =>
  opt.file &&
  opt.file[0] &&
  (data.set('file', opt.file[0]) || (data.noCheckFile = 0)) &&
  void 0
const submitUrl = (opt, data) =>
  (data.set(opt.by, opt[opt.by]) || (data.noCheckFile = 1)) && void 0
const videoBy = (by, opt) => {
  return { [by]: opt[by] ? opt[by] : texts.noFile }
}
const [
  loadImagePreset,
  saveImagePreset,
  applyImagePresetButton,
  saveImagePresetButton,
  applyImagePreset
] = genPresetArgs('image')
const [diagnose] = genDiagnoseArgs()
const [
  loadVideoPreset,
  saveVideoPreset,
  applyVideoPresetButton,
  saveVideoPresetButton,
  applyVideoPreset
] = genPresetArgs('video')
const panels = {
  input: {
    text: '输入',
    description: '选择一张你需要放大的图片，开始体验吧！运行完毕请点击保存',
    position: 0,
    submit: submitFile,
    view: getFileName,
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
    submit: (opt, data) =>
      opt.by === 'file' ? submitFile(opt, data) : submitUrl(opt, data),
    view: opt => (opt.by === 'file' ? getFileName(opt) : videoBy(opt.by, opt)),
    args: {
      by: {
        type: 'radio',
        text: '来源',
        labelClasses: ['full-width'],
        values: [
          {
            value: 'file',
            binds: ['file'],
            classes: ['largeMargin'],
            checked: 1
          },
          { value: 'url', binds: ['url'], classes: ['largeMargin'] },
          { value: 'cmd', binds: ['cmd'], classes: ['largeMargin'] }
        ]
      },
      file: {
        type: 'file',
        name: 'file',
        text: '视频文件',
        classes: ['inputfile-6', 'imgInp'],
        attributes: ['required', 'accept="video/*,application/octet-stream"'],
        notes: ['输入若为视频文件则会复制一份上传，存放在程序的upload目录下']
      },
      url: {
        type: 'text',
        text: 'URL',
        value: '',
        classes: ['input-text', 'full-width'],
        attributes: ['required', 'spellcheck="false"']
      },
      cmd: {
        type: 'text',
        text: 'FFmpeg生成源',
        value: 'testsrc=size=1280x720:rate=10',
        classes: ['input-text', 'full-width'],
        attributes: ['required', 'spellcheck="false"']
      },
      preset: loadVideoPreset,
      apply: applyVideoPresetButton
    }
  },
  inputBatch: {
    text: '批量输入',
    description:
      '将所有需要放大的图片放置到一个文件夹内，并在下方选择路径；或者将多个图像文件一起拖放到下方',
    position: 0,
    submit: (opt, data) =>
      opt.file &&
      opt.file.length &&
      [...opt.file].forEach(
        f => data.append && data.append('file', f, f.name)
      ) &&
      void 0,
    view: opt => ({
      file:
        opt.file && opt.file.length
          ? [opt.file[0].name, '等', opt.file.length, '个'].join('')
          : '请选择'
    }),
    args: {
      file: {
        type: 'file',
        name: 'file',
        text: '文件',
        classes: ['inputfile-6', 'imgInp'],
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
        values: SRScaleValues,
        summary: { op: '*', keys: ['scaleW', 'scaleH'] }
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
              '快速模型能放大2、4或8倍，可以在后面添加“缩放”步骤配合使用'
            ]
          },
          {
            value: 'gan',
            text: 'GAN',
            notes: [
              'GAN模型仅适用于RGB图像，遇到带alpha通道的图像会出错错，反正我遇到的alpha通道都是多余的，用随便什么图片编辑去掉吧',
              'GAN模型能放大2或4倍，可以在后面添加“缩放”步骤配合使用',
              ESRGANNote
            ]
          },
          {
            value: 'gana',
            text: 'GAN动漫',
            notes: [
              '仅适用于RGB动漫图像，遇到带alpha通道的图像会出错错',
              '仅能放大4倍，但是比较快，可以在后面添加“缩放”步骤配合使用',
              ESRGANNote
            ]
          }
        ]
      },
      ensemble: {
        type: 'number',
        text: '额外的处理次数',
        value: 0,
        classes: ['input-number'],
        attributes: ['min="0"', 'max="7"', 'step="1"'],
        notes: [
          '让超分模型额外处理变换的图像，之后混合多次处理的结果，轻微提高质量，要花费时间以此处设置倍数增长',
          '可填0-7倍'
        ]
      }
    }
  },
  resize: {
    text: '缩放',
    description:
      '以插值方法缩放图像，对图像的宽高可以分别设定长度大小或者缩放比例',
    draggable: 1,
    submit: opt => {
      let res = { method: opt.method }
      opt.byW === 'scale' ? (res.scaleW = opt.scaleW) : (res.width = opt.width)
      opt.byH === 'scale'
        ? (res.scaleH = opt.scaleH)
        : (res.height = opt.height)
      return res
    },
    load: opt => {
      opt.byW = opt.scaleW != null ? 'scale' : 'pixel'
      opt.byH = opt.scaleH != null ? 'scale' : 'pixel'
      return opt
    },
    view: opt => {
      let res = {
        method: opt.method
      }
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
        ],
        summary: 1
      },
      scaleW: {
        type: 'number',
        text: '缩放比例',
        value: 1,
        classes: ['input-number'],
        attributes: ['min="0"', 'step="0.1"'],
        summary: '*'
      },
      width: {
        type: 'number',
        text: '大小',
        value: 1920,
        view: appendText('pixel'),
        classes: ['input-number'],
        attributes: ['min="1"', 'step="10"'],
        summary: '='
      },
      byH: {
        type: 'radio',
        text: '高度',
        values: [
          { value: 'scale', binds: ['scaleH'] },
          { value: 'pixel', binds: ['height'], checked: 1 }
        ],
        notes: ['按比例缩放图像长宽的小数部分四舍五入为整数'],
        summary: 1
      },
      scaleH: {
        type: 'number',
        text: '缩放比例',
        value: 1,
        classes: ['input-number'],
        attributes: ['min="0"', 'step="0.1"'],
        summary: '*'
      },
      height: {
        type: 'number',
        text: '大小',
        value: 1080,
        view: appendText('pixel'),
        classes: ['input-number'],
        attributes: ['min="1"', 'step="10"'],
        summary: '='
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
        text: '降噪模型',
        change: _ => 1,
        values: [
          { value: 'lite5', text: '弱' },
          { value: 'lite10', text: '中' },
          { value: 'lite15', text: '强' },
          {
            value: 'MPRNet_denoising',
            text: 'MPRNet',
            notes: [MPRNetNote]
          },
          {
            value: 'NAFNet_32',
            text: 'NAFNet小',
            notes: [NAFNetNote]
          },
          {
            value: 'NAFNet_64',
            text: 'NAFNet大',
            checked: 1,
            notes: [NAFNetNote, '比小个的慢一点，据说效果也好一点']
          },
          {
            value: 'VSR_Cleaning',
            text: '图像清理',
            notes: [
              '来自<a href="https://github.com/ckkelvinchan/RealBasicVSR">Kelvin C.K. Chan</a>',
              '原作者是把它放在视频放大模型的前面，说不定也能用在别的地方呢'
            ]
          }
        ]
      },
      strength: {
        type: 'number',
        text: '强度',
        value: 1.0,
        classes: ['input-number'],
        attributes: ['step="0.05"']
      }
    }
  },
  ednoise: {
    op: 'DN',
    text: '强力降噪',
    description:
      '这里的降噪非常强，涂抹效果显著，可以试试制作油画风格的照片或者galgame的背景图~',
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
      },
      strength: {
        type: 'number',
        text: '强度',
        value: 1.0,
        classes: ['input-number'],
        attributes: ['step="0.05"']
      }
    }
  },
  dehaze: {
    text: '去雨去雾去模糊',
    description: '三个模型分别能够去除静态画面中的雨滴、雾气和模糊效果',
    draggable: 1,
    args: {
      func: {
        type: 'radio',
        text: '功能',
        change: changeFuncs(DehazeModelValues),
        values: [
          { value: 'dehaze', text: '去雾' },
          { value: 'deblur', text: '去模糊', checked: 1 },
          { value: 'derain', text: '去雨' }
        ]
      },
      model: {
        type: 'radio',
        text: '模型',
        change: _ => 1,
        values: DehazeModelValues
      },
      strength: {
        type: 'number',
        text: '强度',
        value: 1.0,
        classes: ['input-number'],
        attributes: ['step="0.05"']
      }
    }
  },
  demoire: {
    op: 'dehaze',
    text: '去摩尔纹',
    description: '去除图像明暗相间的条纹缺陷',
    draggable: 1,
    args: {
      model: {
        type: 'radio',
        text: '模型',
        change: _ => 1,
        values: [
          {
            value: 'sun',
            text: '小模型',
            notes: ['小模型比较节约资源']
          },
          {
            value: 'moire_obj',
            text: '自然模型',
            checked: 1,
            notes: ['自然模型比较擅长保留对象的纹理，也比较占资源']
          },
          {
            value: 'moire_screen_gan',
            text: '屏幕模型',
            notes: ['屏幕模型比较强力地抹掉摩尔纹']
          }
        ]
      },
      strength: {
        type: 'number',
        text: '强度',
        value: 1.0,
        classes: ['input-number'],
        attributes: ['step="0.05"']
      }
    }
  },
  retouch: {
    op: 'dehaze',
    text: '美化调色',
    description: '让贫乏的图像变得更多彩一点点',
    draggable: 1,
    args: {
      func: {
        type: 'radio',
        text: '功能',
        change: changeFuncs(RetouchModelValues),
        values: [
          { value: 'retouch', text: '色彩美化', checked: 1 },
          { value: 'tone-remapping', text: '重排色调' }
        ]
      },
      model: {
        type: 'radio',
        text: '模型',
        change: _ => 1,
        values: RetouchModelValues
      },
      strength: {
        type: 'number',
        text: '强度',
        value: 0.75,
        classes: ['input-number'],
        attributes: ['step="0.05"']
      }
    }
  },
  VSR: {
    text: '视频放大',
    description: [
      '专用于视频的4倍放大，比单张图片的放大快多了，效果说不定还好一点',
      '来自于<a href="https://github.com/xinntao/BasicSR/blob/master/basicsr/archs/basicvsr_arch.py">Xintao Wang的IconVSR</a>'
    ].join('<br />'),
    draggable: 1,
    args: {
      _scale: {
        value: 4,
        summary: { op: '*', keys: ['scaleW', 'scaleH'] }
      }
    }
  },
  demob: {
    text: '清晰动作',
    description: '消除运动模糊，让运动物体清晰起来，是个快乎乎的模型',
    draggable: 1,
    args: {
      model: {
        type: 'radio',
        text: '时间区间',
        values: [
          {
            value: '1ms8ms',
            text: '1到8毫秒'
          },
          {
            value: '2ms16ms',
            text: '2到16毫秒',
            checked: 1
          },
          {
            value: '3ms24ms',
            text: '3到24毫秒'
          }
        ],
        notes: ['来自于<a href="https://github.com/zzh-tech/ESTRNN">Zhihang Zhong的ESTRNN</a>']
      }
    }
  },
  decode: {
    text: '输入解码',
    description:
      '传入ffmpeg的解码参数设定，跟命令行使用是一样的，默认不做额外的解码处理',
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
        attributes: ['min="1"', 'step="10"'],
        summary: '='
      },
      height: {
        type: 'number',
        text: '覆盖输入高度',
        value: 0,
        view: appendText('pixel'),
        classes: ['input-number'],
        attributes: ['min="1"', 'step="10"'],
        notes: [
          '我们从输入视频文件里获取画面大小信息，如果这里的解码处理改变了画面大小，请设置上面的两个覆盖值从而告诉后面的处理过程，否则就不要动它们啦'
        ],
        summary: '='
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
    description:
      '传入ffmpeg的编码参数设定，跟命令行使用是一样的，这里通常指定常见的编码器和颜色格式',
    position: -1,
    submit: copyTruly,
    view: copyTruly,
    args: {
      codec: {
        type: 'text',
        text: '编码参数',
        value:
          'libx264 -crf 17 -bf 10 -refs 12 -coder ac -cmp chroma -profile:v high -level 51 -g 720 -keyint_min 20 -psy 1 -weightb 1 -weightp 2 -mbtree 1 -threads 0 -pix_fmt yuv420p',
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
        ],
        summary: '='
      },
      diagnose,
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
          '为了方便精确地拼接输出视频，若之前的视频处理开始于设定大于0且这里设置了大于1的倍数，那么开始帧之前的那一帧会被用作参考帧，输出的头几帧将会是它与开始帧之间的插入帧，但这一参考帧本身将不会被输出',
          '来自于<a href="https://github.com/avinashpaliwal/Super-SloMo">avinashpaliwal的Super-SloMo</a>'
        ],
        summary: '*'
      }
    }
  }
}

const setupMain = opt => {
  opt.progress = $('#progress')
  opt.beforeSend = submit
  opt.context = context
  for (let key of opt.features) if (key !== 'index') addPanel(key, panels[key])
  let progress = setup(opt)
  initListeners()
  context.setFeatures(opt.features)
  if (opt.features.length > 2) {
    progress
      .on('message', onDiagnoseMessage)
      .on('message', onSummaryMessage)
      .on('open', onSummaryMessage)
    initDiagnoser(panels, document.getElementById('progress'))
  }
  let isVideo = opt.features.indexOf('inputVideo') > -1,
    applyPreset = isVideo ? applyVideoPreset : applyImagePreset,
    preset = urlParams.get('preset')
  preset && applyPreset(preset)
}
const exportApp = { setup: setupMain, texts, getResource }
if (window.app) Object.assign(window.app, exportApp)
else window.app = exportApp
