# pylint: disable=E1101
import torch

def getGPUProperty(i):
  prop = torch.cuda.get_device_properties(i)
  return {
    'name': prop.name,
    'capability': prop.major + prop.minor / 10,
    'total_memory': prop.total_memory // 2**20,
    'processor_count': prop.multi_processor_count
  }

if torch.cuda.is_available():
  j = torch.cuda.device_count()
  i = 0
  while j and i < 4:
    try:
      print('GPU #{}'.format(i))
      print(getGPUProperty(i))
      print(torch.ones((2, 2), device=torch.device('cuda:{}'.format(i))).log().sum())
      j -= 1
    except Exception as e:
      print('GPU #{}'.format(i), e)
    finally:
      i += 1
else:
  print('CUDA isn''t available.')