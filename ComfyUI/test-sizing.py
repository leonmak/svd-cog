import unittest
from PIL import Image

from sizing_strategy import SizingStrategy


test_file = 'ComfyUI/islands.png'
class TestResize(unittest.TestCase):
  
  def test_resizeSVD(self):
    strat = SizingStrategy()
    res = strat.apply(sizing_strategy='maintain_aspect_ratio', 
                            image=test_file)
    img, w, h = res
    img.save('islands-resize.png')

  def test_resizeSVDs(self):
    strat = SizingStrategy()
    res = strat.apply(sizing_strategy='crop_to_16_9', 
                      image=test_file)
    img, w, h = res
    img.save('islands-crop.png')

    
if __name__ == '__main__':
  unittest.main()