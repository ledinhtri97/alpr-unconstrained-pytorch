from PIL import Image

class SquarePad(object):

    def __init__(self, fill=0, padding_mode='constant'):
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        w, h = img.size
        left = top = 0
        bottom = max(w - h, 0)
        right  = max(h - w, 0)
        padding = (left, top, right, bottom)
        
        new_size = max(w, h)
        result = Image.new(img.mode, (new_size, new_size), self.fill)
        result.paste(img, (left, top))
        
        return result
    
    
