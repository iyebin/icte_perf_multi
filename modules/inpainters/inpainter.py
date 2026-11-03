class Inpainter:
    def __init__(self):
        pass

    def inpaint(self, img, mask, exp_dir):
        '''
        :param img:
        :param mask:
        :param exp_dir: 
        :return: img
        '''
        raise NotImplementedError

    def inpaint_rgbd(self, img, distance, mask):
        raise NotImplementedError

    def encode(self, img, mask):
        '''
        :param img: [B, 3, H, W]
        :param mask: [B, 1, H, W]
        :return: z
        '''
        raise NotImplementedError

