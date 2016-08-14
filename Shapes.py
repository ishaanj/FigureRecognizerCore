import numpy as np
import scipy.ndimage as ndi
import cv2
from scipy.misc import imsave, imresize


class Shape:

    def __init__(self, name):
        self.height = 224
        self.width = 224
        self.channels = 3

        self.name = name
        self.channel_index = 0
        self.row_index = 1
        self.col_index = 2

    def getShapeClass(self):
        return self.name

    def createShape(self, height_shift=0, width_shift=0, rotation=None, shear=None):
        pass

    def _checkPoints(self, pt1, pt2):
        minimum = 16
        maximum = 200

        pt1[0] = pt1[0] if pt1[0] > minimum else minimum
        pt1[1] = pt1[1] if pt1[1] > minimum else minimum
        pt2[0] = pt2[0] if pt2[0] > minimum else minimum
        pt2[1] = pt2[1] if pt2[1] > minimum else minimum

        pt1[0] = pt1[0] if pt1[0] < maximum else maximum
        pt1[1] = pt1[1] if pt1[1] < maximum else maximum
        pt2[0] = pt2[0] if pt2[0] < maximum else maximum
        pt2[1] = pt2[1] if pt2[1] < maximum else maximum

        return tuple(pt1), tuple(pt2)


    def __transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix

    def __apply_transform(self, x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
        x = np.rollaxis(x, channel_index, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]
        channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                                                             final_offset, order=0, mode=fill_mode, cval=cval) for
                          x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_index + 1)
        return x

    def _transform(self, x, rotation=None, height_shift=None, width_shift=None, shear=None):
        x = x.transpose((2, 0, 1))

        rotation_range = rotation
        if rotation_range:
            theta = np.pi / 180 * np.random.uniform(-rotation_range, rotation_range)
        else:
            theta = 0

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

        height_shift_range = height_shift
        if height_shift_range:
            tx = np.random.uniform(-height_shift_range, height_shift_range) * x.shape[self.row_index]
        else:
            tx = 0

        width_shift_range = width_shift
        if width_shift_range:
            ty = np.random.uniform(-width_shift_range, width_shift_range) * x.shape[self.col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        shear_range = shear
        if shear_range:
            shear = np.random.uniform(-shear_range, shear_range)
        else:
            shear = 0

        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        transform_matrix = np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix)

        h, w = x.shape[self.row_index], x.shape[self.col_index]
        transform_matrix = self.__transform_matrix_offset_center(transform_matrix, h, w)

        x = self.__apply_transform(x, transform_matrix, self.channel_index, fill_mode='nearest', cval=0)

        x = x.transpose((1, 2, 0))
        return x

class Square(Shape):

    def __init__(self):
        super(Square, self).__init__("square")

    def createShape(self, height_shift=0, width_shift=0, rotation=None, shear=None):
        x = np.ones((self.height, self.width, self.channels), np.uint8) * 255  # Create a white image

        pt1 = [np.random.randint(0, self.height / 2, dtype=np.uint8), np.random.randint(0, self.height / 2, dtype=np.uint8)]

        size = np.random.randint(0, 100, dtype=np.uint8) + 8
        pt2 = [pt1[0] + size, pt1[1] + size]

        pt1, pt2 = self._checkPoints(pt1, pt2)

        cv2.rectangle(x, pt1, pt2, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        x = self._transform(x, rotation=rotation, shear=0, height_shift=height_shift, width_shift=width_shift)
        return x

class Rectangle(Shape):

    def __init__(self):
        super(Rectangle, self).__init__("rectangle")

    def createShape(self, height_shift=0, width_shift=0, rotation=None, shear=None):
        x = np.ones((self.height, self.width, self.channels), np.uint8) * 255 # Create a white image

        pt1 = [np.random.randint(0, self.height / 2, dtype=np.uint8), np.random.randint(0, self.height / 2, dtype=np.uint8)]
        pt2 = [np.random.randint(0, self.height, dtype=np.uint8), np.random.randint(0, self.height, dtype=np.uint8)]

        pt1, pt2 = self._checkPoints(pt1, pt2)

        cv2.rectangle(x, pt1, pt2, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        x = self._transform(x, rotation=rotation, shear=shear, height_shift=height_shift, width_shift=width_shift)

        return x

class Parallelogram(Shape):

    def __init__(self):
        super(Parallelogram, self).__init__('parallelogram')

    def createShape(self, height_shift=0, width_shift=0, rotation=None, shear=None):
        x = np.ones((self.height, self.width, self.channels), np.uint8) * 255 # Create a white image

        pt1 = [np.random.randint(0, self.height / 2, dtype=np.uint8), np.random.randint(0, self.height / 2, dtype=np.uint8)]

        size = np.random.randint(0, 100, dtype=np.uint8) + 8
        pt2 = [pt1[0] + size, pt1[1] + size]

        pt1, pt2 = self._checkPoints(pt1, pt2)

        cv2.rectangle(x, pt1, pt2, (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        x = self._transform(x, rotation=rotation, shear=shear, height_shift=height_shift, width_shift=width_shift)

        return x

class Trapezium(Shape):

    def __init__(self):
        super(Trapezium, self).__init__('trapezium')

    def createShape(self, height_shift=0, width_shift=0, rotation=None, shear=None):
        x = np.ones((self.height, self.width, self.channels), np.uint8) * 255  # Create a white image

        pt1 = [np.random.randint(0, self.height / 2, dtype=np.uint8),
               np.random.randint(0, self.height / 2, dtype=np.uint8)]

        pt2 = [pt1[0] + np.random.randint(8, 96), pt1[1]]

        mid = int((pt2[0] + pt1[0]) / 2)
        depth = np.random.randint(0, 100, dtype=np.uint8) + 8
        angle = np.random.randint(25, 45) / 180

        pt3 = [mid * 2 + int(np.cos(angle) * mid), depth + pt1[1]]
        pt4 = [mid / 2 - int(np.cos(angle) * mid), depth + pt1[1]]

        pt1, pt2 = self._checkPoints(pt1, pt2)
        pt3, pt4 = self._checkPoints(pt3, pt4)

        cv2.line(x, pt1, pt2, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(x, pt2, pt3, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(x, pt3, pt4, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(x, pt4, pt1, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        x = self._transform(x, rotation=rotation, shear=0, height_shift=height_shift, width_shift=width_shift)
        return x

def generateShape(shape:Shape, n=1000, height_shift=0, width_shift=0, rotation=None, shear=None, seed=None):
    if seed:
        np.random.seed(seed)

    for i in range(n):
        name = shape.getShapeClass()
        s = shape.createShape(height_shift=height_shift, width_shift=width_shift, rotation=rotation, shear=shear)

        yield (name, s)

if __name__ == "__main__":
    import seaborn as sns

    square = Trapezium()

    for sq in generateShape(square, n=5, height_shift=5. / 224, width_shift=5. / 224, rotation=0, shear=0.0, seed=0):
        sns.plt.imshow(sq[1])
        sns.plt.show()