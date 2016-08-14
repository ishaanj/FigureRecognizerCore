import numpy as np
import scipy.ndimage as ndi
import cv2
from scipy.misc import imsave

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

    def _checkPoints(self, pts):
        minimum = 16
        maximum = 200

        for i, pt in enumerate(pts):
            pt[0] = pt[0] if pt[0] > minimum else minimum
            pt[1] = pt[1] if pt[1] > minimum else minimum

            pt[0] = pt[0] if pt[0] < maximum else maximum
            pt[1] = pt[1] if pt[1] < maximum else maximum

            pts[i] = (pt[0], pt[1])

        return pts


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

    def createShape(self, height_shift=10, width_shift=10, rotation=None, shear=0.5):
        x = np.ones((self.height, self.width, self.channels), np.uint8) * 255  # Create a white image

        pt1 = [np.random.randint(0, self.height / 2, dtype=np.uint8), np.random.randint(0, self.height / 2, dtype=np.uint8)]

        size = np.random.randint(0, 100, dtype=np.uint8) + 8
        pt2 = [pt1[0] + size, pt1[1] + size]

        pt1, pt2 = self._checkPoints([pt1, pt2])

        cv2.rectangle(x, pt1, pt2, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        x = self._transform(x, rotation=rotation, shear=0, height_shift=height_shift, width_shift=width_shift)
        return x

class Rectangle(Shape):

    def __init__(self):
        super(Rectangle, self).__init__("rectangle")

    def createShape(self, height_shift=10, width_shift=10, rotation=60, shear=0.5):
        x = np.ones((self.height, self.width, self.channels), np.uint8) * 255 # Create a white image

        pt1 = [np.random.randint(0, self.height / 2, dtype=np.uint8), np.random.randint(0, self.height / 2, dtype=np.uint8)]
        pt2 = [np.random.randint(0, self.height, dtype=np.uint8), np.random.randint(0, self.height, dtype=np.uint8)]

        pt1, pt2 = self._checkPoints([pt1, pt2])

        cv2.rectangle(x, pt1, pt2, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        x = self._transform(x, rotation=rotation, shear=shear, height_shift=height_shift, width_shift=width_shift)

        return x

class Parallelogram(Shape):

    def __init__(self):
        super(Parallelogram, self).__init__('parallelogram')

    def createShape(self, height_shift=10, width_shift=10, rotation=45, shear=0.5):
        x = np.ones((self.height, self.width, self.channels), np.uint8) * 255 # Create a white image

        pt1 = [np.random.randint(0, self.height / 2, dtype=np.uint8), np.random.randint(0, self.height / 2, dtype=np.uint8)]

        size = np.random.randint(0, 100, dtype=np.uint8) + 8
        pt2 = [pt1[0] + size, pt1[1] + size]

        pt1, pt2 = self._checkPoints([pt1, pt2])

        cv2.rectangle(x, pt1, pt2, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        x = self._transform(x, rotation=rotation, shear=shear, height_shift=height_shift, width_shift=width_shift)

        return x

class Trapezium(Shape):

    def __init__(self):
        super(Trapezium, self).__init__('trapezium')

    def createShape(self, height_shift=10, width_shift=10, rotation=45, shear=None):
        x = np.ones((self.height, self.width, self.channels), np.uint8) * 255  # Create a white image

        pt1 = [np.random.randint(0, self.height / 2, dtype=np.uint8),
               np.random.randint(0, self.height / 2, dtype=np.uint8)]

        pt2 = [pt1[0] + np.random.randint(8, 96), pt1[1]]

        mid = int((pt2[0] + pt1[0]) / 2)
        depth = np.random.randint(15, 100, dtype=np.uint8) + 8
        angle = np.random.randint(25, 45) / 180

        pt3 = [mid * 2 + int(np.cos(angle) * mid), depth + pt1[1]]
        pt4 = [mid // 2 - int(np.cos(angle) * mid), depth + pt1[1]]

        pt1, pt2, pt3, pt4 = self._checkPoints([pt1, pt2, pt3, pt4])

        cv2.line(x, pt1, pt2, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(x, pt2, pt3, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(x, pt3, pt4, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(x, pt4, pt1, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        x = self._transform(x, rotation=rotation, shear=0, height_shift=height_shift, width_shift=width_shift)
        return x

class Triangle(Shape):

    def __init__(self):
        super(Triangle, self).__init__('triangle')

    def createShape(self, height_shift=10, width_shift=10, rotation=60, shear=None):
        x = np.ones((self.height, self.width, self.channels), np.uint8) * 255  # Create a white image

        pt1 = [np.random.randint(0, self.height / 2), np.random.randint(0, self.height / 2)]

        height = np.random.randint(15, 100, dtype=np.uint8) + 8
        angle = np.random.randint(25, 75) / 180

        pt2 = [height * 2 + int(np.cos(90 - angle) * height), height + pt1[1]]

        rightAngledProba = 0.33
        if np.random.random() >= rightAngledProba:
            pt3 = [height // 2 - int(np.cos(90 - angle) * height), height + pt1[1]]
        else:
            pt3 = [pt1[0], height + pt1[1]]

        pt1, pt2, pt3 = self._checkPoints([pt1, pt2, pt3])

        cv2.line(x, pt1, pt2, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(x, pt2, pt3, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv2.line(x, pt3, pt1, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        x = self._transform(x, rotation=rotation, shear=0, height_shift=height_shift, width_shift=width_shift)
        return x

class Circle(Shape):

    def __init__(self):
        super(Circle, self).__init__("circle")

    def createShape(self, height_shift=None, width_shift=None, rotation=None, shear=None):
        x = np.ones((self.height, self.width, self.channels), np.uint8) * 255  # Create a white image

        center = (np.random.randint(50, 3 * self.height / 4), np.random.randint(50, 3 * self.height / 4))
        radius = np.random.randint(20, 75)

        cv2.circle(x, center, radius, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        return x

class Ellipse(Shape):

    def __init__(self):
        super(Ellipse, self).__init__('ellipse')

    def createShape(self, height_shift=0, width_shift=0, rotation=None, shear=None):
        x = np.ones((self.height, self.width, self.channels), np.uint8) * 255  # Create a white image

        center = (np.random.randint(50, 3 * self.height / 4), np.random.randint(50, 3 * self.height / 4))
        axis = (np.random.randint(20, 75), np.random.randint(10, 50))
        angle = np.random.randint(0, 45)
        startAngle = 0
        endAngle = 360

        cv2.ellipse(x, center, axis, angle, startAngle, endAngle, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        return x

def generateShape(shape:Shape, n=1000, seed=None):
    if seed:
        np.random.seed(seed)

    for i in range(n):
        name = shape.getShapeClass()
        s = shape.createShape()

        yield (name, s)

def saveShapes(shape:Shape, fn):
    import os

    n = 60000
    path = r'data/' + fn + "/"
    if not os.path.exists(path): os.makedirs(path)

    for i, (name, s) in enumerate(generateShape(shape, n=n)):
        file = path + '%d.png' % (i+1)
        imsave(file, s)

        limits = n // 100
        if i % limits == 0: print('%0.2f completed (%s)' % (i / n, fn))

if __name__ == "__main__":
    '''
    Generator Script

    -  Generates 50,000 samples training, 10,000 testing / validation images
    -  Puts them in the correct directory, so that we can use ImageDataGenerator.flow_from_directory() in Keras
       to load the images from specific directories into main memory
    '''
    import os
    if not os.path.exists('data/'): os.makedirs('data/')

    # Square generator
    square = Square()
    saveShapes(square, square.getShapeClass())

    # Rectangle generator
    rect = Rectangle()
    saveShapes(rect, rect.getShapeClass())

    # Parallelogram generator
    parallelogram = Parallelogram()
    saveShapes(parallelogram, parallelogram.getShapeClass())

    # Trapezium generator
    trap = Trapezium()
    saveShapes(trap, trap.getShapeClass())

    # Trangle generator
    tri = Triangle()
    saveShapes(tri, tri.getShapeClass())

    # Circle generator
    circle = Circle()
    saveShapes(circle, circle.getShapeClass())

    # Ellipse generator
    ellipse = Ellipse()
    saveShapes(ellipse, ellipse.getShapeClass())