'''
Created on October 15, 2017

@author: Tuan

Load marker files,
mapping between index of marker (an integer number) to a code form (size x size of 0 and 1)
'''
import os
import numpy as np
import glob
import codecs
from utils import GLYPH_DIR


class Glyph():
    '''
    values is a numpy array of (size, size)
    '''
    def __init__( self, values):
        self.values = values

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if np.all( np.equal( self.values, other.values )):
                return True

            # Rotate 90
            t = np.rot90(other.values)

            if np.all( np.equal( self.values, t )):
                return True

            # Rotate 180
            t = np.rot90(t)

            if np.all( np.equal( self.values, t )):
                return True

            # Rotate 270
            t = np.rot90(t)

            if np.all( np.equal( self.values, t )):
                return True

            return False
        return NotImplemented

    def __ne__(self, other):
        """Define a non-equality test"""
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    @staticmethod
    def getHash( t ):
        code = 1
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                code = int(code * 2 + t[i][j])
        return code

    '''
    Hash value is basically the minimum values among 4 rotated array
    '''
    def __hash__(self):
        """Override the default hash behavior (that returns the id or the object)"""
        t = self.values

        vals = []
        vals.append(Glyph.getHash(t))

        for _ in range(3):
            t = np.rot90(t)
            vals.append(Glyph.getHash(t))

        return int(np.min(vals))


    def __str__(self):
        return str(self.values)

def read_glyphs( glyph_dir ):
    glyphs = {}

    glyph_files = glob.glob( os.path.join( glyph_dir, '*.glyph' ))

    for glyph_file in glyph_files:
        index = int(glyph_file.split(os.path.sep)[-1][:-len('.glyph')])

        number_array = []
        with codecs.open(glyph_file, 'r') as fh:
            for line in fh:
                for char in line.strip():
                    number_array.append(int(char))

        n = np.array(number_array, dtype = np.int32)
        n.shape = (int(np.sqrt(n.shape[0])), int(np.sqrt(n.shape[0])))

        # Mirroring the array
        g = Glyph( np.fliplr(n) )

        glyphs[index] = g

    return glyphs

'''
An array = (size x size) to a glyph
'''
def from_1d_array_to_face_index(array, glyphs):
    n = np.array(array)
    n.shape = (int(np.sqrt(n.shape[0])), int(np.sqrt(n.shape[0])))
    n = Glyph(n)

    keys = [k for k, v in glyphs.items() if v == n]

    if len(keys) > 0:
        return keys[0]
    return None

glyphs = read_glyphs(GLYPH_DIR)


# q = np.array([[1,1,1,0,1], [1,0,0,1,0], [1,0,0,1,0], [1,0,0,1,0], [1,0,0,1,0]])
# g = Glyph(np.rot90(q))

# print (hash(g))
# print (hash(glyphs[426]))

# print (g in glyphs.values())