import numpy as np
from random import random
from random import randint
from random import choice
from numpy import array
from numpy import zeros

def check_in_frame ( size, pos ):
    """
    Allow space around the shape
    """
    if 0 <= pos[0] < size and  0 <= pos[1] < size:
        return True
    return False

def check_in_frame_with_space ( size, pos ):
    """
    Allow space around the shape
    """
    if 1 <= pos[0] < size - 1 and  1 <= pos[1] < size - 1:
        return True
    return False

def check_in_range (value, bottom, left, right, direction, position ):
    """
    bottom, left, right, direction, position is the parameters of U-shape
    
    value is a coordinates (x, y)
    """
    if direction == 0 or direction == 4:
        if position[0] <= value[0] <= position[0] + max(left, right) and position[1] <= value[1] <= position[1] + bottom:
            return True
    
    if direction == 2 or direction == 6:
        if position[0] <= value[0] <= position[0] + bottom and position[1] <= value[1] <= position[1] +  max(left, right):
            return True
    
    return False

"""
=======================================================================================================
U shape
"""
def generate_u_shape ( frame, bottom, left, right, direction, position ):
    """
    Impose an U shape structure on the frame, position is where you start it (top-left corner of rectangle)
    
    This is an U-shape with bottom = 4, left = 4, right = 3, direction = 0
    S
    o x x o
    o x x o
    o o o o
    
    S o o o 
    o x x 
    o x x 
    o o o
    
    S o o o
    o x x o
    o x x o
    o 
    
    # Let consider this case in a later phase
    This is an U-shape with bottom = 4, left = 4, right = 3, direction = 1
    
            o
          o
        o x
      o x x x
        o x x x o
          o x o
            o
    We can start with even direction first ( 0 (North), 2 (East), 4 (South), 6(West) )
    The positions marked with x are the positions that are considered inside the U-shape
    
    This function returns whether imposition successes, and frame would be imposed with the U-shape
    
    Parameters:
    =====================
    frame: squared frame
    bottom: integer value
    left: integer value
    right: integer value
    direction: 
    position: tuple of 2
    
    Returns:
    =====================
    - success: whether imposition successes or not
    - shape: contains the frame and inside points
    
    """
    size = frame.shape[0]
    
    if direction == 0 or direction == 4:
        other_corner = ( position[0] + max(left, right) - 1, position[1] + bottom - 1 )
        
    if direction == 2 or direction == 6:
        other_corner = ( position[0] + bottom - 1, position[1] + max(left, right) - 1 )

    if not check_in_frame_with_space ( size, position) or not check_in_frame_with_space ( size, other_corner ):
#         s = Shape ( frame, ( [], [] ) )
        return False, ( [], [] )
    
    # Let's create one U-shape that has a direction == 0, than we rotate it
    inner_shape = zeros((max(left, right), bottom))
    # Left side
    for i in range (max(left, right)):
        inner_shape[i,0] = 1
    
    # Right side
    for i in range (max(left, right) - min(left, right), max(left, right)):
        inner_shape[i,bottom - 1] = 1
    
    # Bottom side
    for i in range (bottom):
        inner_shape[max(left, right) - 1,i] = 1
        
    for i in range(max(left, right) - min(left, right), max(left, right) - 1):
        for j in range(1, bottom - 1):
            inner_shape[i,j] = 2
    
    rotated_shape = np.rot90 ( inner_shape, -direction // 2 )
    
    
    frame[ position[0] : position[0] + rotated_shape.shape[0], 
          position[1] : position[1] + rotated_shape.shape[1] ] = rotated_shape
    
    inside_points = np.where(frame == 2)
    
    frame[frame == 2] = 0
    
#     s = Shape ( frame, inside_points )
    
    return True, inside_points

def generate_u_shape_frame ( size, bottom_range = list(range(3,5)), 
                  left_range = list(range(3,5)), right_range = list(range(3,5)), 
                  direction_range = list(range(0,8,2)) ):
    frame = zeros((size, size))
    
    # You random a value in those range to add into the frame
    # You also
    bottom = choice(bottom_range)
    left = choice(left_range)
    right = choice(right_range)
    direction = choice(direction_range)
    while True:
        position_x = randint ( 0, size - 1 )
        position_y = randint ( 0, size - 1 )
        position = (position_x, position_y)

        success, inner_list = generate_u_shape  ( frame, bottom, left, right, direction, position )
        
        if success : 
            break
    
    # Random outer position for moving point
    while True:
        pos_start_x = randint ( 0, size - 1 )
        pos_start_y = randint ( 0, size - 1 )
        
        pos_start = (pos_start_x, pos_start_y)
        
        # Just need to be out of the rectangle
        if check_in_range ( pos_start, bottom, left, right, direction, position ):
            continue
            
        break
    
    # Random inner position for moving point
    index = randint ( 0, len(inner_list[0]) - 1 )
    pos_end = (inner_list[0][index], inner_list[1][index])
    
    return frame, pos_start, pos_end

def generate_u_shape_inside ( size, bottom_range = list(range(3,5)), 
                  left_range = list(range(3,5)), right_range = list(range(3,5)), 
                  direction_range = list(range(0,8,2)) ):
    """
    Generate sample for inside generation
    """
    frame = zeros((size, size))
    
    # You random a value in those range to add into the frame
    # You also
    bottom = choice(bottom_range)
    left = choice(left_range)
    right = choice(right_range)
    direction = choice(direction_range)
    while True:
        position_x = randint ( 0, size - 1 )
        position_y = randint ( 0, size - 1 )
        position = (position_x, position_y)

        success, inner_list = generate_u_shape  ( frame, bottom, left, right, direction, position )
        
        if success : 
            break

    inside_frame = zeros((size, size))

    inside_frame[inner_list] = 1

    return frame, inside_frame


"""
=======================================================================================================
Jar shape
"""

def generate_jar_shape ( frame, width, height, bulb, side, side_start, side_len, direction, position ):
    """
    Impose an jar shape structure on the frame, position is where you start it (top-left corner of rectangle).
    This is actually a variance of the U-shape.
    The main difference is 
    
    All of the followings have direction = 0, S is where position is
    
    width = 3, height = 5, bulb = 1 (bulb extended by 1 cell), side = 0 (left, 1 for right), 
    side_start = 1 (from the top), side_len = 3 ( side_start + side_len < height )
    
      
    S o x o
    o o x o
    o x x o
    o o x o
      o o o
    
    side_start = 2
    
    S o x o
      o x o
    o o x o
    o x x o
    o o o o
    
    bulb = 2, side_start = 2
    S   o x o
        o x o
    o o o x o
    o x x x o
    o o o o o
    
    This function returns whether imposition successes, and frame would be imposed with the jar-shape
    
    Parameters:
    =====================
    frame: squared frame
    width: width of the main rectangular
    height: height of the main rectangular
    bulb: 
    side: 
    side_start: 
    side_len:
    direction: 
    position: tuple of 2
    
    Returns:
    =====================
    - success: whether imposition successes or not
    - inside_points: list of points inside the shape (x)
    
    """
    size = frame.shape[0]
    
    if direction == 0 or direction == 4:
        other_corner = ( position[0] + height - 1, position[1] + width + bulb - 1 )
        
    if direction == 2 or direction == 6:
        other_corner = ( position[0] + width + bulb - 1, position[1] + height - 1 )

    if not check_in_frame_with_space ( size, position) or not check_in_frame_with_space ( size, other_corner ):
        return False, ( [], [] )
    
    if side_start + side_len >= height:
        return False, ( [], [] )
    
    # Let's create one jar-shape that has a direction == 0, and side = 0
    inner_shape = np.zeros((height, width + bulb))

    # Right side: draw straight line
    for i in range (height):
        inner_shape[i,-1] = 1
    
    # Left side bulb: 5 segments
    for i in range (0, side_start):
        inner_shape[i, bulb] = 1
    
    for i in range (0, bulb + 1):
        inner_shape[side_start, i] = 1
        
    for i in range (side_start, side_start + side_len):
        inner_shape[i, 0] = 1
        
    for i in range (0, bulb + 1):
        inner_shape[side_start + side_len, i] = 1
    
    for i in range (side_start + side_len, height):
        inner_shape[i, bulb] = 1
    
    # Bottom side
    for i in range (bulb, width + bulb):
        inner_shape[height - 1, i] = 1
    
    # Jar inner
    for i in range(0, height - 1):
        for j in range(bulb + 1, width + bulb - 1):
            inner_shape[i,j] = 2
    
    # Bulb inner
    for i in range(side_start + 1, side_start + side_len):
        for j in range(1, width):
            inner_shape[i,j] = 2
    
    if side == 1:
        # Flip through the Oy axis
        inner_shape = inner_shape[:,::-1]
        
    rotated_shape = np.rot90 ( inner_shape, -direction // 2 )
    
    frame[ position[0] : position[0] + rotated_shape.shape[0], 
          position[1] : position[1] + rotated_shape.shape[1] ] = rotated_shape
    
    inside_points = np.where(frame == 2)
    
    frame[frame == 2] = 0
    
    return True, inside_points

def generate_jar_shape_frame ( size, width_range = list(range(3,6)), 
                  height_range = list(range(4,7)), bulb_range = list(range(1,3)),
                  side_start_range = list(range(0,3)), side_len_range = list(range(2,4)),
                  direction_range = list(range(0,8,2)) ):
    frame = zeros((size,size))
    
    # You random a value in those range to add into the frame
    # You also
    
    width = choice(width_range)
    height = choice(height_range)
    bulb = choice(bulb_range)
    side = choice([0,1])
    side_start = choice(side_start_range)
    side_len = choice(side_len_range)
    direction = choice(direction_range)
    
    for _ in range(20):
        position_x = randint ( 0, size - 1 )
        position_y = randint ( 0, size - 1 )
        position = (position_x, position_y)

        success, inner_list = generate_jar_shape  ( frame, width, height, bulb, side, 
                                                 side_start, side_len, direction, position )
        
        
        if success :
            break
    else:
        return None, None, None
    
    # Random outer position for moving point
    for _ in range(20):
        pos_start_x = randint ( 0, size - 1 )
        pos_start_y = randint ( 0, size - 1 )
        
        pos_start = (pos_start_x, pos_start_y)
        
        # Just need to be out of the rectangle
        if not check_in_range_jar ( pos_start, width, height, bulb, direction, position ):
            break
    else:
        return None, None, None
    
    # Random inner position for moving point
    index = randint ( 0, len(inner_list[0]) - 1 )
    pos_end = (inner_list[0][index], inner_list[1][index])
    
#     frame[pos_start] = 3
#     frame[pos_end] = 4
    
    return frame, pos_start, pos_end

def generate_jar_shape_inside ( size, width_range = list(range(3,6)), 
                  height_range = list(range(4,7)), bulb_range = list(range(1,3)),
                  side_start_range = list(range(0,3)), side_len_range = list(range(2,4)),
                  direction_range = list(range(0,8,2)) ):
    frame = zeros((size,size))
    
    # You random a value in those range to add into the frame
    # You also
    
    width = choice(width_range)
    height = choice(height_range)
    bulb = choice(bulb_range)
    side = choice([0,1])
    side_start = choice(side_start_range)
    side_len = choice(side_len_range)
    direction = choice(direction_range)
    
    for _ in range(20):
        position_x = randint ( 0, size - 1 )
        position_y = randint ( 0, size - 1 )
        position = (position_x, position_y)

        success, inner_list = generate_jar_shape  ( frame, width, height, bulb, side, 
                                                 side_start, side_len, direction, position )
        
        
        if success :
            break
    else:
        return None, None
    
    inside_frame = zeros((size, size))

    inside_frame[inner_list] = 1

    return frame, inside_frame