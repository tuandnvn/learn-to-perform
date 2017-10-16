'''
Created on Oct 15, 2017

@author: Tuan

===========================
Projecting a segment of data into two dimensional plane 
delineated by the table surface
'''
import numpy as np
from utils import SESSION_OBJECTS

# Count the number of finite element in an array
def count_finite( numpy_array ):
	return np.sum( np.isfinite( numpy_array )) 

'''
===========
Params: session_data: check the return value of read_utils.load_one_param_file

Return: object_data: Dictionary
	object_data[object_name] = interpolated and 2-d data projected onto the table
	object_data[object_name][frameNo] = interpolated data from 3d
'''
def project_to2d ( session_data ):
	object_data = {}

	for object_name in session_data[SESSION_OBJECTS]:
		if object_name == 'table':
			polygon = []
			for frameNo in session_data[SESSION_OBJECTS][object_name]:
				frame_polygon = session_data[SESSION_OBJECTS][object_name][frameNo]

			polygon.append(frame_polygon)

			polygon = np.concatenate(polygon)
			polygon.shape = (polygon.shape[0] /3 , 3) 
			
			table_markers = polygon
	
	for object_name in session_data[SESSION_OBJECTS]:
		if object_name != 'table':
			for frameNo in session_data[SESSION_OBJECTS][object_name]:
				frame_data = session_data[SESSION_OBJECTS][object_name][frameNo]

				q = [(count_finite(frame_data[face_index]), face_index) for face_index in frame_data]
				q = sorted(q, key = lambda t: t[0], reverse = True)

				# Pick out the face_index with the most number of non-infinite values
				best_face_index = q[0][0]
