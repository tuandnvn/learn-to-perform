'''
Created on Mar 4, 2017

@author: Tuan
'''
import os
import shutil
import glob
import codecs
from os import path
from utils import DATA_DIR, FIRST_EXPERIMENT_CLASSES

'''
source_dir: Annotator Workspace from Ecat (C:\Users\CWC-BH2\AnnotatorWorkspace)
classes: A list of event types (taken as project names)
'''
def copy_param_files( source_dir, classes ) :
	for c in classes:
		project_path = path.join( source_dir, c)

		project_file = path.join( project_path, c + '.project' )

		with codecs.open (project_file, 'r') as fh:
			for line in fh:
				session_name = line.strip()

				session_path = path.join( project_path, session_name )

				src_body_scheme_file = path.join( session_path, 'bodyScheme.xml' )
				src_param_file = path.join( session_path, 'files.param' )
				src_rig_files = glob.glob(path.join( session_path, 'rig*.xml' ))
				print (src_rig_files)

				des_dir = path.join( DATA_DIR, c, session_name )

				if not os.path.exists(des_dir):
					os.makedirs(des_dir)

				shutil.copy( src_body_scheme_file, des_dir )
				shutil.copy( src_param_file, des_dir )
				for f in src_rig_files:
					shutil.copy( f, des_dir )



copy_param_files('C:\Users\CWC-BH2\AnnotatorWorkspace', FIRST_EXPERIMENT_CLASSES)

