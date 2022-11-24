import sys
import bpy #pip install fake-bpy-module-3.3
import random
import math


'''
(VENV) pberck@ip30-163 Blender % pip list
Package             Version
------------------- --------
fake-bpy-module-3.3 20221006
numpy               1.23.5
pandas              1.5.2
pip                 22.3.1
python-dateutil     2.8.2
pytz                2022.6
setuptools          58.0.4
six                 1.16.0

Start Blender: /Applications/Blender.app/Contents/MacOS/Blender
'''

if False:
    import pandas as pd
    pos_filename = "0001.txt"
    df = pd.read_csv(
        pos_filename,
        delim_whitespace=True,
    )
    print( df.head() )
    # For testing, we just save channel 1
    header = ["Ch1_X", "Ch1_Y", "Ch1_Z", "Ch1_phi", "Ch1_theta", "Ch1_RMS", "Ch1_Extra"]
    df.to_csv(pos_filename+'.ch01.csv', columns=header, index=False)
    for c in range(1,17):
        header = ["Ch"+str(c)+"_X", "Ch"+str(c)+"_Y", "Ch"+str(c)+"_Z", "Ch"+str(c)+"_phi", "Ch"+str(c)+"_theta", "Ch"+str(c)+"_RMS", "Ch"+str(c)+"_Extra"]
        df.to_csv(pos_filename+'.ch'+str(c)+'.csv', columns=header, index=False)
    sys.exit(0)
    
# Blender code

def create_balls(n):
    balls = []
    #makes the objects
    for i in range(n): 
        x, y, z  = 0, 0, 0
        bpy.ops.mesh.primitive_uv_sphere_add( 
            location = [ x, y, z],
            segments = 16
        )
        ob = bpy.ops.object
        balls.append(ob)
        #smoothes the spheres 
        #bpy.ops.object.shade_smooth()
        #bpy.ops.object.shade_smooth()
        #bpy.ops.object.modifier_add(type='SUBSURF')
    return balls

# ----

sensors = [] # Global list with sensors

class Sensor:
    def __init__(self, name):
        self.name      = name
        self.positions = []
        self.filename  = None
        self.jheader   = None
        self.obj       = None # Blender object
    def __str__(self):
        return self.name
    def read_data(self, filename):
        self.filename = filename
        with open(filename, "r") as f:
            self.header = f.readline()
            print( self.header.strip() )
            for line in f:
                bits = line.split(",")
                self.positions.append( (float(bits[0]), float(bits[1]), float(bits[2])-40.0) )
            print( "Read", len(self.positions), "positions." )
    def create_obj(self):
        x, y, z = 0, 0, 0
        bpy.ops.mesh.primitive_uv_sphere_add( 
            location = [ x, y, z],
            segments = 16
        )
        #bpy.ops.object.shade_smooth()
        #bpy.ops.object.shade_smooth()
        #bpy.ops.object.modifier_add(type='SUBSURF')

        activeObject = bpy.context.active_object #Set active object to variable
        mat = bpy.data.materials.new(name="MaterialName") #set new material to variable
        activeObject.data.materials.append(mat) #add the material to the object
        bpy.context.object.active_material.diffuse_color = (random.random(), random.random(), random.random(), 1) 
        
        bpy.context.active_object.name = self.name
        self.obj = bpy.context.active_object
        # my_obj = bpy.context.selectable_objects[0]
        
# Clear all in Blender
bpy.ops.object.select_all(action='SELECT')
for ob in bpy.context.selectable_objects:
    bpy.ops.object.delete(use_global=False)

for c in range(1, 17):
    sensor = Sensor("Channel "+str(c)) 
    sensor.read_data( "0001.txt.ch"+str(c)+".csv" )
    sensor.create_obj()
    sensors.append( sensor )

bpy.ops.object.select_all(action='SELECT')
for ob in bpy.context.selectable_objects:
    print( ob )

# For rendering
bpy.context.scene.render.fps = 200
bpy.data.scenes['Scene'].render.fps = 200
#bge.logic.setLogicTicRate(200)

# Set all positions
for sensor in sensors:
    frame_num = 0
    for position in sensor.positions: # all positions in the object positions list
        bpy.context.scene.frame_set( frame_num )
        sensor.obj.location = position
        sensor.obj.keyframe_insert( data_path="location", index = -1 )
        frame_num += 1

bpy.ops.object.camera_add(
    enter_editmode=False,
    align='VIEW',
    location=(0, 0, 0),
    rotation=(1.50668, 0.0143282, -2.10721),
    scale=(1, 1, 1)
)
obj = bpy.context.object.name = "Camera 1"
'''
# Create a camera
scn = bpy.context.scene
cam = bpy.data.cameras.new("Camera 1")
cam.lens = 18

# create the first camera object
cam_obj = bpy.data.objects.new("Camera 1", cam)
cam_obj.location = (9.69, -10.85, 12.388)
cam_obj.rotation_euler = (0.6799, 0, 0.8254)
scn.collection.objects.link(cam_obj)
'''

#bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
# Create light datablock
light_data = bpy.data.lights.new(name="my-light-data", type='POINT')
light_data.energy = 1000
light_data.shadow_soft_size = 2
# Create new object, pass the light data 
light_object = bpy.data.objects.new(name="Light 1", object_data=light_data)
# Link object to collection in context
bpy.context.collection.objects.link(light_object)
# Change light position
light_object.location = (40, -20, 3)

