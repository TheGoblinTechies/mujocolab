from import_head import *

static_model = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""

rope_model = """
<mujoco model="Rope">
    <!--
      <include file="scene.xml"/> 
    -->
    <option timestep="0.002" jacobian="dense"/>
    <size nconmax="100" njmax="300" nstack="50000"/>
    <worldbody>
        <light name="top" pos="0 0 1"/>
        <body name="B10" pos="0 0 0">
            <freejoint/>
            <composite type="rope" count="21 1 1" spacing="0.02" offset="0 0 2">
                <joint kind="main" damping="0.005" stiffness="0.4"/>
                <geom type="capsule" size=".01 .015" rgba=".8 .2 .1 1"/>
            </composite>
        </body>
    </worldbody>
</mujoco>
"""

# physics = mujoco.Physics.from_xml_string(rope_model)
# pixels = physics.render()
# PIL.Image.fromarray(pixels).show()


# # depth is a float array, in meters.
# depth = physics.render(depth=True)
# # Shift nearest values to the origin.
# depth -= depth.min()
# # Scale by 2 mean distances of near rays.
# depth /= 2*depth[depth <= 1].mean()
# # Scale to [0, 255]
# pixels = 255*np.clip(depth, 0, 1)
# PIL.Image.fromarray(pixels.astype(np.uint8)).show()

tippe_top = """
<mujoco model="tippe top">
  <option integrator="RK4"/>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3" 
     rgb2=".2 .3 .4" width="1200" height="1200"/>
    <material name="grid" texture="grid" texrepeat="12 12" reflectance=".2"/>
  </asset>
  <worldbody>
    <geom size=".4 .4 .01" type="plane" material="grid"/>
    <light pos="0 0 .6"/>
    <camera name="closeup" pos="0 -.8 .4" xyaxes="1 0 0 0 1 2"/>
        <body name="B10" pos="0 0 0.1" euler="0 0 -30">
            <freejoint/>
            <composite type="rope" count="21 1 1" spacing="0.02" offset="0 0 2">
                <joint kind="main" damping="0.005" stiffness="0.4"/>
                <geom type="capsule" size=".01 .015" rgba=".8 .2 .1 1"/>
            </composite>
        </body>
  </worldbody>
  <!--
  <keyframe>
    <key name="spinning" qpos="0 0 0 1 0 0 0" qvel="0 0 0 0 1 200" />
  </keyframe>
  -->
</mujoco>
"""
physics = mujoco.Physics.from_xml_string(tippe_top)
PIL.Image.fromarray(physics.render(camera_id='closeup')).show()

# depth is a float array, in meters.
depth = physics.render(depth=True)
# Shift nearest values to the origin.
depth -= depth.min()
# Scale by 2 mean distances of near rays.
depth /= 2*depth[depth <= 1].mean()
# Scale to [0, 255]
pixels = 255*np.clip(depth, 0, 1)
PIL.Image.fromarray(pixels.astype(np.uint8)).show()


seg = physics.render(segmentation=True)
# Display the contents of the first channel, which contains object
# IDs. The second channel, seg[:, :, 1], contains object types.
geom_ids = seg[:, :, 0]
# Infinity is mapped to -1
geom_ids = geom_ids.astype(np.float64) + 1
# Scale to [0, 1]
geom_ids = geom_ids / geom_ids.max()
pixels = 255*geom_ids
PIL.Image.fromarray(pixels.astype(np.uint8)).show()