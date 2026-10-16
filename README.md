```
Architecture/
|-- README.md  # repo file tree with short descriptions
|-- human/
|   |-- average_human.py  # draw average human silhouette from ANSUR II data
|   |-- gaussian_rejection.py  # gaussian-based triangle rejection by body width
|   |-- human_triangle_rejection.py  # progressive rejection using pendulum RNG
|   |-- scatter_canvas.py  # map pendulum-derived values to scatter points
|   |-- triangle_scatter.py  # place isosceles triangles using pendulum data
|   |-- scatter_coordinates.csv  # scatter points exported by scatter_canvas.py
|   |-- average_human.png  # raster output of average silhouette
|   |-- average_human.svg  # vector output of average silhouette
|   |-- gaussian_bands.png  # gaussian band visualization
|   |-- gaussian_rejection.png  # gaussian rejection result image
|   |-- human_silhouette.svg  # vector outline for human figure
|   |-- human_triangle_rejection.png  # rejection result image
|   |-- scatter_canvas.png  # scatter plot output
|   |-- triangle_scatter.png  # triangle scatter output (raster)
|   |-- triangle_scatter.svg  # triangle scatter output (vector)
|   |-- gaussian_frames/
|   |   |-- frame_000.png  # gaussian rejection frame
|   |   `-- frame_001.png  # gaussian rejection frame
|   `-- rejection_frames/
|       |-- frame_000.png  # rejection animation frame
|       |-- frame_001.png  # rejection animation frame
|       |-- frame_002.png  # rejection animation frame
|       |-- frame_003.png  # rejection animation frame
|       |-- frame_004.png  # rejection animation frame
|       |-- frame_005.png  # rejection animation frame
|       |-- frame_006.png  # rejection animation frame
|       |-- frame_007.png  # rejection animation frame
|       |-- frame_008.png  # rejection animation frame
|       |-- frame_009.png  # rejection animation frame
|       |-- frame_010.png  # rejection animation frame
|       |-- frame_011.png  # rejection animation frame
|       |-- frame_012.png  # rejection animation frame
|       |-- frame_013.png  # rejection animation frame
|       `-- frame_target.png  # target silhouette frame
|-- latex_physics/
|   |-- double_pendulum_math.tex  # LaTeX derivation of double pendulum equations
|   |-- README.md  # notes and build instructions for LaTeX
|   |-- double_pendulum_math.pdf  # compiled PDF output
|   |-- double_pendulum_math.aux  # LaTeX build artifact
|   |-- double_pendulum_math.log  # LaTeX build artifact
|   |-- double_pendulum_math.out  # LaTeX build artifact
|   `-- double_pendulum_math.toc  # LaTeX table of contents
|-- randomness/
|   |-- double_pendulum_animation.py  # animate multiple pendulums with trails
|   |-- double_pendulum_sim.py  # RK4 solver and inverse kinematics utilities
|   |-- pendulum_panel.py  # generate a panel of pendulum snapshots
|   |-- single_pendulum_demo.py  # single pendulum demo animation
|   |-- spatial_perturbation/
|   |   |-- double_pendulum_histograms.py  # x/y histograms over time
|   |   |-- double_pendulum_histograms_x.py  # x-only histograms over time
|   |   |-- double_pendulum_sampler.py  # sample positions across initial x range
|   |   |-- double_pendulum_sampler_anim.py  # animate sampler trajectories
|   |   |-- sample_initial_positions.py  # map initial x to final x at t=10s
|   |   |-- spatial_autocorrelation.py  # spatial autocorrelation of final x
|   |   |-- sample_initial_positions.csv  # sampled initial/final x data
|   |   |-- spatial_autocorrelation.csv  # autocorrelation data
|   |   |-- double_pendulum_histograms.png  # histogram figure
|   |   |-- double_pendulum_histograms_x.png  # x histogram figure
|   |   |-- sample_initial_positions.png  # mapping/histogram figure
|   |   |-- spatial_autocorrelation.png  # autocorrelation figure
|   |   `-- __pycache__/
|   |       `-- double_pendulum_sampler.cpython-314.pyc  # python bytecode cache
|   |-- temporal_sampling/
|   |   |-- pendulum_autocorrelation.py  # time series + autocorrelation of x
|   |   `-- pendulum_autocorrelation.png  # autocorrelation figure
|   |-- Untitled  # scratch pointer to double_pendulum_animation.py
|   |-- double_pendulum_trails.gif  # saved animation output
|   `-- __pycache__/
|       |-- double_pendulum_sampler.cpython-314.pyc  # python bytecode cache
|       `-- double_pendulum_sim.cpython-314.pyc  # python bytecode cache
|-- double_pendulum_histograms_x.png  # top-level x histogram output
`-- pendulum_panel.png  # pendulum panel output
```