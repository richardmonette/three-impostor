Title: Examples: Add impostors community example.

---

Related issue: #22043

**Description**

This adds a community example for [three-impostor](https://github.com/richardmonette/three-impostor), a small package that bakes a mesh into a few textures and then draws it as a 12 triangle box with `WebGPURenderer`. Like #35043, the code lives in its own package and the example loads it from npm.

The example shows impostors the way they're meant to be used, as levels of detail. It puts five dragons (91,216 triangles each) in a row, and each one is a `THREE.LOD` with three levels:

- the mesh up close,
- a **box** impostor at middle distance, which captures six views and ray marches them,
- an **octahedral** impostor further away, which captures a grid of views and blends the nearest three.

The LOD switches between them as the camera moves, and each dragon's label shows which level it's drawing. The settings let you change the capture resolution of each impostor (and the number of views for octahedral), turn LOD off to draw every dragon as a mesh, and toggle a wireframe that is coloured by level (blue mesh, pink box, green octahedral).

lod_row.jpg

The dragon count can also be set to 100 or 400, which lays them out in a grid. That's where impostors pay off: on a laptop Radeon Pro 5300M, 400 dragons run at 60 fps with LOD on, and drop to about 28 fps with LOD off, where every dragon is a mesh (36 million triangles).

lod_grid.jpg

Up close, next to the mesh:

| Box | Octahedral |
| --- | --- |
| box_near.jpg | octahedral_near.jpg |
| box_far.jpg | octahedral_far.jpg |

Box holds up well near the camera. Octahedral shows some blending between its views up close, which is why it's the far level. From further away, both are hard to tell apart from the mesh.
