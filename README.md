# three-impostor

Impostors for three.js `WebGPURenderer`.

[Demo](https://richardmonette.github.io/three-impostor/)

<img src="https://raw.githubusercontent.com/richardmonette/three-impostor/main/examples/screenshots/dragon.jpg" width="49%"> <img src="https://raw.githubusercontent.com/richardmonette/three-impostor/main/examples/screenshots/dragon_wireframe.jpg" width="49%">

An object is captured once into textures. It can then be drawn as a box of 12 triangles whose shader
reconstructs the object from those textures and shades it through the usual three.js lighting. Each
pixel writes its depth, so impostors intersect each other and the rest of the scene.

There are two layouts:

- **`'box'`** captures six views and ray marches them. It gives true depth, looks the same from every
  angle and holds up at closer range.
- **`'octahedral'`** captures a grid of views from around the object and blends the three nearest.
  It's cheaper per pixel and suits objects further away.

```
npm install three-impostor
```

```js
import * as THREE from 'three/webgpu';
import { ImpostorBaker, ImpostorNodeMaterial } from 'three-impostor';

const baker = new ImpostorBaker( renderer );
const near = await baker.bake( mesh );
const far = await baker.bake( mesh, { layout: 'octahedral' } );

const lod = new THREE.LOD();
lod.addLevel( mesh, 0 );
lod.addLevel( new THREE.Mesh( near.geometry, new ImpostorNodeMaterial( near ) ), 20 );
lod.addLevel( new THREE.Mesh( far.geometry, new ImpostorNodeMaterial( far ) ), 60 );
```

Many copies work with `InstancedMesh`:

```js
const forest = new THREE.InstancedMesh( far.geometry, new ImpostorNodeMaterial( far ), 5000 );
```

## API

### `new ImpostorBaker( renderer )`

`bake( object, options )` captures `object` in its local space and returns a promise for an
`ImpostorData`. The object's own position, rotation and scale are ignored, so give the impostor the
same transform as the object.

- `layout`: `'box'` (default) or `'octahedral'`
- `resolution`: the size in pixels of each view (default `256` for box, `128` for octahedral)
- `frames`: octahedral only, the number of views along each side of the grid (default `12`)
- `hemisphere`: octahedral only, capture views from above only, for objects never seen from below (default `false`)

### `ImpostorData`

- `layout`: the layout it was captured with
- `box`: the box the object was captured in, in local space
- `geometry`: a box geometry to draw the impostor with
- `colorMap`, `normalMap`, `intervalMap`: the captured textures (`intervalMap` is box only)
- `dispose()`

### `new ImpostorNodeMaterial( impostor, parameters )`

A `MeshStandardNodeMaterial`, so `color`, `roughness`, `metalness` and the other standard parameters
apply. `color` tints the captured colour.

- `steps`: box only, the largest number of ray steps for a pixel (default `64`)

## Notes

- Requires three.js r186 or later. Works with both the WebGPU and WebGL 2 backends of `WebGPURenderer`.
- The capture records base colour and normals. Roughness and metalness come from the material.
- Box: hollows that can't be seen from any of the six sides are filled in.
- Octahedral: silhouettes are slightly softer, and the default capture uses about 38 MB of GPU memory.
- Impostors don't cast shadows yet.
