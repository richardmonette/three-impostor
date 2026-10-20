# three-impostor

Ray-marched impostors for three.js `WebGPURenderer`.

[Demo](https://richardmonette.github.io/three-impostor/)

<img src="https://raw.githubusercontent.com/richardmonette/three-impostor/main/examples/screenshots/dragon.jpg" width="49%"> <img src="https://raw.githubusercontent.com/richardmonette/three-impostor/main/examples/screenshots/dragon_wireframe.jpg" width="49%">

An object is captured once from six sides. It can then be drawn as a box of 12 triangles whose shader
marches each pixel's ray through the captured data, finds the surface, and shades it with the captured
colour and normal through the usual three.js lighting. The hit writes its real depth, so impostors
intersect each other and the rest of the scene correctly.

```
npm install three-impostor
```

```js
import * as THREE from 'three/webgpu';
import { ImpostorBaker, ImpostorNodeMaterial } from 'three-impostor';

const impostor = await new ImpostorBaker( renderer ).bake( mesh );

const lod = new THREE.LOD();
lod.addLevel( mesh, 0 );
lod.addLevel( new THREE.Mesh( impostor.geometry, new ImpostorNodeMaterial( impostor ) ), 20 );
```

Many copies work with `InstancedMesh`:

```js
const forest = new THREE.InstancedMesh( impostor.geometry, new ImpostorNodeMaterial( impostor ), 5000 );
```

## API

### `new ImpostorBaker( renderer )`

`bake( object, { resolution = 256 } )` captures `object` in its local space and returns a promise for
an `ImpostorData`. The object's own position, rotation and scale are ignored, so give the impostor the
same transform as the object.

### `ImpostorData`

- `box`: the box the object was captured in, in local space
- `geometry`: a box geometry to draw the impostor with
- `colorMap`, `normalMap`, `intervalMap`: the captured textures
- `dispose()`

### `new ImpostorNodeMaterial( impostor, parameters )`

A `MeshStandardNodeMaterial`, so `color`, `roughness`, `metalness` and the other standard parameters
apply. `color` tints the captured colour.

- `steps`: the largest number of ray steps for a pixel (default `64`)

## Notes

- Requires three.js r186 or later. Works with both the WebGPU and WebGL 2 backends of `WebGPURenderer`.
- The capture records base colour and normals. Roughness and metalness come from the material.
- Hollows that can't be seen from any of the six sides are filled in.
- Impostors don't cast shadows yet.
