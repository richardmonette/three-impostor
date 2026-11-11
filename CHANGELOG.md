# Changelog

## Unreleased

- Point lights, spot lights and received shadows are now worked out at the surface the impostor finds,
  not at the face of the box that was drawn, so they light and shadow it as they do the mesh. Directional
  and environment light were already right.
- Box: thin parts, like rods and wires two or three texels across, no longer break up into dots. The
  texels around a surface now close it half a texel out, where its edge is, whatever its thickness.

## 0.4.0

- Textured materials keep their roughness, metalness, ambient occlusion and emissive light: the bake
  captures them into `materialMap` (roughness, metalness and ambient occlusion in one 8-bit atlas) and
  `emissiveMap` (only when some part of the object is emissive).
- `bake( object, { maps } )` chooses which of `'roughness'`, `'metalness'`, `'ao'` and `'emissive'` to
  capture; all four by default. `maps: []` captures none, as before.
- The material's `roughness` and `metalness` now scale the captured values, as they do a texture map, and
  `metalness` defaults to `1` when metalness was captured. Objects with a single plain material look the
  same as before without passing `roughness` or `metalness` to the impostor.
- Box: open meshes, like a scan with no base, no longer lose the part behind the opening. A side that
  looks into an opening now takes the object to run on to that side of the box, and the other two axes
  carve its shape. Closed meshes are unaffected.

## 0.3.1

- Box: the ray march skips empty space using a small range texture made at bake time (`rangeMap`) and
  steps a texel and a half at a time elsewhere, so thin parts like a bow or a string are no longer
  stepped over and thin flat parts no longer show stripes. `steps` now defaults to `128`.
- Box: a thin part is seen from both sides, so only views whose surface faces the ray now shade it. Its
  normals no longer cancel out, which drew it black.

## 0.3.0

- Box: keep the first gap inside the object along each axis, so parts in front of each other no longer
  merge into one solid
- Box: shade each point from the views that face it, which removes streaks where a view saw the
  surface edge on
- Box: surfaces that no view saw, like the far side of a gap, take their normal from the shape of the
  intervals, so they are lit like the rest of the object

## 0.2.0

- Octahedral layout: `bake( object, { layout: 'octahedral' } )`
- Screenshots in the README

## 0.1.0

- Initial release
