# Changelog

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
