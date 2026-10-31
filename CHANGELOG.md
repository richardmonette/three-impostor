# Changelog

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
