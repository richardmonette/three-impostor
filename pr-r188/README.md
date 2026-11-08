# r188 PR materials

Drafts and images for the three.js community example PR, and for the three-impostor 0.3.0 release.

The example itself lives on `richardmonette/three.js`, branch `impostors-community-example`
(`examples/webgpu_impostors.html`, its screenshot, and the `files.json` and `tags.json` entries).

- `PR.md`: description for the three.js PR. Replace each image filename line with the uploaded image.
- `lod_row.jpg`, `lod_grid.jpg`: the example's five dragon LOD row, and the 400 dragon grid.
- `box_near.jpg`, `box_far.jpg`, `octahedral_near.jpg`, `octahedral_far.jpg`: each layout next to the mesh.
- `PR-0.3.0.md`, `0.3.0/`: description and before/after images for the 0.3.0 release (already merged).
- `capture.mjs`: retakes the row and grid images from a local three.js checkout.
- `debug.html`, `lib/`: a test page for comparing versions. `lib/v020.js` and `lib/v030.js` are copies of
  `Impostor.js` with a debug mode that colours each pixel by the box view it is shaded from.
  Serve this repository and open `pr-r188/debug.html?lib=v030&debug=views&az=30&el=10`.
  Other parameters: `layout`, `res`, `frames`, `dist`, `show=mesh`, `webgl=1`.
