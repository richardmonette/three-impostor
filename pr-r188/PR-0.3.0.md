Two improvements to the box layout.

**Parts in front of each other stay separate.** For each axis, the bake now also finds the first gap inside the object, where the first part ends and the next one starts, from a back face pass and a peeled front face pass. It's stored in the two spare channels of the interval texture. Before, a leg in front of the body merged with it into one solid whenever the space between them couldn't be seen straight on from any of the six sides.

**Fewer streaks.** Each point is now shaded mostly from the views that face it. A view that saw the surface edge on smeared its pixels along its line of sight, which showed up as streaks in the colour and the lighting.

**Hidden surfaces are lit properly.** Where a gap shows a surface that none of the six views saw, like the body behind the leg, the normal now comes from the shape of the intervals instead of from a view that saw something else. Gaps are also trimmed to what the neighbouring texels agree is empty, so filtering can't carve cracks into solid parts.

angle.jpg

front.jpg

The debug views below colour each pixel by the captured view it's shaded from. In 0.2.0 large areas blend several views, including ones that saw the surface edge on. In 0.3.0 each area comes mostly from the view facing it, and the body behind the leg, which no view saw directly, now shows up as its own surface.

angle_views.jpg

front_views.jpg

Extra texture reads at runtime only happen on the few pixels that no view saw, and the bake does six more small renders. The octahedral layout is unchanged. Tested on both the WebGPU and WebGL 2 backends.
