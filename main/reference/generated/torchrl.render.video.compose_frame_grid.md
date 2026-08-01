# compose_frame_grid

torchrl.render.video.compose_frame_grid(*frames: Sequence[ndarray]*, *layout: Literal['single', 'grid', 'horizontal', 'vertical'] = 'grid'*) → ndarray[[source]](../../_modules/torchrl/render/video.html#compose_frame_grid)

Composes multiple frames into one RGB image.

Parameters:

- **frames** - Frames to compose.
- **layout** - `"grid"` (and `"single"`) tile the frames into a
near-square grid, `"horizontal"` composes one row, and
`"vertical"` composes one column.