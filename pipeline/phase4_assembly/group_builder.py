"""Phase 4: Group builder for nested elements."""

from pipeline.models import DetectedElement, ElementTree

def render_group(element: DetectedElement, target, rendered_ids: set, tree: ElementTree):
    """Create an empty GroupShape, render parent + all children INTO it."""
    # Note: add_group_shape() might require specific coordinates depending on python-pptx version,
    # but the python-pptx API add_group_shape() takes no arguments and auto-sizes based on content.
    try:
        grp = target.shapes.add_group_shape()
    except Exception as e:
        # Fallback if add_group_shape requires args or is not available
        grp = target
        
    from pipeline.phase4_assembly.assembler import render_shape, render_tile, render_element
    from pipeline.phase4_assembly.text_placer import render_text_lines

    # Render parent element into group (its own shape/tile if it has one)
    if element.processing_path == "reconstruct" and element.shape_type:
        render_shape(element, grp)
    elif element.processing_path == "crop" and element.tile_path:
        render_tile(element, grp)

    # Render children into group (sorted largest-first for z-order)
    for child in sorted(element.children, key=lambda e: e.bbox.area, reverse=True):
        render_element(child, grp, rendered_ids, tree)

    # Render text boxes always last (always on top within the group)
    render_text_lines(element, grp)
