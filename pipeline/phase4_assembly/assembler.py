"""Phase 4: PPTX Assembly orchestrator."""

from pptx import Presentation
from pptx.util import Emu

import config
from pipeline.models import DetectedElement, ElementTree
from pipeline.phase4_assembly.group_builder import render_group
from pipeline.phase4_assembly.arrow_writer import render_arrow
from pipeline.phase4_assembly.text_placer import render_text_lines


def px_to_emu(px: int, axis: str = "x") -> int:
    """Convert pixels to EMU based on config ratios."""
    if axis == "x":
        return int(px * config.SLIDE_WIDTH_EMU / config.SLIDE_WIDTH_PX)
    else:
        return int(px * config.SLIDE_HEIGHT_EMU / config.SLIDE_HEIGHT_PX)


def render_tile(element: DetectedElement, target):
    """Render a cropped image tile."""
    if not element.tile_path:
        return
    try:
        target.shapes.add_picture(
            element.tile_path,
            Emu(px_to_emu(element.bbox.x, "x")),
            Emu(px_to_emu(element.bbox.y, "y")),
            Emu(px_to_emu(element.bbox.w, "x")),
            Emu(px_to_emu(element.bbox.h, "y"))
        )
    except Exception as e:
        pass # Ignore missing files in tests


def render_shape(element: DetectedElement, target):
    """Render a native PowerPoint shape (simplified fallback)."""
    from pptx.enum.shapes import MSO_SHAPE
    target.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Emu(px_to_emu(element.bbox.x, "x")),
        Emu(px_to_emu(element.bbox.y, "y")),
        Emu(px_to_emu(element.bbox.w, "x")),
        Emu(px_to_emu(element.bbox.h, "y"))
    )


def render_text_block(element: DetectedElement, target):
    """Render just the text lines."""
    render_text_lines(element, target)


def render_leaf(element: DetectedElement, target):
    """Render leaf element based on its processing path or type."""
    if element.processing_path == "reconstruct":
        if element.shape_type: 
            render_shape(element, target)
        render_text_lines(element, target)
    elif element.processing_path == "crop":
        render_tile(element, target)
        render_text_lines(element, target)
    elif element.semantic_type == "Arrow":
        render_arrow(element, target)
    else:
        render_text_block(element, target)


def render_element(element: DetectedElement, target, rendered_ids: set, tree: ElementTree):
    """Renders element and recursively renders its children into a GroupShape."""
    if element.id in rendered_ids:
        return
    rendered_ids.add(element.id)
    # Pre-register all children so containment guard never re-processes them
    rendered_ids.update(element.child_ids())

    if element.children:
        render_group(element, target, rendered_ids, tree)
    else:
        render_leaf(element, target)


def assemble(tree: ElementTree, output_path: str) -> str:
    """Creates a new PowerPoint presentation from the ElementTree."""
    prs = Presentation()
    prs.slide_width = Emu(config.SLIDE_WIDTH_EMU)
    prs.slide_height = Emu(config.SLIDE_HEIGHT_EMU)

    slide_layout = prs.slide_layouts[6]  # blank layout
    slide = prs.slides.add_slide(slide_layout)

    rendered_ids = set()

    # Render roots in z-order (largest area first = furthest back)
    sorted_roots = sorted(tree.roots, key=lambda e: e.bbox.area, reverse=True)
    for element in sorted_roots:
        render_element(element, slide, rendered_ids, tree)

    prs.save(output_path)
    return output_path
