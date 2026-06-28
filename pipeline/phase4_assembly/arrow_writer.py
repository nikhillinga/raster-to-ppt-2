"""Phase 4: Arrow XML writer."""

from pptx.oxml.ns import qn
from pptx.util import Pt, Emu
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.dml.color import RGBColor

import config
from pipeline.models import DetectedElement


def render_arrow(element: DetectedElement, target):
    """
    Creates a STRAIGHT connector from element.arrow_start to element.arrow_end.
    Writes arrowhead directly to XML.
    """
    from pipeline.phase4_assembly.assembler import px_to_emu

    if not element.arrow_start or not element.arrow_end:
        return   # No geometry -> skip (do not draw fallback rectangle)

    x1, y1 = element.arrow_start
    x2, y2 = element.arrow_end

    connector = target.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        Emu(px_to_emu(x1, "x")), Emu(px_to_emu(y1, "y")),
        Emu(px_to_emu(x2, "x")), Emu(px_to_emu(y2, "y"))
    )
    connector.line.color.rgb = RGBColor(0, 0, 0)
    connector.line.width = Pt(config.ARROW_LINE_WIDTH_PT)

    # Write arrowhead directly to XML (python-pptx high-level API is a no-op for this)
    _set_arrowhead(connector)


def _set_arrowhead(connector, end="tail", arrow_type="triangle", width="med", length="med"):
    ln = connector.line._get_or_add_ln()
    tag = qn("a:tailEnd") if end == "tail" else qn("a:headEnd")
    existing = ln.find(tag)
    if existing is not None:
        ln.remove(existing)
    el = ln.makeelement(tag, {"type": arrow_type, "w": width, "len": length})
    ln.append(el)
