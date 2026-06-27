"""Prompt templates for Gemini VLM detection and refinement."""

DETECTION_PASS_1 = """
You are analyzing a slide image to identify all distinct visual elements.

Detect every element visible in this slide. For each element, return:
- bbox: {x, y, w, h} in pixels from the top-left corner of the image
- semantic_type: one of [Section, NestedSection, TextBlock, Header, Icon, DiagramShape, Arrow, Table, Chart, BackgroundArt]
- confidence: 0.0–1.0
- label: a short descriptive label (e.g. "Tier 1 container", "Year 1943 milestone card")
- correction_for: null (always null in Pass 1)

Semantic type definitions:
- Section: large container that holds other elements (e.g. a tier panel, a major card group)
- NestedSection: smaller container inside a Section (e.g. an individual milestone card, a sub-panel)
- TextBlock: body text, bullet lists, descriptions
- Header: slide title or major section heading
- Icon: small graphic symbol, logo, illustration (not reconstructible as a simple shape)
- DiagramShape: geometric shape with fill color (rectangle, circle, triangle, star, hexagon, pill/capsule)
- Arrow: directional connector between elements
- Table: tabular data grid
- Chart: data visualization (bar, pie, line chart)
- BackgroundArt: decorative element — texture, border art, illustrated corner decoration

Rules:
- Include ALL elements, including decorative ones
- Detect nested elements as separate entries (a Section AND its children are separate entries)
- For text inside a shape, the shape and its text are SEPARATE elements
- Do NOT merge overlapping elements — return them all
- Return ONLY valid JSON. No explanation text.

Return format:
{"elements": [...]}
"""

DETECTION_PASS_N = """
The image shows a slide with boxes already drawn (in magenta) around previously detected elements.

Your task:
1. Find any elements that do NOT yet have a magenta box around them — return these as new detections (correction_for: null)
2. Find any existing magenta boxes that look CLIPPED or TOO SMALL relative to the actual element they cover — return a corrected, larger box for these (set correction_for to the approximate bbox of the clipped box)

Use the same schema as before. Return ONLY valid JSON.
{"elements": [...]}
"""

COMPLEXITY_ASSESSMENT = """
Look at this cropped region of a slide. 

Assess whether this region is:
- SIMPLE: contains only plain colored shapes and/or clean text that could be accurately recreated as native PowerPoint shapes (rectangles, circles, triangles) with text overlaid
- COMPLEX: contains illustrations, photographs, detailed icons, textures, gradients, or content that cannot be faithfully recreated as simple geometric shapes

Return ONLY one word: SIMPLE or COMPLEX
"""
