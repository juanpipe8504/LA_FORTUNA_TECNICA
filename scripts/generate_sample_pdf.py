"""Generate ``docs/sample/Manual_Motor.pdf`` – a realistic sample technical manual.

The document is intentionally "multimodal" so it exercises the whole ingestion
pipeline: heading hierarchy (font size / bold), paragraphs with specifications,
ruled tables (detected by PyMuPDF ``find_tables``), a raster image (PNG rendered
programmatically) and a vector diagram, each with a "Figura N:" caption.

Only depends on PyMuPDF:

    backend/.venv/Scripts/python scripts/generate_sample_pdf.py   # Windows
    backend/.venv/bin/python scripts/generate_sample_pdf.py        # Linux/macOS
"""

from __future__ import annotations

import math
from pathlib import Path

import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "sample" / "Manual_Motor.pdf"

PAGE_W, PAGE_H = fitz.paper_size("a4")
MARGIN_X = 56
TOP = 70
BOTTOM = PAGE_H - 60
CONTENT_W = PAGE_W - 2 * MARGIN_X

REGULAR = "helv"
BOLD = "hebo"
ITALIC = "heit"

BLACK = (0, 0, 0)
DARK = (0.15, 0.15, 0.2)
GREY = (0.45, 0.45, 0.45)
ACCENT = (0.05, 0.3, 0.55)
LIGHT = (0.88, 0.91, 0.95)

MODEL = "MTX-450"
TITLE = f"Manual Técnico del Motor de Inducción Trifásico {MODEL}"


# --------------------------------------------------------------------------- #
# Minimal flow-layout engine
# --------------------------------------------------------------------------- #
class Writer:
    def __init__(self) -> None:
        self.doc = fitz.open()
        self.page: fitz.Page | None = None
        self.y = TOP

    # -- pages ---------------------------------------------------------------
    def new_page(self) -> None:
        self.page = self.doc.new_page(width=PAGE_W, height=PAGE_H)
        n = self.doc.page_count
        self.page.draw_line((MARGIN_X, 48), (PAGE_W - MARGIN_X, 48), color=ACCENT, width=0.8)
        self.page.insert_text((MARGIN_X, 42), f"Electromotores Andinos S.A. · Manual {MODEL} · Rev. 3.2",
                              fontname=REGULAR, fontsize=8, color=GREY)
        self.page.draw_line((MARGIN_X, PAGE_H - 42), (PAGE_W - MARGIN_X, PAGE_H - 42), color=GREY, width=0.4)
        self.page.insert_text((PAGE_W - MARGIN_X - 50, PAGE_H - 30), f"Página {n}",
                              fontname=REGULAR, fontsize=8, color=GREY)
        self.y = TOP

    def ensure(self, height: float) -> None:
        if self.page is None or self.y + height > BOTTOM:
            self.new_page()

    # -- text ----------------------------------------------------------------
    @staticmethod
    def wrap(text: str, fontname: str, fontsize: float, width: float) -> list[str]:
        lines: list[str] = []
        for paragraph in text.split("\n"):
            current = ""
            for word in paragraph.split():
                candidate = f"{current} {word}".strip()
                if fitz.get_text_length(candidate, fontname=fontname, fontsize=fontsize) <= width:
                    current = candidate
                else:
                    if current:
                        lines.append(current)
                    current = word
            lines.append(current)
        return lines

    def text(self, text: str, *, fontname: str = REGULAR, fontsize: float = 10.5, color=DARK,
             indent: float = 0, leading: float = 1.45, space_after: float = 8) -> None:
        width = CONTENT_W - indent
        lines = self.wrap(text, fontname, fontsize, width)
        line_h = fontsize * leading
        for line in lines:
            self.ensure(line_h)
            self.page.insert_text((MARGIN_X + indent, self.y + fontsize), line,
                                  fontname=fontname, fontsize=fontsize, color=color)
            self.y += line_h
        self.y += space_after

    def heading(self, text: str, level: int) -> None:
        size = {1: 17, 2: 13.5, 3: 11.5}[level]
        self.ensure(size * 2 + 40)  # keep heading with the next lines
        self.y += 10 if level == 1 else 6
        self.text(text, fontname=BOLD, fontsize=size, color=ACCENT if level < 3 else DARK,
                  leading=1.3, space_after=6)

    def bullets(self, items: list[str], numbered: bool = False) -> None:
        for i, item in enumerate(items, 1):
            marker = f"{i}." if numbered else "•"
            lines = self.wrap(item, REGULAR, 10.5, CONTENT_W - 18)
            self.ensure(15)
            self.page.insert_text((MARGIN_X + 4, self.y + 10.5), marker, fontname=REGULAR, fontsize=10.5,
                                  color=DARK)
            for line in lines:
                self.ensure(15)
                self.page.insert_text((MARGIN_X + 18, self.y + 10.5), line, fontname=REGULAR,
                                      fontsize=10.5, color=DARK)
                self.y += 15
            self.y += 2
        self.y += 6

    def caption(self, text: str) -> None:
        self.text(text, fontname=ITALIC, fontsize=9.5, color=GREY, space_after=12)

    # -- tables ----------------------------------------------------------------
    def table(self, header: list[str], rows: list[list[str]], widths: list[float]) -> None:
        scale = CONTENT_W / sum(widths)
        widths = [w * scale for w in widths]
        fontsize, pad, line_h = 9, 4, 11.5

        def row_height(cells: list[str], font: str) -> float:
            return max(len(self.wrap(c, font, fontsize, w - 2 * pad)) for c, w in zip(cells, widths)) * line_h + 2 * pad

        def draw_row(cells: list[str], font: str, fill) -> None:
            h = row_height(cells, font)
            x = MARGIN_X
            for cell, w in zip(cells, widths):
                rect = fitz.Rect(x, self.y, x + w, self.y + h)
                self.page.draw_rect(rect, color=GREY, fill=fill, width=0.6)
                ty = self.y + pad + fontsize
                for line in self.wrap(cell, font, fontsize, w - 2 * pad):
                    self.page.insert_text((x + pad, ty), line, fontname=font, fontsize=fontsize, color=BLACK)
                    ty += line_h
                x += w
            self.y += h

        self.ensure(row_height(header, BOLD) + row_height(rows[0], REGULAR))
        draw_row(header, BOLD, LIGHT)
        for row in rows:
            if self.y + row_height(row, REGULAR) > BOTTOM:
                self.new_page()
                draw_row(header, BOLD, LIGHT)
            draw_row(row, REGULAR, None)
        self.y += 8

    # -- figures -------------------------------------------------------------
    def raster_figure(self, png: bytes, width: float, height: float, caption: str) -> None:
        self.ensure(height + 40)
        x0 = MARGIN_X + (CONTENT_W - width) / 2
        rect = fitz.Rect(x0, self.y, x0 + width, self.y + height)
        self.page.insert_image(rect, stream=png)
        self.y += height + 8
        self.caption(caption)

    def reserve(self, height: float) -> fitz.Rect:
        self.ensure(height + 40)
        rect = fitz.Rect(MARGIN_X, self.y, MARGIN_X + CONTENT_W, self.y + height)
        self.y += height + 8
        return rect


# --------------------------------------------------------------------------- #
# Figure 1 - exploded view, drawn as vectors on a scratch page then rasterized
# --------------------------------------------------------------------------- #
def render_exploded_view_png() -> bytes:
    w, h = 460, 250
    scratch = fitz.open()
    p = scratch.new_page(width=w, height=h)
    p.draw_rect(p.rect, color=None, fill=(1, 1, 1))
    steel, copper, dark = (0.62, 0.66, 0.72), (0.80, 0.48, 0.20), (0.2, 0.22, 0.26)
    cy = 120
    # Housing (stator frame) with cooling fins
    p.draw_rect(fitz.Rect(150, 60, 300, 180), color=dark, fill=steel, width=1.2)
    for i in range(9):
        x = 158 + i * 16
        p.draw_rect(fitz.Rect(x, 50, x + 6, 60), color=dark, fill=steel, width=0.6)
        p.draw_rect(fitz.Rect(x, 180, x + 6, 190), color=dark, fill=steel, width=0.6)
    # Stator windings
    p.draw_rect(fitz.Rect(165, 78, 285, 95), color=dark, fill=copper, width=0.6)
    p.draw_rect(fitz.Rect(165, 145, 285, 162), color=dark, fill=copper, width=0.6)
    # Terminal box
    p.draw_rect(fitz.Rect(200, 30, 250, 50), color=dark, fill=(0.35, 0.38, 0.45), width=1)
    # Rotor + shaft (pulled out to the left)
    p.draw_rect(fitz.Rect(30, cy - 6, 140, cy + 6), color=dark, fill=(0.75, 0.75, 0.78), width=1)
    p.draw_rect(fitz.Rect(70, cy - 25, 130, cy + 25), color=dark, fill=(0.55, 0.58, 0.62), width=1)
    for i in range(6):
        x = 74 + i * 10
        p.draw_line((x, cy - 25), (x + 4, cy + 25), color=dark, width=0.5)
    # Bearings
    for x in (55, 145):
        p.draw_circle((x, cy), 9, color=dark, fill=(0.9, 0.9, 0.9), width=1)
        p.draw_circle((x, cy), 4, color=dark, fill=dark, width=0.5)
    # End shield, fan and fan cover (to the right)
    p.draw_rect(fitz.Rect(315, 70, 325, 170), color=dark, fill=steel, width=1)
    for k in range(8):
        a = k * math.pi / 4
        p.draw_line((360, cy), (360 + 30 * math.cos(a), cy + 30 * math.sin(a)), color=dark, width=3)
    p.draw_circle((360, cy), 8, color=dark, fill=dark)
    p.draw_rect(fitz.Rect(400, 75, 440, 165), color=dark, fill=None, width=1.2)
    for i in range(6):
        p.draw_line((405 + i * 6, 80), (405 + i * 6, 160), color=GREY, width=0.6)
    # Callout labels
    labels = [
        ((85, cy - 25), (60, 215), "1 Rotor"),
        ((55, cy + 9), (20, 235), "2 Rodamiento"),
        ((225, 70), (190, 215), "3 Estator"),
        ((225, 30), (260, 18), "4 Caja de bornes"),
        ((320, 170), (290, 235), "5 Escudo"),
        ((360, cy + 30), (345, 215), "6 Ventilador"),
        ((420, 165), (400, 235), "7 Tapa"),
    ]
    for (ax, ay), (tx, ty), label in labels:
        p.draw_line((ax, ay), (tx + 15, ty - 9), color=(0.7, 0.1, 0.1), width=0.6)
        p.insert_text((tx, ty), label, fontname=BOLD, fontsize=8, color=(0.7, 0.1, 0.1))
    pix = p.get_pixmap(dpi=200, alpha=False)
    return pix.tobytes("png")


# --------------------------------------------------------------------------- #
# Figure 2 - wiring diagram drawn directly as page vectors
# --------------------------------------------------------------------------- #
def draw_wiring_diagram(page: fitz.Page, area: fitz.Rect) -> None:
    page.draw_rect(area, color=GREY, width=0.5)
    half = area.width / 2

    def terminal_block(ox: float, title: str, star: bool) -> None:
        page.insert_text((ox + 20, area.y0 + 20), title, fontname=BOLD, fontsize=10, color=ACCENT)
        xs = [ox + 45 + i * 60 for i in range(3)]
        top, bottom = area.y0 + 60, area.y0 + 130
        names_top, names_bottom = ["U1", "V1", "W1"], ["W2", "U2", "V2"]
        for x, nt, nb in zip(xs, names_top, names_bottom):
            for y, name in ((top, nt), (bottom, nb)):
                page.draw_circle((x, y), 7, color=DARK, fill=(1, 1, 1), width=1)
                page.insert_text((x + 10, y + 3), name, fontname=REGULAR, fontsize=8, color=DARK)
            # internal winding (zig-zag) between top and bottom terminal
            pts = [(x, top + 7)]
            for k in range(1, 8):
                pts.append((x + (5 if k % 2 else -5), top + 7 + k * 7))
            pts.append((x, bottom - 7))
            page.draw_polyline(pts, color=(0.8, 0.45, 0.15), width=1)
        # supply lines L1, L2, L3
        for x, line in zip(xs, ("L1", "L2", "L3")):
            page.draw_line((x, area.y0 + 32), (x, top - 7), color=BLACK, width=1.2)
            page.insert_text((x - 6, area.y0 + 30), line, fontname=BOLD, fontsize=8, color=BLACK)
        # bridges
        if star:
            page.draw_rect(fitz.Rect(xs[0] - 9, bottom + 12, xs[2] + 9, bottom + 20), color=BLACK,
                           fill=(0.85, 0.85, 0.85), width=1)
            for x in xs:
                page.draw_line((x, bottom + 7), (x, bottom + 12), color=BLACK, width=1.2)
            page.insert_text((ox + 50, bottom + 38), "Puente común W2-U2-V2 (Y)", fontname=REGULAR,
                             fontsize=8, color=DARK)
        else:
            for x in xs:
                page.draw_rect(fitz.Rect(x - 22, top - 4, x - 16, bottom + 4), color=BLACK,
                               fill=(0.85, 0.85, 0.85), width=1)
                page.draw_line((x - 16, top), (x - 7, top), color=BLACK, width=1.2)
                page.draw_line((x - 16, bottom), (x - 7, bottom), color=BLACK, width=1.2)
            page.insert_text((ox + 40, bottom + 38), "Puentes verticales U1-W2, V1-U2, W1-V2 (D)",
                             fontname=REGULAR, fontsize=8, color=DARK)

    terminal_block(area.x0, "Conexión estrella (Y) - 690 V", star=True)
    page.draw_line((area.x0 + half, area.y0 + 10), (area.x0 + half, area.y1 - 10), color=GREY, width=0.4,
                   dashes="[3 3] 0")
    terminal_block(area.x0 + half, "Conexión triángulo (D) - 400 V", star=False)
    page.insert_text((area.x0 + 20, area.y1 - 10), "PE: borne de tierra en la caja de bornes (sección mínima 16 mm2)",
                     fontname=ITALIC, fontsize=8, color=GREY)


# --------------------------------------------------------------------------- #
# Document content
# --------------------------------------------------------------------------- #
def build() -> fitz.Document:
    w = Writer()
    w.new_page()

    # Title block
    w.y += 20
    w.text(TITLE, fontname=BOLD, fontsize=20, color=ACCENT, leading=1.25, space_after=4)
    w.text("Potencia nominal 45 kW · 400/690 V · 50 Hz · IE3 · Carcasa IEC 225M", fontname=REGULAR,
           fontsize=11, color=GREY, space_after=18)

    w.heading("1. Introducción", 1)
    w.text(
        f"Este manual describe la instalación, operación, mantenimiento y diagnóstico de averías del motor de "
        f"inducción trifásico {MODEL}, de rotor en jaula de ardilla, fabricado por Electromotores Andinos S.A. "
        "El motor está diseñado para accionar bombas centrífugas, ventiladores, compresores y cintas "
        "transportadoras en servicio continuo S1."
    )
    w.heading("1.1 Alcance", 2)
    w.text(
        "La información aplica a todas las variantes del modelo fabricadas a partir del número de serie "
        "MTX450-2023-0001, con montaje B3 (patas) y B35 (patas y brida). Para variantes con protección contra "
        "explosiones (Ex d) consulte el suplemento ATEX correspondiente."
    )
    w.heading("1.2 Instrucciones de seguridad", 2)
    w.text(
        "Antes de cualquier intervención desconecte y bloquee la alimentación eléctrica (procedimiento LOTO) y "
        "espere al menos 5 minutos para la descarga de los condensadores de compensación. Solo personal "
        "cualificado según EN 50110 debe realizar trabajos en la caja de bornes."
    )
    w.bullets([
        "Utilice siempre guantes dieléctricos clase 0 y gafas de protección.",
        "No ponga el motor en marcha sin la tapa del ventilador instalada.",
        "La temperatura de la carcasa puede superar los 80 °C durante el funcionamiento normal.",
        "Verifique la puesta a tierra (PE) antes de energizar el equipo.",
    ])

    w.new_page()  # keep the specifications table on a single page
    w.heading("2. Especificaciones técnicas", 1)
    w.text(
        f"La Tabla 1 resume los datos nominales del {MODEL}. Los valores se refieren a una temperatura ambiente de "
        "40 °C y una altitud máxima de 1000 m sobre el nivel del mar. La eficiencia corresponde a la clase IE3 "
        "según IEC 60034-30-1."
    )
    w.text("Tabla 1: Datos nominales del motor MTX-450", fontname=BOLD, fontsize=9.5, color=DARK, space_after=4)
    w.table(
        ["Parámetro", "Valor", "Unidad"],
        [
            ["Potencia nominal", "45", "kW"],
            ["Tensión nominal (D / Y)", "400 / 690", "V"],
            ["Corriente nominal a 400 V", "80,5", "A"],
            ["Frecuencia", "50", "Hz"],
            ["Velocidad nominal", "1480", "rpm"],
            ["Par nominal", "290", "N·m"],
            ["Factor de potencia (cos phi)", "0,86", "-"],
            ["Eficiencia a plena carga", "94,2", "%"],
            ["Relación corriente de arranque Ia/In", "7,2", "-"],
            ["Clase de aislamiento / calentamiento", "F / B", "-"],
            ["Grado de protección", "IP55", "-"],
            ["Peso", "310", "kg"],
        ],
        [3, 1.6, 1],
    )
    w.heading("2.1 Condiciones ambientales", 2)
    w.text(
        "El motor puede operar entre -20 °C y +40 °C sin reducción de potencia. Por encima de 40 °C aplique un "
        "factor de reducción del 3 % por cada 5 °C adicionales. La humedad relativa no debe superar el 95 % sin "
        "condensación; en ambientes húmedos se recomienda activar la resistencia anticondensación de 50 W."
    )

    w.new_page()  # keep each figure on the same page as the text that references it
    w.heading("3. Descripción del equipo", 1)
    w.text(
        "El motor está compuesto por un estator bobinado alojado en una carcasa de fundición con aletas de "
        "refrigeración, un rotor de jaula de aluminio inyectado montado sobre el eje, dos rodamientos de bolas y "
        "un ventilador radial de refrigeración (IC 411). La Figura 1 muestra una vista explosionada de los "
        "componentes principales."
    )
    w.raster_figure(
        render_exploded_view_png(), 400, 217,
        "Figura 1: Vista explosionada del motor MTX-450: (1) rotor, (2) rodamiento, (3) estator, "
        "(4) caja de bornes, (5) escudo trasero, (6) ventilador, (7) tapa del ventilador.",
    )
    w.heading("3.1 Componentes principales", 2)
    w.bullets([
        "Rodamiento lado accionamiento (LA): 6313-2Z/C3; lado opuesto (LOA): 6212-2Z/C3.",
        "Ventilador radial de polipropileno reforzado, bidireccional.",
        "Caja de bornes orientable 4 × 90°, con 6 bornes M10 y borne de tierra.",
        "Sensores PTC de 150 °C en cada fase del bobinado (protección térmica).",
    ])

    w.new_page()
    w.heading("4. Instalación y conexión eléctrica", 1)
    w.text(
        "Monte el motor sobre una base rígida y nivelada. La desalineación máxima admisible del acoplamiento es de "
        "0,05 mm radial y 0,05 mm/100 mm angular. Apriete los pernos de anclaje M16 a 210 N·m."
    )
    w.heading("4.1 Esquema de conexión", 2)
    w.text(
        "Para una red de 400 V conecte el bobinado en triángulo (D); para una red de 690 V o para arranque "
        "estrella-triángulo conecte en estrella (Y). La Figura 2 muestra la disposición de los puentes en la "
        "placa de bornes."
    )
    area = w.reserve(200)
    draw_wiring_diagram(w.page, area)
    w.caption("Figura 2: Esquema de conexión de la placa de bornes en estrella (Y) y en triángulo (D).")
    w.heading("4.2 Puesta en marcha", 2)
    w.bullets([
        "Mida la resistencia de aislamiento con un megóhmetro de 500 V c.c.: el valor mínimo es 10 MOhm a 25 °C.",
        "Compruebe el sentido de giro con el motor desacoplado; invierta dos fases si es necesario.",
        "Verifique que la corriente en vacío esté entre el 30 % y el 40 % de la corriente nominal.",
        "Controle vibraciones: el valor máximo admisible es 2,8 mm/s RMS (ISO 10816-3, zona B).",
    ], numbered=True)

    w.new_page()
    w.heading("5. Mantenimiento", 1)
    w.text(
        "Un mantenimiento preventivo adecuado prolonga la vida útil del bobinado y de los rodamientos. Los "
        "intervalos de la Tabla 2 corresponden a servicio continuo con temperatura ambiente de hasta 40 °C; "
        "redúzcalos a la mitad en ambientes polvorientos o con vibraciones elevadas."
    )
    w.text("Tabla 2: Plan de mantenimiento preventivo", fontname=BOLD, fontsize=9.5, color=DARK, space_after=4)
    w.table(
        ["Tarea", "Intervalo", "Observaciones"],
        [
            ["Inspección visual y limpieza de aletas", "Mensual", "Aire comprimido seco a máx. 2 bar"],
            ["Medición de vibraciones y temperatura", "Mensual", "Registrar tendencia"],
            ["Relubricación de rodamientos", "4000 h", "Grasa de litio NLGI 3; 20 g (LA) y 15 g (LOA)"],
            ["Medición de aislamiento", "6 meses", "Mínimo 10 MOhm a 500 V c.c."],
            ["Revisión de apriete en caja de bornes", "12 meses", "Par 20 N·m en bornes M10"],
            ["Sustitución de rodamientos", "40 000 h", "O antes si la vibración supera 4,5 mm/s"],
        ],
        [2.6, 1.1, 2.6],
    )
    w.heading("5.1 Procedimiento de relubricación de rodamientos", 2)
    w.bullets([
        "Con el motor en marcha (si es seguro) limpie los engrasadores y retire el tapón de purga de grasa.",
        "Inyecte 20 g de grasa en el lado LA y 15 g en el lado LOA utilizando una bomba manual.",
        "Mantenga el motor en marcha 30 minutos para expulsar la grasa sobrante por el orificio de purga.",
        "Vuelva a colocar el tapón de purga y registre la fecha y las horas de servicio.",
    ], numbered=True)
    w.text(
        "Advertencia: el exceso de grasa provoca sobrecalentamiento de los rodamientos. No mezcle grasas de "
        "distinta base (litio y poliurea son incompatibles).",
        fontname=BOLD, fontsize=10, color=(0.65, 0.1, 0.1),
    )

    w.heading("6. Solución de problemas", 1)
    w.text("La Tabla 3 enumera las averías más frecuentes, sus causas probables y la acción correctiva recomendada.")
    w.text("Tabla 3: Diagnóstico de averías", fontname=BOLD, fontsize=9.5, color=DARK, space_after=4)
    w.table(
        ["Síntoma", "Causa probable", "Acción correctiva"],
        [
            ["El motor no arranca", "Falta de una fase o fusible fundido", "Verificar tensión en L1, L2, L3 y fusibles"],
            ["Disparo del relé térmico", "Sobrecarga o tensión baja (< 360 V)",
             "Medir corriente; comprobar carga y tensión de red"],
            ["Ruido en rodamientos", "Falta de grasa o rodamiento dañado", "Relubricar; sustituir si persiste"],
            ["Vibración excesiva", "Desalineación o desequilibrio del acoplamiento",
             "Realinear (máx. 0,05 mm); equilibrar"],
            ["Sobrecalentamiento de la carcasa", "Aletas obstruidas o ventilador dañado",
             "Limpiar aletas; revisar ventilador y tapa"],
            ["Disparo de sondas PTC", "Temperatura del bobinado > 150 °C",
             "Reducir carga; revisar refrigeración y arranques/hora (máx. 6)"],
        ],
        [2, 2.3, 2.6],
    )
    w.heading("6.1 Asistencia técnica", 2)
    w.text(
        "Si el problema persiste, contacte con el servicio técnico indicando el número de serie de la placa de "
        "características, las horas de funcionamiento y los valores medidos de corriente, tensión y vibración. "
        "La garantía de 24 meses queda anulada si el motor se abre sin autorización del fabricante."
    )
    return w.doc


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = build()
    doc.set_metadata({
        "title": TITLE,
        "author": "Electromotores Andinos S.A. (ficticio)",
        "subject": "Documento de ejemplo para el sistema RAG multimodal",
    })
    doc.save(OUTPUT, garbage=4, deflate=True)
    print(f"Wrote {OUTPUT} ({doc.page_count} pages)")


if __name__ == "__main__":
    main()
