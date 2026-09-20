"""Build the designed Local and Databricks parameter guide PDF."""

from __future__ import annotations

import argparse
import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tools" / "pdf_sources" / "local_and_databricks_parameters.md"
LOGO = ROOT / "dashboard" / "assets" / "ubuntu_mark.png"
DEFAULT_OUTPUT = ROOT / "docs" / "local_and_databricks_parameters.pdf"

PAGE_WIDTH, PAGE_HEIGHT = landscape(letter)
CREAM = colors.HexColor("#FFF9F4")
PAPER = colors.HexColor("#FFFFFF")
INK = colors.HexColor("#2D2430")
MUTED = colors.HexColor("#6B626B")
ORANGE = colors.HexColor("#E95420")
ORANGE_LIGHT = colors.HexColor("#FBE4D9")
AUBERGINE = colors.HexColor("#77216F")
AUBERGINE_DARK = colors.HexColor("#45143F")
AUBERGINE_LIGHT = colors.HexColor("#EBDDE9")
GOLD = colors.HexColor("#F5A623")
GREEN = colors.HexColor("#2D7D46")
GREEN_LIGHT = colors.HexColor("#E4F2E8")
GRID = colors.HexColor("#DDD4D9")


def ascii_text(value: str) -> str:
    replacements = {
        "\u2014": "N/A",
        "\u2013": "-",
        "\u2011": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u2192": "->",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = value.replace("\\|", "|")
    value = re.sub(r"\*\*(.*?)\*\*", r"\1", value)
    value = value.replace("`", "")
    return value.strip()


def extract_markdown_table(markdown: str, heading: str) -> tuple[list[str], list[list[str]]]:
    marker = f"### {heading}"
    if marker not in markdown:
        marker = f"## {heading}"
    if marker not in markdown:
        marker = heading
    start = markdown.index(marker)
    lines = markdown[start:].splitlines()[1:]
    table_lines: list[str] = []
    started = False
    for line in lines:
        if line.startswith("|"):
            started = True
            table_lines.append(line)
        elif started:
            break
    if len(table_lines) < 2:
        raise ValueError(f"No table found after heading: {heading}")

    def split_row(line: str) -> list[str]:
        placeholder = "__ESCAPED_PIPE__"
        protected = line.replace("\\|", placeholder)
        cells = [cell.strip().replace(placeholder, "|") for cell in protected.strip("|").split("|")]
        return [ascii_text(cell) for cell in cells]

    header = split_row(table_lines[0])
    rows = [split_row(line) for line in table_lines[2:]]
    return header, rows


class GuideDocTemplate(SimpleDocTemplate):
    def afterFlowable(self, flowable: Flowable) -> None:
        if isinstance(flowable, Paragraph) and flowable.style.name == "SectionPageTitle":
            title = flowable.getPlainText()
            key = f"section-{self.page}-{re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title, key, level=0, closed=False)


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_kicker": ParagraphStyle(
            "CoverKicker",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=colors.HexColor("#FFD7C4"),
            tracking=1.2,
            spaceAfter=12,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=31,
            leading=34,
            textColor=colors.white,
            spaceAfter=14,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=14,
            leading=20,
            textColor=colors.HexColor("#F5EAF3"),
            spaceAfter=24,
        ),
        "section": ParagraphStyle(
            "SectionPageTitle",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=AUBERGINE,
            spaceAfter=8,
        ),
        "lede": ParagraphStyle(
            "Lede",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=MUTED,
            spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=INK,
            spaceBefore=5,
            spaceAfter=7,
        ),
        "h3_orange": ParagraphStyle(
            "H3Orange",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=ORANGE,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13,
            textColor=INK,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=INK,
        ),
        "small_bold": ParagraphStyle(
            "SmallBold",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=10,
            textColor=INK,
        ),
        "small_code": ParagraphStyle(
            "SmallCode",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=6.8,
            leading=9,
            textColor=INK,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.4,
            leading=10.2,
            textColor=INK,
            leftIndent=6,
            rightIndent=6,
            spaceAfter=0,
        ),
        "card_title": ParagraphStyle(
            "CardTitle",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.white,
            alignment=TA_LEFT,
        ),
        "card_body": ParagraphStyle(
            "CardBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=INK,
        ),
        "number": ParagraphStyle(
            "Number",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=20,
            textColor=colors.white,
            alignment=TA_CENTER,
        ),
        "center": ParagraphStyle(
            "Center",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=INK,
            alignment=TA_CENTER,
        ),
    }


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(ascii_text(text)).replace("\n", "<br/>"), style)


def rich_para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def code_box(code: str, style: ParagraphStyle, width: float) -> Table:
    code = escape(ascii_text(code)).replace(" ", "&nbsp;").replace("\n", "<br/>")
    table = Table([[Paragraph(code, style)]], colWidths=[width])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F3EEF2")),
                ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#CDBFCA")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def callout(title: str, body: str, palette: str, s: dict[str, ParagraphStyle], width: float) -> Table:
    if palette == "orange":
        bar, bg = ORANGE, ORANGE_LIGHT
    elif palette == "green":
        bar, bg = GREEN, GREEN_LIGHT
    else:
        bar, bg = AUBERGINE, AUBERGINE_LIGHT
    content = [
        rich_para(f'<font color="{bar.hexval()}"><b>{escape(ascii_text(title))}</b></font>', s["h2"]),
        para(body, s["body"]),
    ]
    table = Table([["", content]], colWidths=[7, width - 7])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), bar),
                ("BACKGROUND", (1, 0), (1, 0), bg),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (0, 0), 0),
                ("RIGHTPADDING", (0, 0), (0, 0), 0),
                ("TOPPADDING", (1, 0), (1, 0), 9),
                ("BOTTOMPADDING", (1, 0), (1, 0), 8),
                ("LEFTPADDING", (1, 0), (1, 0), 11),
                ("RIGHTPADDING", (1, 0), (1, 0), 11),
                ("BOX", (0, 0), (-1, -1), 0.5, bar),
            ]
        )
    )
    return table


def route_card(
    title: str,
    subtitle: str,
    steps: list[str],
    color: colors.Color,
    s: dict[str, ParagraphStyle],
    width: float,
) -> Table:
    rows: list[list[object]] = [[rich_para(escape(title), s["card_title"]), ""]]
    rows.append([rich_para(f'<font color="{color.hexval()}"><b>{escape(subtitle)}</b></font>', s["h2"]), ""])
    for index, step in enumerate(steps, 1):
        number = Table([[Paragraph(str(index), s["number"])]], colWidths=[25], rowHeights=[25])
        number.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), color), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        rows.append([number, para(step, s["card_body"])])
    table = Table(rows, colWidths=[35, width - 35])
    table.setStyle(
        TableStyle(
            [
                ("SPAN", (0, 0), (1, 0)),
                ("SPAN", (0, 1), (1, 1)),
                ("BACKGROUND", (0, 0), (-1, 0), color),
                ("BACKGROUND", (0, 1), (-1, -1), PAPER),
                ("BOX", (0, 0), (-1, -1), 1.1, color),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, 0), 14),
                ("RIGHTPADDING", (0, 0), (-1, 0), 14),
                ("TOPPADDING", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 11),
                ("TOPPADDING", (0, 1), (-1, 1), 10),
                ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
                ("LEFTPADDING", (0, 2), (0, -1), 12),
                ("RIGHTPADDING", (0, 2), (0, -1), 5),
                ("TOPPADDING", (0, 2), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 2), (-1, -1), 7),
                ("LINEBELOW", (0, 2), (-1, -2), 0.35, GRID),
            ]
        )
    )
    return table


def reference_table(
    header: list[str],
    rows: list[list[str]],
    s: dict[str, ParagraphStyle],
    widths: list[float],
) -> Table:
    converted: list[list[Paragraph]] = []
    converted.append([rich_para(f'<font color="#FFFFFF"><b>{escape(cell)}</b></font>', s["small"]) for cell in header])
    for row in rows:
        converted.append(
            [
                para(row[0], s["small_bold"]),
                *[para(cell, s["small_code"] if index in (1, 2) else s["small"]) for index, cell in enumerate(row[1:], 1)],
            ]
        )
    table = Table(converted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    style_commands = [
        ("BACKGROUND", (0, 0), (-1, 0), AUBERGINE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_index in range(1, len(converted)):
        if row_index % 2 == 0:
            style_commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#FAF6F8")))
    if len(header) >= 3:
        style_commands.extend(
            [
                ("BACKGROUND", (1, 1), (1, -1), colors.HexColor("#FFF7F2")),
                ("BACKGROUND", (2, 1), (2, -1), colors.HexColor("#F8F0F7")),
            ]
        )
    table.setStyle(TableStyle(style_commands))
    return table


def draw_page(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFillColor(CREAM)
    canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
    if doc.page == 1:
        canvas.setFillColor(AUBERGINE_DARK)
        canvas.rect(0, 0, PAGE_WIDTH, PAGE_HEIGHT, stroke=0, fill=1)
        canvas.setFillColor(ORANGE)
        canvas.circle(PAGE_WIDTH - 35, PAGE_HEIGHT - 35, 145, stroke=0, fill=1)
        canvas.setFillColor(GOLD)
        canvas.circle(PAGE_WIDTH - 42, 42, 74, stroke=0, fill=1)
        canvas.setFillColor(AUBERGINE)
        canvas.circle(PAGE_WIDTH - 170, PAGE_HEIGHT - 125, 70, stroke=0, fill=1)
    else:
        canvas.setFillColor(AUBERGINE)
        canvas.rect(0, PAGE_HEIGHT - 25, PAGE_WIDTH, 25, stroke=0, fill=1)
        canvas.setFillColor(ORANGE)
        canvas.rect(0, 0, PAGE_WIDTH, 4, stroke=0, fill=1)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(colors.white)
        canvas.drawString(36, PAGE_HEIGHT - 17, "UBUNTU DIALOGUE CORPUS  /  EXECUTION PARAMETERS")
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(PAGE_WIDTH - 36, 16, f"{doc.page - 1}")
    canvas.restoreState()


def build_pdf(output_path: Path) -> None:
    markdown = SOURCE.read_text(encoding="utf-8")
    s = styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    doc = GuideDocTemplate(
        str(output_path),
        pagesize=landscape(letter),
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
        topMargin=0.53 * inch,
        bottomMargin=0.36 * inch,
        title="Local and Databricks Parameters",
        author="Lily Wool",
        subject="Ubuntu Dialogue Corpus execution reference",
    )
    story: list[Flowable] = []
    content_width = PAGE_WIDTH - doc.leftMargin - doc.rightMargin

    # Cover
    story.extend(
        [
            Spacer(1, 0.5 * inch),
            Image(str(LOGO), width=0.92 * inch, height=0.92 * inch),
            Spacer(1, 0.25 * inch),
            Paragraph("UBUNTU DIALOGUE CORPUS", s["cover_kicker"]),
            Paragraph("Local + Databricks<br/>Parameter Guide", s["cover_title"]),
            Paragraph(
                "A visual operating manual for moving from raw dialogue to Silver features and Gold summaries.",
                s["cover_subtitle"],
            ),
            Table(
                [["LOCAL", "DATABRICKS", "SILVER", "GOLD"]],
                colWidths=[1.28 * inch] * 4,
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#5D1A56")),
                        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
                        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#A66BA0")),
                        ("INNERGRID", (0, 0), (-1, -1), 0.7, colors.HexColor("#A66BA0")),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
            ),
            Spacer(1, 0.4 * inch),
            Paragraph("ONE PIPELINE. TWO ENVIRONMENTS. FOUR ENTRY POINTS.", s["cover_kicker"]),
            PageBreak(),
        ]
    )

    # Route map
    story.extend(
        [
            Paragraph("Choose your route", s["section"]),
            Paragraph(
                "The data layers are the same in both environments. What changes is how each stage is launched and how its parameters are entered.",
                s["lede"],
            ),
        ]
    )
    gap = 14
    card_width = (content_width - gap) / 2
    local_card = route_card(
        "LOCAL",
        "Two PowerShell commands",
        [
            "Raw CSV -> Silver CSV with python -m pipeline.pipeline",
            "Silver CSV -> one selected Gold CSV with python -m pipeline.aggregation",
            "Use ordinary command-line flags. No JSON brackets.",
        ],
        ORANGE,
        s,
        card_width,
    )
    db_card = route_card(
        "DATABRICKS",
        "Two dependent job tasks",
        [
            "Bronze -> Delta Silver with bronze_to_silver_ubuntu.py",
            "Delta Silver -> one selected Delta Gold table with silver_to_gold_ubuntu.py",
            "Use a JSON array of strings in each task's Parameters box.",
        ],
        AUBERGINE,
        s,
        card_width,
    )
    story.append(Table([[local_card, db_card]], colWidths=[card_width, card_width], hAlign="LEFT", style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (0, 0), gap / 2), ("LEFTPADDING", (1, 0), (1, 0), gap / 2), ("RIGHTPADDING", (1, 0), (1, 0), 0)])))
    story.extend(
        [
            Spacer(1, 16),
            callout(
                "The order never changes",
                "Silver performs row-level cleaning and enrichment. Gold aggregates that Silver output into one selected analytical grain. Run Silver first, then run Gold once for each output you want.",
                "green",
                s,
                content_width,
            ),
            PageBreak(),
        ]
    )

    # Local quick start
    local_silver = '''# Raw CSV -> Silver CSV
& .\\.venv\\python.exe -m pipeline.pipeline `
    ".\\data\\dialogueText_196.csv" `
    --sample-size 10000 `
    --sample-seed 42 `
    --residual-policy reviewed `
    --sentiment vader `
    --output-dir ".\\outputs\\silver_10k"'''
    local_gold = '''# Silver CSV -> conversation Gold CSV
& .\\.venv\\python.exe -m pipeline.aggregation `
    ".\\outputs\\silver_10k\\silver_ubuntu_dialogue.csv" `
    --gold-level conversation `
    --output-dir ".\\outputs\\gold_10k"'''
    key_prompt = """$openAiSecureKey = Read-Host "Paste your OpenAI API key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $openAiSecureKey).Password
Remove-Variable openAiSecureKey

try {
    # Run Silver here with --residual-policy api.
}
finally {
    Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue
}"""
    story.extend(
        [
            Paragraph("Local PowerShell", s["section"]),
            Paragraph(
                "Open PowerShell in the repository root. Flags follow the module name; a backtick continues the same command.",
                s["lede"],
            ),
            Table(
                [
                    [
                        [Paragraph("1  BUILD SILVER", s["h3_orange"]), code_box(local_silver, s["code"], 333)],
                        [Paragraph("2  BUILD GOLD", s["h3_orange"]), code_box(local_gold, s["code"], 333)],
                    ]
                ],
                colWidths=[content_width / 2, content_width / 2],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 8), ("LEFTPADDING", (1, 0), (1, 0), 8), ("RIGHTPADDING", (1, 0), (1, 0), 0)]),
            ),
            Spacer(1, 15),
            callout(
                "Using the OpenAI residual-classification option",
                "The pipeline does not open its own key prompt. Run the secure prompt below in the same PowerShell session, place the Silver command inside the try block, and let finally remove the key.",
                "orange",
                s,
                content_width,
            ),
            Spacer(1, 8),
            code_box(key_prompt, s["code"], content_width),
            PageBreak(),
        ]
    )

    # Databricks quick start
    setup_rows = []
    setup_steps = [
        "Upload source data to a Unity Catalog volume and create Bronze, or use a supported CSV/Delta path.",
        "Add the repository as a Databricks Git folder under Workspace.",
        "Create a job with a Python script task for Bronze-to-Silver.",
        "Paste a JSON array of strings into that task's Parameters box.",
        "Add a dependent Python script task for Silver-to-Gold with its own Parameters JSON.",
    ]
    for i, step in enumerate(setup_steps, 1):
        badge = Table([[Paragraph(str(i), s["number"])]], colWidths=[25], rowHeights=[25])
        badge.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), AUBERGINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        setup_rows.append([badge, para(step, s["body"])])
    setup_table = Table(setup_rows, colWidths=[35, 300])
    setup_table.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -2), 0.35, GRID), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
    db_right = [
        Paragraph("JSON grammar", s["h2"]),
        para("Every flag and value is a separate quoted string. Numbers are quoted. Standalone flags have no value after them.", s["body"]),
        code_box('["--flag", "value", "--standalone-flag"]', s["code"], 320),
        Spacer(1, 12),
        Paragraph("OpenAI key", s["h2"]),
        para("Databricks does not prompt during a job. Never place the key in Parameters JSON. Store it in a Databricks secret scope and expose OPENAI_API_KEY to the Python task.", s["body"]),
        code_box("OPENAI_API_KEY={{secrets/<scope-name>/<secret-name>}}", s["code"], 320),
        Spacer(1, 8),
        para("Classic compute: configure the secret-backed environment variable on the compute. Serverless: use job-level environment variables when enabled; otherwise use a small secret-reading wrapper.", s["small"]),
    ]
    story.extend(
        [
            Paragraph("Databricks Jobs", s["section"]),
            Paragraph(
                "Configure two Python-script tasks. Each task receives its own JSON array in the Parameters textbox.",
                s["lede"],
            ),
            Table(
                [[[Paragraph("SETUP", s["h3_orange"]), setup_table], db_right]],
                colWidths=[content_width / 2, content_width / 2],
                style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 10), ("LEFTPADDING", (1, 0), (1, 0), 10), ("RIGHTPADDING", (1, 0), (1, 0), 0), ("LINEBEFORE", (1, 0), (1, 0), 0.8, GRID)]),
            ),
            Spacer(1, 15),
            callout(
                "Remember",
                "Local commands are ordinary CLI syntax. Databricks Python-script parameters are a JSON array of strings. Notebook tasks use a different key-value interface.",
                "purple",
                s,
                content_width,
            ),
            PageBreak(),
        ]
    )

    # Sampling
    story.extend(
        [
            Paragraph("Two sampling controls", s["section"]),
            Paragraph(
                "The same flag name appears at different stages, so read it in context: one changes what reaches Silver; the other creates a sampled Gold output.",
                s["lede"],
            ),
        ]
    )
    sample_cards = Table(
        [
            [
                [
                    Paragraph("BEFORE SILVER", s["h3_orange"]),
                    Paragraph("Sample the workload", s["h2"]),
                    code_box("--sample-size N  --sample-seed S", s["code"], 315),
                    Spacer(1, 9),
                    para("Selects N individual input messages before cleaning, enrichment, sentiment, or advanced NLP. Only those messages are written to Silver.", s["body"]),
                ],
                [
                    Paragraph("AT GOLD", s["h3_orange"]),
                    Paragraph("Create a sample output", s["h2"]),
                    code_box("--gold-level sample  --sample-size N  --random-state S", s["code"], 315),
                    Spacer(1, 9),
                    para("Creates a Gold output containing N individual messages selected from Silver. It does not limit any other Gold output.", s["body"]),
                ],
            ]
        ],
        colWidths=[content_width / 2, content_width / 2],
        style=TableStyle([("BACKGROUND", (0, 0), (0, 0), ORANGE_LIGHT), ("BACKGROUND", (1, 0), (1, 0), AUBERGINE_LIGHT), ("BOX", (0, 0), (0, 0), 1, ORANGE), ("BOX", (1, 0), (1, 0), 1, AUBERGINE), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 15), ("RIGHTPADDING", (0, 0), (-1, -1), 15), ("TOPPADDING", (0, 0), (-1, -1), 15), ("BOTTOMPADDING", (0, 0), (-1, -1), 15)]),
    )
    story.extend(
        [
            sample_cards,
            Spacer(1, 18),
            Table(
                [
                    [Paragraph("RAW / BRONZE", s["center"]), Paragraph("SILVER", s["center"]), Paragraph("GOLD", s["center"])],
                    [Paragraph("message sample", s["small"]), Paragraph("cleaned + enriched rows", s["small"]), Paragraph("selected analytical grain", s["small"])],
                ],
                colWidths=[content_width / 3] * 3,
                style=TableStyle([("BACKGROUND", (0, 0), (0, 0), ORANGE), ("BACKGROUND", (1, 0), (1, 0), AUBERGINE), ("BACKGROUND", (2, 0), (2, 0), GOLD), ("TEXTCOLOR", (0, 0), (2, 0), colors.white), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("GRID", (0, 0), (-1, -1), 0.7, GRID), ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)]),
            ),
            PageBreak(),
        ]
    )

    # Silver references
    silver_sections = [
        ("Silver reference 1/3", "Input, output, execution, and sampling", "How data enters, where it lands, and how the work is controlled."),
        ("Silver reference 2/3", "Residual classification", "Human review, reviewed overlays, and optional OpenAI classification."),
        ("Silver reference 3/3", "Sentiment transformer", "VADER/transformer selection and reproducible transformer settings."),
    ]
    for page_title, heading, lede in silver_sections:
        header, rows = extract_markdown_table(markdown, heading)
        story.extend(
            [
                Paragraph(page_title, s["section"]),
                Paragraph(f"<b>{escape(heading)}</b> - {escape(lede)}", s["lede"]),
                reference_table(header, rows, s, [155, 185, 230, 150]),
                PageBreak(),
            ]
        )

    # Advanced NLP gets its own page because it is intentionally dense.
    header, rows = extract_markdown_table(markdown, "Advanced NLP, topics, and emotion")
    story.extend(
        [
            Paragraph("Silver reference: advanced NLP", s["section"]),
            Paragraph(
                "Optional spaCy, topic-modeling, and transformer-emotion controls. Leave these disabled for the lightest CPU path.",
                s["lede"],
            ),
            reference_table(header, rows, s, [155, 185, 230, 150]),
            Spacer(1, 8),
            callout(
                "Programmatic-only local controls",
                "PipelineConfig also supports spacy_batch_size, include_token_details, topic_fit_sample_size, topic_batch_size, and validate_advanced_nlp. The local CLI does not expose them.",
                "green",
                s,
                content_width,
            ),
            PageBreak(),
        ]
    )

    # Gold references
    gold_header, gold_rows = extract_markdown_table(markdown, "Gold parameter reference")
    level_header, level_rows = extract_markdown_table(markdown, "Gold-level choices are:")
    story.extend(
        [
            Paragraph("Gold parameter reference", s["section"]),
            Paragraph(
                "Run the Gold entry point once for each analytical output you want. Local and Databricks expose the same grains with environment-specific I/O.",
                s["lede"],
            ),
            reference_table(gold_header, gold_rows, s, [155, 185, 230, 150]),
            PageBreak(),
            Paragraph("Choose a Gold grain", s["section"]),
            Paragraph("Each Gold level produces one row per selected analytical unit.", s["lede"]),
        ]
    )
    midpoint = (len(level_rows) + 1) // 2
    left_rows, right_rows = level_rows[:midpoint], level_rows[midpoint:]
    left = reference_table(level_header, left_rows, s, [145, 205])
    right = reference_table(level_header, right_rows, s, [145, 205])
    story.extend(
        [
            Table([[left, right]], colWidths=[content_width / 2, content_width / 2], style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (0, 0), 0), ("RIGHTPADDING", (0, 0), (0, 0), 8), ("LEFTPADDING", (1, 0), (1, 0), 8), ("RIGHTPADDING", (1, 0), (1, 0), 0)])),
            Spacer(1, 16),
            callout(
                "Fast selection rule",
                "Choose conversation, user, date, channel, or release for operational summaries. Choose language, residual, technical, topic, or entity for NLP distributions. Choose sample only when you want a message-level Gold extract.",
                "orange",
                s,
                content_width,
            ),
            PageBreak(),
        ]
    )

    # Final memory page
    rules = [
        ("LOCAL", "Ordinary PowerShell command. No JSON brackets."),
        ("DATABRICKS", "JSON array of strings. Quote numbers too."),
        ("STANDALONE FLAG", "One JSON item: \"--advanced-nlp\"."),
        ("SILVER SAMPLE", "--sample-seed controls pre-Silver message selection."),
        ("GOLD SAMPLE", "--random-state controls only --gold-level sample."),
        ("OPENAI KEY", "Environment variable or Databricks secret. Never CLI, JSON, Git, or audit output."),
    ]
    rule_rows = []
    for label, explanation in rules:
        rule_rows.append([rich_para(f'<font color="{ORANGE.hexval()}"><b>{escape(label)}</b></font>', s["h2"]), para(explanation, s["body"])])
    story.extend(
        [
            Paragraph("Six rules to remember", s["section"]),
            Paragraph("If nothing else stays in memory, keep these six distinctions.", s["lede"]),
            Table(
                rule_rows,
                colWidths=[165, content_width - 165],
                style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), PAPER), ("BOX", (0, 0), (-1, -1), 0.8, GRID), ("INNERGRID", (0, 0), (-1, -1), 0.35, GRID), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 12), ("RIGHTPADDING", (0, 0), (-1, -1), 12), ("TOPPADDING", (0, 0), (-1, -1), 9), ("BOTTOMPADDING", (0, 0), (-1, -1), 9)]),
            ),
            Spacer(1, 13),
            Table(
                [
                    [
                        [
                            rich_para(f'<font color="{GREEN.hexval()}"><b>Verify the installation</b></font>', s["h2"]),
                            para("Run the complete tracked test suite from the repository root.", s["body"]),
                        ],
                        code_box(r".\.venv\python.exe -m unittest discover -s tests -v", s["code"], 390),
                    ]
                ],
                colWidths=[content_width - 410, 410],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), GREEN_LIGHT),
                        ("BOX", (0, 0), (-1, -1), 0.7, GREEN),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 12),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                        ("TOPPADDING", (0, 0), (-1, -1), 8),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ]
                ),
            ),
            Spacer(1, 13),
            callout(
                "Source of truth",
                "This guide is generated from tools/pdf_sources/local_and_databricks_parameters.md and the current pipeline interfaces. Rebuild it after parameter changes so the PDF and executable entry points remain aligned.",
                "purple",
                s,
                content_width,
            ),
        ]
    )

    doc.build(story, onFirstPage=draw_page, onLaterPages=draw_page)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build_pdf(args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
