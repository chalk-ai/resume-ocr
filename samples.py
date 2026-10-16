"""Generates the three fictional sample resumes the demo page offers.

Every person, company, school, email address and phone number here is invented. The third
sample is rasterized, skewed and speckled into an image-only PDF, so its text exists only as
pixels and can only be read back with OCR.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import pypdfium2 as pdfium
from PIL import Image, ImageFilter
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

PAGE_W, PAGE_H = letter
INK = HexColor("#1d2b28")
MUTED = HexColor("#56675f")
ACCENT = HexColor("#279467")


@dataclass(frozen=True)
class Sample:
    id: str
    title: str
    blurb: str
    pdf: bytes


class Writer:
    """A tiny top-down text cursor over a reportlab canvas."""

    def __init__(self, canvas: Canvas, x: float, y: float, width: float) -> None:
        super().__init__()
        self.c = canvas
        self.x = x
        self.y = y
        self.width = width

    def text(self, s: str, font: str = "Helvetica", size: float = 10, color=INK, gap: float = 4) -> None:
        self.c.setFont(font, size)
        self.c.setFillColor(color)
        self.c.drawString(self.x, self.y, s)
        self.y -= size + gap

    def wrapped(self, s: str, font: str = "Helvetica", size: float = 10, color=INK, bullet: bool = False) -> None:
        indent = 10 if bullet else 0
        words = s.split()
        line = ""
        first = True
        for w in words:
            trial = f"{line} {w}".strip()
            if self.c.stringWidth(trial, font, size) > self.width - indent:
                self._line(line, font, size, color, indent, bullet and first)
                first = False
                line = w
            else:
                line = trial
        if line:
            self._line(line, font, size, color, indent, bullet and first)

    def _line(self, s: str, font: str, size: float, color, indent: float, dot: bool) -> None:
        self.c.setFont(font, size)
        self.c.setFillColor(color)
        if dot:
            self.c.drawString(self.x, self.y, "•")
        self.c.drawString(self.x + indent, self.y, s)
        self.y -= size + 3

    def heading(self, s: str) -> None:
        self.y -= 6
        self.text(s.upper(), font="Helvetica-Bold", size=10, color=ACCENT, gap=3)
        self.c.setStrokeColor(ACCENT)
        self.c.setLineWidth(0.6)
        self.c.line(self.x, self.y + 9, self.x + self.width, self.y + 9)
        self.y -= 4

    def space(self, pts: float) -> None:
        self.y -= pts


def _job(w: Writer, title: str, org: str, dates: str, bullets: list[str]) -> None:
    w.c.setFont("Helvetica-Bold", 10.5)
    w.c.setFillColor(INK)
    w.c.drawString(w.x, w.y, title)
    w.c.setFont("Helvetica", 9.5)
    w.c.setFillColor(MUTED)
    w.c.drawRightString(w.x + w.width, w.y, dates)
    w.y -= 13
    w.text(org, font="Helvetica-Oblique", size=9.5, color=MUTED, gap=4)
    for b in bullets:
        w.wrapped(b, size=9.5, bullet=True)
    w.space(5)


def software_engineer() -> bytes:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=letter)
    c.setTitle("Avery Lindqvist - Resume")
    w = Writer(c, 54, PAGE_H - 64, PAGE_W - 108)
    w.text("Avery Lindqvist", font="Helvetica-Bold", size=24, gap=6)
    w.text("Senior Backend Engineer", size=12, color=ACCENT, gap=8)
    w.text(
        "Portland, OR  ·  avery.lindqvist@example.com  ·  (555) 014-2290  ·  github.com/avery-lq-example",
        size=9.5,
        color=MUTED,
    )
    w.heading("Summary")
    w.wrapped(
        "Backend engineer with eight years building low-latency data services. Led the move of a "
        + "payments ledger from nightly batch jobs to streaming, cutting reconciliation time from hours "
        + "to seconds. Comfortable owning systems from schema design to on-call."
    )
    w.heading("Experience")
    _job(
        w,
        "Senior Backend Engineer",
        "Northwind Payments, Portland, OR",
        "Mar 2021 – Present",
        [
            "Designed an event-sourced ledger in Go and Kafka processing 40k transactions per second.",
            "Cut p99 API latency from 220 ms to 35 ms by moving hot reads to a Redis-backed cache tier.",
            "Mentored five engineers and ran the backend interview loop.",
        ],
    )
    _job(
        w,
        "Software Engineer",
        "Fabrikam Logistics, Seattle, WA",
        "Jun 2017 – Feb 2021",
        [
            "Built the route-pricing service in Python and PostgreSQL used by 1,200 carriers.",
            "Introduced contract tests that reduced integration incidents by 60 percent.",
        ],
    )
    _job(
        w,
        "Software Engineering Intern",
        "Contoso Labs, Redmond, WA",
        "Summer 2016",
        ["Prototyped a telemetry pipeline that became the team's production metrics system."],
    )
    w.heading("Education")
    _job(w, "B.S. Computer Science", "University of Lakeshore", "2013 – 2017", ["Minor in Mathematics. GPA 3.8."])
    w.heading("Skills")
    w.wrapped("Go, Python, Rust, SQL, PostgreSQL, Kafka, Redis, Kubernetes, Terraform, gRPC, AWS")
    c.showPage()
    c.save()
    return buf.getvalue()


def data_scientist() -> bytes:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=letter)
    c.setTitle("Mateo Okonkwo-Reyes - Resume")
    sidebar_w = 178
    c.setFillColor(HexColor("#e6f4ec"))
    c.rect(0, 0, sidebar_w, PAGE_H, stroke=0, fill=1)

    s = Writer(c, 28, PAGE_H - 70, sidebar_w - 48)
    s.text("Mateo", font="Helvetica-Bold", size=17, gap=4)
    s.text("Okonkwo-Reyes", font="Helvetica-Bold", size=17, gap=10)
    s.text("Data Scientist", size=11, color=ACCENT, gap=16)
    s.heading("Contact")
    for line in ["mateo.or@example.com", "+1 555 017 4410", "Austin, TX", "linkedin.com/in/mateo-or-example"]:
        s.text(line, size=8.5, gap=5)
    s.heading("Skills")
    for line in [
        "Python, SQL, R",
        "PyTorch, XGBoost",
        "Causal inference",
        "A/B testing",
        "dbt, Airflow",
        "Snowflake, BigQuery",
    ]:
        s.text(line, size=8.5, gap=5)
    s.heading("Languages")
    for line in ["English (native)", "Spanish (fluent)", "Yoruba (conversational)"]:
        s.text(line, size=8.5, gap=5)

    w = Writer(c, sidebar_w + 30, PAGE_H - 70, PAGE_W - sidebar_w - 66)
    w.heading("Profile")
    w.wrapped(
        "Data scientist focused on pricing and experimentation. I turn messy marketplace data into "
        + "models that ship, and I care about whether the model changed the business metric, not just "
        + "the offline score."
    )
    w.heading("Experience")
    _job(
        w,
        "Lead Data Scientist",
        "Tailspin Marketplace, Austin, TX",
        "2022 – Present",
        [
            "Built a demand-forecasting model that raised fill rate 7 points across 30 metro areas.",
            "Designed the company's experimentation platform; 300+ experiments run in its first year.",
        ],
    )
    _job(
        w,
        "Data Scientist",
        "Woodgrove Insurance, Dallas, TX",
        "2019 – 2022",
        [
            "Shipped a gradient-boosted claims-fraud model that saved an estimated $4.1M per year.",
            "Replaced spreadsheet reporting with dbt models and a self-serve dashboard.",
        ],
    )
    w.heading("Education")
    _job(
        w,
        "M.S. Statistics",
        "Riverbend State University",
        "2017 – 2019",
        ["Thesis on heterogeneous treatment effects."],
    )
    _job(w, "B.A. Economics", "Riverbend State University", "2013 – 2017", [])
    w.heading("Certifications")
    w.wrapped("Snowflake SnowPro Core (2023)")
    c.showPage()
    c.save()
    return buf.getvalue()


def operations_manager_text() -> bytes:
    buf = io.BytesIO()
    c = Canvas(buf, pagesize=letter)
    w = Writer(c, 72, PAGE_H - 80, PAGE_W - 144)
    w.text("JUNE HARTWELL-OSEI", font="Courier-Bold", size=18, gap=8)
    w.text("Clinical Operations Manager", font="Courier", size=11, gap=6)
    w.text("12 Alder Lane, Burlington, VT 05401", font="Courier", size=10, gap=3)
    w.text("june.hartwell@example.com  /  555-019-8832", font="Courier", size=10, gap=10)

    def section(name: str) -> None:
        w.space(6)
        w.text(name, font="Courier-Bold", size=11, gap=6)

    def para(s: str) -> None:
        w.wrapped(s, font="Courier", size=10)

    section("OBJECTIVE")
    para("Registered nurse and operations lead seeking to run a multi-site outpatient network.")
    section("EXPERIENCE")
    w.text("Clinical Operations Manager, Green Mountain Health   2020-present", font="Courier-Bold", size=10, gap=4)
    para("- Ran scheduling and staffing for 4 outpatient clinics and 85 staff.")
    para("- Reduced average patient wait time from 34 to 12 minutes.")
    w.space(4)
    w.text("Charge Nurse, Lakeview General Hospital   2014-2020", font="Courier-Bold", size=10, gap=4)
    para("- Led a 22-bed medical-surgical unit across day and night shifts.")
    para("- Trained 30+ new graduate nurses on the unit's onboarding program.")
    section("EDUCATION")
    para("Master of Health Administration, Champlain Valley College, 2019")
    para("Bachelor of Science in Nursing, Champlain Valley College, 2014")
    section("LICENSES")
    para("Registered Nurse (RN), State of Vermont, active through 2027")
    para("Basic Life Support (BLS) certified")
    section("SKILLS")
    para("Staff scheduling, Epic EHR, budgeting, quality improvement, Lean")
    c.showPage()
    c.save()
    return buf.getvalue()


def scan(pdf: bytes, seed: int) -> bytes:
    """Rasterizes a PDF the way an office scanner would: grayscale, a slight skew, speckle, blur."""
    rng = random.Random(seed)
    page = pdfium.PdfDocument(pdf)[0]
    img = page.render(scale=150 / 72).to_pil().convert("L")
    img = img.rotate(rng.uniform(-1.2, 1.2), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=255)
    px = img.load()
    for _ in range(img.width * img.height // 180):
        x, y = rng.randrange(img.width), rng.randrange(img.height)
        px[x, y] = rng.choice((0, 70, 140, 200))
    img = img.filter(ImageFilter.GaussianBlur(0.6)).point(lambda v: int(30 + v * 0.86))
    out = io.BytesIO()
    img.save(out, format="PDF", resolution=150)
    return out.getvalue()


def build_samples() -> list[Sample]:
    return [
        Sample(
            "backend-engineer",
            "Single Column",
            "Single column, born-digital PDF",
            software_engineer(),
        ),
        Sample(
            "data-scientist",
            "Dual Column w/ Sidebar",
            "Two-column layout with a sidebar",
            data_scientist(),
        ),
        Sample(
            "scanned-nurse",
            "Scanned Page",
            "Scanned page: image only, skewed and noisy",
            scan(operations_manager_text(), seed=7),
        ),
    ]


if __name__ == "__main__":
    for sample in build_samples():
        path = f"{sample.id}.pdf"
        with open(path, "wb") as f:
            f.write(sample.pdf)
        print(path, len(sample.pdf))
